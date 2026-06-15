import cv2
import numpy as np
import time
import threading
import os
import json
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from app.pipeline.gate_pipeline import SmartGatePipeline
from app.schemas import DetectionResult, Detection, RecognitionStageResult, TransactionResult
from app.utils.image import perspective_warp

logger = logging.getLogger("SmartGatePipeline")

class Track:
    """Represents an active object track (e.g. plate or container) across frames."""
    def __init__(self, track_id: int, class_name: str, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int], frame: np.ndarray, yolo_conf: float, obb_pts: List[Tuple[float, float]]):
        self.track_id = track_id
        self.class_name = class_name
        self.centroid = centroid
        self.bbox = bbox
        self.last_seen = 0
        
        # Buffers for transaction processing
        self.frames: List[np.ndarray] = [frame.copy()]
        self.yolo_confs: List[float] = [yolo_conf]
        self.obb_points_list: List[List[Tuple[float, float]]] = [obb_pts]

class VideoSimulator:
    """Simulates a port gate camera feed using a video file, 
    implementing a presence-based state machine to capture frames and trigger
    one transaction after the ROI has truly cleared.
    """
    
    def __init__(
        self,
        pipeline: SmartGatePipeline,
        static_dir: str = "app/static",
        results_dir: Optional[str] = None,
        results_url_prefix: str = "/outputs/results",
    ):
        self.pipeline = pipeline
        self.static_dir = static_dir
        self.results_dir = results_dir or os.path.join(static_dir, "results")
        self.results_url_prefix = results_url_prefix.rstrip("/")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # State variables
        self.video_path: Optional[str] = None
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        
        # Stream frame storage
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        
        # State Machine States: "IDLE", "COLLECTING", "PROCESSING", "COOLDOWN"
        self.state = "IDLE"
        self.consecutive_empty_frames = 0
        self.max_empty_frames = 15       # Tolerate short detector dropouts before closing a vehicle (~0.6s at 25 FPS)
        self.max_transaction_frames = 30 # Cap useful detection frame buffer size; do not end a track by time alone
        self.cooldown_frames = 15        # Short cooldown (15 frames = ~0.6s) to let vehicle clear
        self.cooldown_counter = 0
        self.empty_rearm_frames = 5      # Extra empty frames needed only while waiting in IDLE/COOLDOWN
        self.new_object_min_frames = 3   # Require sustained movement before splitting a new vehicle
        self.new_object_jump_px = 220.0  # Same-class centroid distance that suggests another vehicle
        
        # Tracking & Transaction variables
        self.next_track_id = 0
        self.active_tracks: List[Track] = []
        self.tx_id: Optional[str] = None
        self.tx_start_time: Optional[str] = None
        self.entry_frame: Optional[np.ndarray] = None
        self.best_frame_score = 0.0
        self.transaction_frames: List[np.ndarray] = []
        self.transaction_camera_ids: List[str] = []
        self.transaction_frame_scores: List[float] = []
        self.last_centroids: Dict[str, Tuple[float, float]] = {}
        self.pending_new_object_frames: List[Tuple[np.ndarray, float, Dict[str, Tuple[float, float]]]] = []
        self.empty_frames_cooldown_idle = 0
        
        # Shared Results History
        self.results_history: List[Dict[str, Any]] = []
        self.results_lock = threading.Lock()
        
        # ROI Configuration
        self.roi_polygon: Optional[np.ndarray] = None
        self.load_roi()
        
        # Real-time UI info
        self.status_message = "Waiting..."
        self.last_detection_time = 0.0

    def load_roi(self) -> None:
        """Loads ROI from configuration."""
        try:
            config_file = self.pipeline.config.get("roi", {}).get("config_file", "config/roi_config.json")
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pts = data.get("polygon", [])
                if len(pts) >= 3:
                    self.roi_polygon = np.array(pts, dtype=np.int32)
                    logger.info(f"Simulator loaded ROI polygon with {len(pts)} points.")
                    return
            self.roi_polygon = None
        except Exception as e:
            logger.error(f"Error loading ROI in Simulator: {e}")
            self.roi_polygon = None

    def start(self, video_path: str) -> bool:
        """Starts the video simulation thread."""
        if self.is_running:
            self.stop()
            
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return False
            
        self.video_path = video_path
        self.is_running = True
        self.state = "IDLE"
        self.transaction_frames = []
        self.transaction_camera_ids = []
        self.transaction_frame_scores = []
        self.pending_new_object_frames = []
        self.cooldown_counter = 0
        self.status_message = "Waiting..."
        self.active_tracks = []
        self.last_centroids = {}
        self.empty_frames_cooldown_idle = 0
        self.load_roi()
        
        self.thread = threading.Thread(target=self._run_simulation, daemon=True)
        self.thread.start()
        logger.info(f"Started video simulation on: {video_path}")
        return True

    def stop(self) -> None:
        """Stops the video simulation thread."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        logger.info("Stopped video simulation.")

    def _is_point_in_roi(self, px: float, py: float) -> bool:
        """Checks if a point is inside the ROI polygon."""
        if self.roi_polygon is None:
            return True # If no ROI, everything is inside
        return cv2.pointPolygonTest(self.roi_polygon, (px, py), False) >= 0

    def _looks_like_new_object(self, class_centroids: Dict[str, Tuple[float, float]]) -> bool:
        """Returns True only for sustained same-class movement far from the active track."""
        if not self.last_centroids or not class_centroids:
            return False

        overlapping_classes = [
            cls for cls in class_centroids
            if cls in self.last_centroids and self.last_centroids[cls] is not None
        ]
        if not overlapping_classes:
            return False

        min_dist = float("inf")
        for cls in overlapping_classes:
            curr_c = class_centroids[cls]
            last_c = self.last_centroids[cls]
            dist = np.sqrt((curr_c[0] - last_c[0])**2 + (curr_c[1] - last_c[1])**2)
            min_dist = min(min_dist, dist)

        return min_dist > self.new_object_jump_px

    def _append_transaction_frame(
        self,
        frame: np.ndarray,
        score: float,
        class_centroids: Dict[str, Tuple[float, float]],
    ) -> None:
        """Appends a useful detection frame and keeps the best-scoring buffer."""
        self.consecutive_empty_frames = 0
        for cls, curr_centroid in class_centroids.items():
            self.last_centroids[cls] = curr_centroid

        if score > self.best_frame_score:
            self.best_frame_score = score
            self.entry_frame = frame.copy()

        self.transaction_frames.append(frame.copy())
        self.transaction_camera_ids.append("CAM01")
        self.transaction_frame_scores.append(score)

        if len(self.transaction_frames) > self.max_transaction_frames:
            drop_idx = min(
                range(len(self.transaction_frame_scores)),
                key=lambda idx: self.transaction_frame_scores[idx],
            )
            self.transaction_frames.pop(drop_idx)
            self.transaction_camera_ids.pop(drop_idx)
            self.transaction_frame_scores.pop(drop_idx)

    def _start_transaction(
        self,
        frame: np.ndarray,
        score: float,
        class_centroids: Dict[str, Tuple[float, float]],
    ) -> None:
        """Starts a new vehicle transaction from a detection frame."""
        self.state = "COLLECTING"
        self.tx_id = str(uuid.uuid4())[:8]
        self.tx_start_time = datetime.now().strftime("%H:%M:%S")
        self.entry_frame = frame.copy()
        self.best_frame_score = score
        self.transaction_frames = []
        self.transaction_camera_ids = []
        self.transaction_frame_scores = []
        self.consecutive_empty_frames = 0
        self.empty_frames_cooldown_idle = 0
        self.last_centroids = {}
        self.pending_new_object_frames = []
        self._append_transaction_frame(frame, score, class_centroids)
        self.status_message = f"Recording ({len(self.transaction_frames)}f)"
        logger.info(f"Transaction {self.tx_id} started at {self.tx_start_time} with score {self.best_frame_score:.2f}")

    def _start_transaction_from_pending(self) -> None:
        """Starts a new transaction with frames buffered while confirming a new vehicle."""
        pending_frames = self.pending_new_object_frames
        self.pending_new_object_frames = []
        if not pending_frames:
            return

        first_frame, first_score, first_centroids = pending_frames[0]
        self._start_transaction(first_frame, first_score, first_centroids)
        for frame, score, centroids in pending_frames[1:]:
            self._append_transaction_frame(frame, score, centroids)
        self.status_message = f"Recording ({len(self.transaction_frames)}f)"

    def _run_simulation(self) -> None:
        """Main loop reading frames from video."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.is_running = False
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        frame_delay = 1.0 / fps
        
        frame_idx = 0
        
        while self.is_running:
            start_time = time.perf_counter()
            ret, frame = cap.read()
            
            # Loop the video if it ends
            if not ret:
                if self.state == "COLLECTING" and self.transaction_frames:
                    logger.info(f"Video ended while collecting. Finalizing transaction {self.tx_id} before looping.")
                    self._trigger_transaction_processing()
                    self.state = "COOLDOWN"
                    self.cooldown_counter = self.cooldown_frames
                    self.last_centroids = {}
                    self.empty_frames_cooldown_idle = self.empty_rearm_frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue
                
            frame_idx += 1
            
            # Thread-safe frame copy for processing
            img_process = frame.copy()
            
            # Process state machine
            self._process_frame_state(img_process, frame_idx)
            
            # Render visual overlays on display frame
            display_frame = self._render_visuals(frame, img_process)
            
            # Update current stream frame
            with self.frame_lock:
                self.current_frame = display_frame
                
            # Maintain video frame rate
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0.001, frame_delay - elapsed)
            time.sleep(sleep_time)
            
        cap.release()

    def _process_frame_state(self, frame: np.ndarray, frame_idx: int) -> None:
        """Manages state transitions for one transaction per continuous ROI presence."""
        # 1. Handle Cooldown decay
        if self.state == "COOLDOWN":
            self.cooldown_counter -= 1
            self.status_message = f"Cooldown ({self.cooldown_counter}f)"
            if self.cooldown_counter <= 0:
                self.state = "IDLE"
                self.status_message = "Waiting..."
            # Note: We do NOT return/skip frame processing in Cooldown anymore!
            # This ensures we don't miss closely following vehicles.

        # 2. Run detection on every frame to avoid missing fast vehicles
        ingestion_res = self.pipeline.ingestion_stage.run(frame, camera_id="CAM01", lane_id="LANE01")
        detection_res = self.pipeline.detection_stage.run(ingestion_res)
        
        # 3. Filter detections inside the ROI polygon
        roi_detections = []
        for det in detection_res.detections:
            cx = sum(p[0] for p in det.obb.points) / 4.0
            cy = sum(p[1] for p in det.obb.points) / 4.0
            if self._is_point_in_roi(cx, cy):
                roi_detections.append(det)

        has_target = len(roi_detections) > 0
        
        # Calculate current centroids per class for detections in ROI
        class_centroids = {}
        class_detections = {}
        for det in roi_detections:
            cls = det.class_name
            cx = sum(p[0] for p in det.obb.points) / 4.0
            cy = sum(p[1] for p in det.obb.points) / 4.0
            if cls not in class_detections:
                class_detections[cls] = []
            class_detections[cls].append((cx, cy))
            
        for cls, pts in class_detections.items():
            avg_x = sum(p[0] for p in pts) / len(pts)
            avg_y = sum(p[1] for p in pts) / len(pts)
            class_centroids[cls] = (avg_x, avg_y)

        # Track empty frames when in IDLE or COOLDOWN to clear old vehicle memory
        if self.state in ["IDLE", "COOLDOWN"]:
            if has_target:
                self.empty_frames_cooldown_idle = 0
            else:
                self.empty_frames_cooldown_idle += 1
                if self.empty_frames_cooldown_idle >= self.empty_rearm_frames:
                    self.last_centroids = {}

        # 4. Handle State Machine Logic
        if self.state == "IDLE" or self.state == "COOLDOWN":
            if has_target:
                # Start only after old ROI memory has been cleared by a real empty gap.
                if not self.last_centroids:
                    self._start_transaction(
                        frame,
                        sum(det.confidence for det in roi_detections),
                        class_centroids,
                    )
                
        elif self.state == "COLLECTING":
            if has_target:
                current_score = sum(det.confidence for det in roi_detections)
                if self._looks_like_new_object(class_centroids):
                    self.pending_new_object_frames.append((frame.copy(), current_score, class_centroids.copy()))
                    self.status_message = f"Confirming next vehicle ({len(self.pending_new_object_frames)}f)"
                    if len(self.pending_new_object_frames) >= self.new_object_min_frames:
                        logger.info(
                            f"New vehicle confirmed after {len(self.pending_new_object_frames)} frames. "
                            f"Finalizing transaction {self.tx_id} and starting next transaction."
                        )
                        self._trigger_transaction_processing()
                        self._start_transaction_from_pending()
                    return

                self.pending_new_object_frames = []
                self._append_transaction_frame(frame, current_score, class_centroids)
            else:
                self.pending_new_object_frames = []
                self.consecutive_empty_frames += 1
                
            self.status_message = f"Recording ({len(self.transaction_frames)}f)"
            
            # End transaction criteria
            should_end = self.consecutive_empty_frames >= self.max_empty_frames
            
            if should_end:
                self._trigger_transaction_processing()
                
                # The ROI has already been empty for max_empty_frames, so the next
                # visible object can start a new transaction without stale memory.
                self.state = "COOLDOWN"
                self.cooldown_counter = self.cooldown_frames
                self.last_centroids = {}
                self.empty_frames_cooldown_idle = self.empty_rearm_frames

    def _trigger_transaction_processing(self) -> None:
        """Helper to fire asynchronous OCR and voting aggregation processing."""
        self.state = "PROCESSING"
        self.status_message = "Analyzing..."
        
        # Optimize frame buffer: only send the top 12 highest-scoring frames
        # This keeps the voting pipeline fast and avoids GPU latency bottlenecks
        optimized_frames = list(self.transaction_frames)
        optimized_cam_ids = list(self.transaction_camera_ids)
        optimized_scores = list(self.transaction_frame_scores)

        if not optimized_frames and self.entry_frame is not None:
            optimized_frames = [self.entry_frame.copy()]
            optimized_cam_ids = ["CAM01"]
            optimized_scores = [self.best_frame_score]
        
        # If we collected a lot of frames, send the top-scoring detection frames.
        if len(optimized_frames) > 12:
            indices = sorted(
                range(len(optimized_frames)),
                key=lambda idx: optimized_scores[idx] if idx < len(optimized_scores) else 0.0,
                reverse=True,
            )[:12]
            indices.sort()
            optimized_frames = [optimized_frames[idx] for idx in indices]
            optimized_cam_ids = [optimized_cam_ids[idx] for idx in indices]
            
        threading.Thread(
            target=self._async_process_transaction, 
            args=(self.tx_id, self.tx_start_time, self.entry_frame, optimized_frames, optimized_cam_ids),
            daemon=True
        ).start()

    def _async_process_transaction(
        self, 
        tx_id: str, 
        start_time: str, 
        entry_frame: np.ndarray, 
        frames: List[np.ndarray], 
        camera_ids: List[str]
    ) -> None:
        """Processes transaction frames and runs OCR + aggregation stage."""
        try:
            logger.info(f"Processing transaction {tx_id} with {len(frames)} frames...")
            
            # Save the best entry frame image to disk now that transaction completed
            entry_filename = f"entry_{tx_id}.jpg"
            entry_path = os.path.join(self.results_dir, entry_filename)
            cv2.imwrite(entry_path, entry_frame)
            
            # Execute Pipeline End-to-End Transaction (Stages 1-5)
            # This runs detector -> warps -> OCR -> aggregation
            result, details = self.pipeline.process_transaction(
                frames=frames,
                camera_ids=camera_ids,
                lane_id="LANE01",
                return_details=True
            )
            
            # Save crops of the elements detected in this transaction
            # Keep the crop with the HIGHEST YOLO confidence for each class across all frames
            saved_crops = {}
            for stage_res in details:
                for obj in stage_res.recognized_objects:
                    cls_name = obj.detection.class_name
                    conf = obj.detection.confidence
                    
                    # If this class is not saved yet, OR if this detection has HIGHER confidence
                    # than the previously saved crop for this class, overwrite it!
                    if cls_name not in saved_crops or conf > saved_crops[cls_name]["yolo_conf"]:
                        pts = np.array(obj.detection.obb.points, dtype=np.float32)
                        try:
                            # Apply the geometry helper to warp the image
                            warped = perspective_warp(stage_res.transformation_result.detection_result.ingestion.image, pts, padding_px=2)
                            
                            # Upscale crop for clearer visual display in UI (avoid browser stretching blur)
                            h_crop, w_crop = warped.shape[:2]
                            if h_crop > 0 and w_crop > 0:
                                target_h = 128 if cls_name == "plate" else 96
                                if h_crop < target_h:
                                    scale = target_h / h_crop
                                    warped = cv2.resize(
                                        warped, 
                                        (int(w_crop * scale), int(h_crop * scale)), 
                                        interpolation=cv2.INTER_CUBIC
                                    )
                                    
                            crop_filename = f"crop_{tx_id}_{cls_name}.jpg"
                            crop_path = os.path.join(self.results_dir, crop_filename)
                            cv2.imwrite(crop_path, warped)
                            saved_crops[cls_name] = {
                                "url": f"{self.results_url_prefix}/{crop_filename}",
                                "yolo_conf": conf
                            }
                        except Exception as e:
                            logger.error(f"Error saving crop in Simulator: {e}")

            # Assemble transaction result dict
            formatted_detections = []
            
            # Plates
            for plate in result.plates:
                crop_info = saved_crops.get("plate", {"url": None, "yolo_conf": 0.0})
                formatted_detections.append({
                    "class_name": "plate",
                    "display_name": "Biển số",
                    "crop_image_url": crop_info["url"],
                    "yolo_confidence": crop_info["yolo_conf"],
                    "ocr_text": plate.get("text", "Không rõ"),
                    "ocr_confidence": plate.get("confidence", 0.0),
                    "is_valid": plate.get("is_valid", False),
                    "validation_message": plate.get("validation_message", "")
                })
                
            # Containers
            for container in result.containers:
                # Container Code
                crop_code = saved_crops.get("container_code", {"url": None, "yolo_conf": 0.0})
                formatted_detections.append({
                    "class_name": "container_code",
                    "display_name": "Số Container",
                    "crop_image_url": crop_code["url"],
                    "yolo_confidence": crop_code["yolo_conf"],
                    "ocr_text": container.get("container_code", "Không rõ"),
                    "ocr_confidence": container.get("confidence", 0.0),
                    "is_valid": container.get("is_valid", False),
                    "validation_message": container.get("validation_message", "")
                })
                
                # ISO Type
                crop_iso = saved_crops.get("iso_type", {"url": None, "yolo_conf": 0.0})
                formatted_detections.append({
                    "class_name": "iso_type",
                    "display_name": "Loại ISO",
                    "crop_image_url": crop_iso["url"],
                    "yolo_confidence": crop_iso["yolo_conf"],
                    "ocr_text": container.get("iso_type", "Không rõ"),
                    "ocr_confidence": container.get("iso_confidence", 0.0),
                    "is_valid": container.get("is_valid", True),
                    "validation_message": ""
                })

            tx_result = {
                "id": tx_id,
                "timestamp": start_time,
                "entry_image_url": f"{self.results_url_prefix}/entry_{tx_id}.jpg",
                "detections": formatted_detections,
                "status": result.status
            }
            
            with self.results_lock:
                # Prepend to keep newest on top
                self.results_history.insert(0, tx_result)
                # Cap history at 50 results
                if len(self.results_history) > 50:
                    self.results_history.pop()
                    
            logger.info(f"Transaction {tx_id} finished processing. Status: {result.status}")
        except Exception as e:
            logger.error(f"Error processing transaction asynchronously: {e}", exc_info=True)

    def _render_visuals(self, display_frame: np.ndarray, process_frame: np.ndarray) -> np.ndarray:
        """Draws current ROI boundaries, state overlay, and real-time bounding boxes."""
        img = display_frame.copy()
        h, w = img.shape[:2]
        
        # 1. Draw ROI Polygon
        if self.roi_polygon is not None:
            overlay = img.copy()
            cv2.fillPoly(overlay, [self.roi_polygon], (0, 140, 255))
            cv2.addWeighted(overlay, 0.15, img, 0.85, 0, dst=img)
            cv2.polylines(img, [self.roi_polygon], isClosed=True, color=(0, 110, 220), thickness=2, lineType=cv2.LINE_AA)
            
            # Label ROI
            cv2.putText(img, "ROI Active", (self.roi_polygon[0][0] + 5, self.roi_polygon[0][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 110, 220), 1, cv2.LINE_AA)
            
        # 2. Draw Status Text Overlay at Top-Left
        text_bg = np.zeros_like(img[:50, :350])
        sub_img = img[:50, :350]
        cv2.addWeighted(text_bg, 0.7, sub_img, 0.3, 0, dst=sub_img)
        
        # State Color mapping
        state_colors = {
            "IDLE": (0, 255, 0),       # Green
            "COLLECTING": (0, 140, 255),# Orange
            "PROCESSING": (0, 0, 255),  # Red
            "COOLDOWN": (255, 100, 0)   # Blue
        }
        color = state_colors.get(self.state, (255, 255, 255))
        
        cv2.putText(img, f"STATE: {self.state}", (15, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(img, f"STATUS: {self.status_message}", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    
        return img

    def get_frame_bytes(self):
        """Thread-safe frame generator for MJPEG streaming."""
        while self.is_running:
            with self.frame_lock:
                if self.current_frame is None:
                    time.sleep(0.01)
                    continue
                ret, jpeg = cv2.imencode('.jpg', self.current_frame)
                
            if not ret:
                time.sleep(0.01)
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.04) # cap stream yield to ~25 FPS
