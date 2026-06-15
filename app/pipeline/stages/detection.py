import time
import os
import json
import logging
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from app.schemas import BaseDetector, IngestionResult, DetectionResult, Detection

logger = logging.getLogger("SmartGatePipeline")

class DetectionStage:
    """Tầng 2: Detection - Thực hiện định vị các vùng đối tượng (biển số, mã container, loại container)."""
    
    def __init__(self, detector: BaseDetector, config: Dict[str, Any]):
        self.detector = detector
        self.config = config.get("pipeline", {})
        self.conf_threshold = self.config.get("confidence_threshold", 0.5)
        self.target_classes = self.config.get("target_classes", ["plate", "container_code", "iso_type"])
        
        # ROI Configuration
        self.roi_config = config.get("roi", {})
        self.roi_enabled = self.roi_config.get("enabled", False)
        self.crop_roi = self.roi_config.get("crop_roi", True)
        self.roi_config_file = self.roi_config.get("config_file", "config/roi_config.json")

    def _get_roi_polygon(self, camera_id: str) -> Optional[np.ndarray]:
        """Loads the ROI polygon from the config json file if enabled."""
        if not self.roi_enabled or not os.path.exists(self.roi_config_file):
            return None
            
        try:
            with open(self.roi_config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            pts = None
            if camera_id in data:
                pts = data[camera_id]
            elif "polygon" in data:
                pts = data["polygon"]
                
            if pts and len(pts) >= 3:
                return np.array(pts, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error loading ROI configuration: {e}")
            
        return None

    def run(self, ingestion_result: IngestionResult) -> DetectionResult:
        """Runs detection on the ingested image, supporting ROI cropping and filtering.
        
        Args:
            ingestion_result: Result from the Ingestion stage.
            
        Returns:
            DetectionResult container.
        """
        start_time = time.perf_counter()
        
        # If image quality check failed, we can skip detection to save CPU/GPU cycles
        if not ingestion_result.quality_passed:
            return DetectionResult(
                ingestion=ingestion_result,
                detections=[],
                latency_ms=(time.perf_counter() - start_time) * 1000.0
            )
            
        polygon = self._get_roi_polygon(ingestion_result.camera_id)
        filtered_detections: List[Detection] = []
        
        if polygon is not None:
            img_h, img_w = ingestion_result.image.shape[:2]
            
            if self.crop_roi:
                # 1. Compute bounding box of the polygon ROI
                x_min = int(max(0, np.min(polygon[:, 0])))
                y_min = int(max(0, np.min(polygon[:, 1])))
                x_max = int(min(img_w, np.max(polygon[:, 0])))
                y_max = int(min(img_h, np.max(polygon[:, 1])))
                
                if x_max > x_min and y_max > y_min:
                    # 2. Crop the image to ROI bounding box
                    crop_image = ingestion_result.image[y_min:y_max, x_min:x_max].copy()
                    
                    # 3. Mask out region outside the polygon
                    mask = np.zeros(crop_image.shape[:2], dtype=np.uint8)
                    rel_poly = polygon - np.array([x_min, y_min])
                    cv2.fillPoly(mask, [rel_poly.astype(np.int32)], 255)
                    crop_image = cv2.bitwise_and(crop_image, crop_image, mask=mask)
                    
                    # 4. Run detector on the crop
                    raw_detections = self.detector.detect(crop_image)
                    
                    # 5. Map coordinates back to the original image coordinate system and filter
                    for det in raw_detections:
                        if det.class_name in self.target_classes and det.confidence >= self.conf_threshold:
                            # Map coordinates
                            mapped_points = []
                            for pt in det.obb.points:
                                mapped_points.append((pt[0] + x_min, pt[1] + y_min))
                            det.obb.points = mapped_points
                            
                            # Double check if centroid is within the ROI polygon
                            cx = sum(p[0] for p in det.obb.points) / 4.0
                            cy = sum(p[1] for p in det.obb.points) / 4.0
                            if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                                filtered_detections.append(det)
                else:
                    logger.warning("Invalid ROI bounding box coordinates. Running detector on full image.")
                    polygon = None
            else:
                # Run standard detector on full image, but filter based on polygon
                raw_detections = self.detector.detect(ingestion_result.image)
                for det in raw_detections:
                    if det.class_name in self.target_classes and det.confidence >= self.conf_threshold:
                        cx = sum(p[0] for p in det.obb.points) / 4.0
                        cy = sum(p[1] for p in det.obb.points) / 4.0
                        if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                            filtered_detections.append(det)
                            
        # Standard flow without ROI polygon
        if polygon is None:
            raw_detections = self.detector.detect(ingestion_result.image)
            for det in raw_detections:
                if det.class_name in self.target_classes and det.confidence >= self.conf_threshold:
                    filtered_detections.append(det)
                    
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        return DetectionResult(
            ingestion=ingestion_result,
            detections=filtered_detections,
            latency_ms=latency_ms
        )
