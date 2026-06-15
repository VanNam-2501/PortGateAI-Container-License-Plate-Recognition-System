import time
import numpy as np
from typing import Dict, Any, List
from app.schemas import DetectionResult, TransformationResult, WarpedCrop
from app.utils.image import perspective_warp

class TransformationStage:
    """Tầng 3: Transformation - Cắt hình và biến đổi phối cảnh OBB thành hình chữ nhật phẳng 90 độ."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("transformation", {})
        self.padding_px = self.config.get("padding_px", 8)  # Padding around the cropped image for better OCR

    def run(self, detection_result: DetectionResult) -> TransformationResult:
        """Applies perspective warp transformation to all detections in the image.
        
        Args:
            detection_result: Result from the Detection stage.
            
        Returns:
            TransformationResult with cropped warped images.
        """
        start_time = time.perf_counter()
        crops: List[WarpedCrop] = []
        
        image = detection_result.ingestion.image
        camera_id = detection_result.ingestion.camera_id
        lane_id = detection_result.ingestion.lane_id
        timestamp = detection_result.ingestion.timestamp
        
        for det in detection_result.detections:
            pts = np.array(det.obb.points, dtype=np.float32)
            try:
                # Apply the geometry helper to warp the image
                warped_img = perspective_warp(image, pts, padding_px=self.padding_px)
                
                crops.append(WarpedCrop(
                    crop_image=warped_img,
                    detection=det,
                    lane_id=lane_id,
                    camera_id=camera_id,
                    timestamp=timestamp
                ))
            except Exception as e:
                # In production, we log the error but do not break the whole pipeline
                # because one bad detection shouldn't stop others from processing.
                pass
                
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        return TransformationResult(
            detection_result=detection_result,
            crops=crops,
            latency_ms=latency_ms
        )
