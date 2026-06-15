import datetime
import os
import cv2
import numpy as np
from typing import Dict, Any, Union, Tuple
from app.schemas import IngestionResult
from app.utils.image_quality import evaluate_image_quality

class IngestionStage:
    """Tầng 1: Ingestion - Tiền xử lý, Đọc dữ liệu, Ghi siêu dữ liệu và Kiểm tra chất lượng."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("quality_gate", {})
        self.enabled = self.config.get("enabled", True)
        self.blur_threshold = self.config.get("blur_threshold", 50.0)
        self.min_brightness = self.config.get("min_brightness", 40.0)
        self.max_brightness = self.config.get("max_brightness", 235.0)
        self.min_resolution = tuple(self.config.get("min_resolution", [640, 480]))

    def run(
        self, 
        image_input: Union[str, np.ndarray], 
        camera_id: str = "CAM01", 
        lane_id: str = "LANE01"
    ) -> IngestionResult:
        """Runs the ingestion stage.
        
        Args:
            image_input: File path to the image, or a pre-loaded numpy array.
            camera_id: Identifier of the camera.
            lane_id: Identifier of the lane.
            
        Returns:
            IngestionResult containing the loaded image, timestamp, and metrics.
        """
        # 1. Capture exact arrival timestamp in UTC+7 (Vietnam)
        tz_vietnam = datetime.timezone(datetime.timedelta(hours=7))
        timestamp = datetime.datetime.now(tz_vietnam)
        
        # 2. Read Image
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            # Read image using OpenCV
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Failed to decode image from path: {image_input}")
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
        else:
            raise TypeError("image_input must be a file path string or numpy ndarray")
            
        original_shape = image.shape
        
        # 3. Quality Gate
        quality_passed = True
        metrics = {}
        
        if self.enabled:
            quality_passed, metrics = evaluate_image_quality(
                image=image,
                blur_threshold=self.blur_threshold,
                min_brightness=self.min_brightness,
                max_brightness=self.max_brightness,
                min_res=self.min_resolution
            )
            
        return IngestionResult(
            image=image,
            timestamp=timestamp,
            camera_id=camera_id,
            lane_id=lane_id,
            quality_passed=quality_passed,
            quality_metrics=metrics,
            original_shape=original_shape
        )
