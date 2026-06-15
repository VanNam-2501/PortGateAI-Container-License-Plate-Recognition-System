import os
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from app.schemas import BaseDetector, Detection, OBBBox

logger = logging.getLogger("SmartGatePipeline")

# Map custom trained YOLO OBB class names to pipeline internal target classes
CLASS_NAME_MAP = {
    "Container_Number": "iso_type",
    "ISO_Code": "container_code",
    "License_Plate": "plate"
}

class YoloDetector(BaseDetector):
    """Ultralytics YOLOv11-OBB Model Wrapper."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("detector", {})
        self.model_path = self.config.get("model_path", "models/yolo11n-obb.pt")
        self.device = self.config.get("device", "cuda:0")
        self.imgsz = self.config.get("imgsz", 1024)
        self.conf = self.config.get("conf", 0.5)
        self.iou = self.config.get("iou", 0.45)
        
        self.model = None
        self.mock_mode = False
        
        # Try to load real Ultralytics model
        try:
            from ultralytics import YOLO
            if os.path.exists(self.model_path):
                logger.info(f"Loading YOLO-OBB model from {self.model_path} on {self.device}")
                self.model = YOLO(self.model_path)
            else:
                logger.warning(f"YOLO model weights not found at {self.model_path}. Running in MOCK detector mode.")
                self.mock_mode = True
        except ImportError:
            logger.warning("ultralytics package not installed. Running in MOCK detector mode.")
            self.mock_mode = True

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detects objects in the image.
        
        Args:
            image: Input image.
            
        Returns:
            List of Detection objects.
        """
        if self.mock_mode or self.model is None:
            return self._generate_mock_detections(image)
            
        # Run real YOLO inference
        results = self.model.predict(
            source=image,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )
        
        detections: List[Detection] = []
        if not results:
            return detections
            
        result = results[0]
        
        # Check if OBB attributes exist
        if hasattr(result, "obb") and result.obb is not None:
            # xyxyxyxy shape: (N, 4, 2)
            xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy() if hasattr(result.obb.xyxyxyxy, "cpu") else result.obb.xyxyxyxy
            confidences = result.obb.conf.cpu().numpy() if hasattr(result.obb.conf, "cpu") else result.obb.conf
            class_ids = result.obb.cls.cpu().numpy() if hasattr(result.obb.cls, "cpu") else result.obb.cls
            class_names = result.names
            
            for i in range(len(xyxyxyxy)):
                cls_id = int(class_ids[i])
                class_name = class_names.get(cls_id, "unknown")
                # Map to internal pipeline class name
                class_name = CLASS_NAME_MAP.get(class_name, class_name)
                confidence = float(confidences[i])
                pts = xyxyxyxy[i]  # shape (4, 2)
                
                # Convert points to List of Tuple
                points_list = [(float(pt[0]), float(pt[1])) for pt in pts]
                
                detections.append(Detection(
                    class_name=class_name,
                    confidence=confidence,
                    obb=OBBBboxWrapper(points_list)
                ))
                
        return detections

    # def _generate_mock_detections(self, image: np.ndarray) -> List[Detection]:
    #     """Generates fake detections for development testing when no model weights are available."""
    #     h, w = image.shape[:2]
    #     detections = []
        
    #     # Mock Plate
    #     # Create a rotated box around the center-bottom of the image
    #     plate_pts = [
    #         (w * 0.45, h * 0.8),
    #         (w * 0.55, h * 0.81),
    #         (w * 0.54, h * 0.86),
    #         (w * 0.44, h * 0.85)
    #     ]
    #     detections.append(Detection(
    #         class_name="plate",
    #         confidence=0.89,
    #         obb=OBBBboxWrapper(plate_pts)
    #     ))
        
    #     # Mock Container Code
    #     # Create a rotated box on the side of the container (center-right)
    #     container_pts = [
    #         (w * 0.6, h * 0.4),
    #         (w * 0.9, h * 0.42),
    #         (w * 0.89, h * 0.5),
    #         (w * 0.59, h * 0.48)
    #     ]
    #     detections.append(Detection(
    #         class_name="container_code",
    #         confidence=0.92,
    #         obb=OBBBboxWrapper(container_pts)
    #     ))
        
    #     # Mock ISO Type
    #     # Create a small box near the container code (more square)
    #     iso_pts = [
    #         (w * 0.7, h * 0.52),
    #         (w * 0.74, h * 0.52),
    #         (w * 0.74, h * 0.57),
    #         (w * 0.7, h * 0.57)
    #     ]
    #     detections.append(Detection(
    #         class_name="iso_type",
    #         confidence=0.85,
    #         obb=OBBBboxWrapper(iso_pts)
    #     ))
        
    #     return detections


class OBBBboxWrapper(OBBBox):
    """Helper to instantiate OBBBox with points list."""
    def __init__(self, points: List[Tuple[float, float]]):
        super().__init__(points=points)
