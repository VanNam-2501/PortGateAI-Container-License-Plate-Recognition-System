import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

@dataclass
class OBBBox:
    """Oriented Bounding Box representation.
    points: List of 4 coordinate pairs [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] in clockwise order starting from top-left.
    """
    points: List[Tuple[float, float]]

@dataclass
class Detection:
    """Output of the detection stage."""
    class_name: str         # 'plate', 'container_code', 'iso_type'
    confidence: float       # Confidence score (0.0 to 1.0)
    obb: OBBBox             # Oriented bounding box details
    box_id: Optional[int] = None # ID tracking across frames (if tracker is used)

@dataclass
class RecognitionResult:
    """Output of the recognition (OCR) stage."""
    text: str
    confidence: float
    is_valid: bool = False
    validation_message: str = ""

@dataclass
class IngestionResult:
    """Output of the Ingestion stage."""
    image: np.ndarray
    timestamp: datetime.datetime
    camera_id: str
    lane_id: str
    quality_passed: bool
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    original_shape: Tuple[int, int, int] = (0, 0, 0)

@dataclass
class DetectionResult:
    """Output of the Detection stage."""
    ingestion: IngestionResult
    detections: List[Detection]
    latency_ms: float

@dataclass
class WarpedCrop:
    """Individual warped image crop of a detected object."""
    crop_image: np.ndarray
    detection: Detection
    lane_id: str
    camera_id: str
    timestamp: datetime.datetime

@dataclass
class TransformationResult:
    """Output of the Transformation (Warping) stage."""
    detection_result: DetectionResult
    crops: List[WarpedCrop]
    latency_ms: float

@dataclass
class RecognizedObject:
    """A single recognized object with location, raw crop image, and recognized text."""
    detection: Detection
    recognition: RecognitionResult
    crop_image: np.ndarray

@dataclass
class RecognitionStageResult:
    """Output of the Recognition stage."""
    transformation_result: TransformationResult
    recognized_objects: List[RecognizedObject]
    latency_ms: float

@dataclass
class TransactionResult:
    """Final output of the E2E pipeline after aggregation and validation."""
    status: str                         # "success", "partial", "failed"
    lane_id: str
    timestamp: datetime.datetime
    total_latency_ms: float
    plates: List[Dict[str, Any]]        # Combined plate results
    containers: List[Dict[str, Any]]    # Combined container results
    stages_latency: Dict[str, float]


class BaseDetector(ABC):
    """Abstract class for object detectors."""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detects objects of interest (plates, container codes, ISO types).
        
        Args:
            image: Input image as a numpy array.
            
        Returns:
            List of Detection objects.
        """
        pass


class BaseRecognizer(ABC):
    """Abstract class for text recognition (OCR) engines."""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        pass

    @abstractmethod
    def recognize(self, crop: np.ndarray) -> RecognitionResult:
        """Recognizes text within a cropped image.
        
        Args:
            crop: Warped image crop of the target object.
            
        Returns:
            RecognitionResult containing text, confidence, etc.
        """
        pass
