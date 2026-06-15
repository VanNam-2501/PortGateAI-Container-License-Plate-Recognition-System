import os
import yaml
import logging
from typing import Dict, Any, List, Union, Tuple
import numpy as np

from app.detectors.yolo_detector import YoloDetector
from app.ocr.paddle_ocr import OcrRecognizer
from app.schemas import TransactionResult, RecognitionStageResult

from app.pipeline.stages.ingestion import IngestionStage
from app.pipeline.stages.detection import DetectionStage
from app.pipeline.stages.transformation import TransformationStage
from app.pipeline.stages.recognition import RecognitionStage
from app.pipeline.stages.aggregation import AggregationStage

from app.utils.logger import get_logger

logger = get_logger("SmartGatePipeline")

class SmartGatePipeline:
    """🎯 Main Orchestrator for the Smart Port Gate ALPR & Container OCR System."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        logger.info("Initializing Smart Gate Pipeline modules...")
        
        # 1. Initialize Core Models (Adapters)
        self.recognizer = OcrRecognizer(self.config)
        self.detector = YoloDetector(self.config)
        
        # 2. Initialize Pipeline Stages
        self.ingestion_stage = IngestionStage(self.config)
        self.detection_stage = DetectionStage(self.detector, self.config)
        self.transformation_stage = TransformationStage(self.config)
        self.recognition_stage = RecognitionStage(self.recognizer, self.config)
        self.aggregation_stage = AggregationStage(self.config)
        
        logger.info("Pipeline initialized successfully.")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Loads configuration from YAML file."""
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found at {config_path}. Using default settings.")
            return {}
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error reading config file: {e}")
            return {}

    def process_single(
        self, 
        image_input: Union[str, np.ndarray], 
        camera_id: str = "CAM01", 
        lane_id: str = "LANE01",
        return_details: bool = False
    ) -> Union[TransactionResult, Tuple[TransactionResult, List[RecognitionStageResult]]]:
        """Processes a single frame through the pipeline and aggregates results immediately.
        
        Args:
            image_input: Path to image or loaded numpy array.
            camera_id: Identifier of camera source.
            lane_id: Identifier of lane.
            return_details: If True, returns both the TransactionResult and the intermediate RecognitionStageResult.
            
        Returns:
            TransactionResult JSON-ready dataclass, or Tuple of (TransactionResult, List[RecognitionStageResult]).
        """
        # Step 1: Ingestion
        ingestion_res = self.ingestion_stage.run(image_input, camera_id, lane_id)
        
        # Step 2: Detection
        detection_res = self.detection_stage.run(ingestion_res)
        
        # Step 3: Transformation
        transformation_res = self.transformation_stage.run(detection_res)
        
        # Step 4: Recognition
        recognition_res = self.recognition_stage.run(transformation_res)
        
        # Step 5: Aggregation (Run single-frame through aggregation for standard output)
        final_result = self.aggregation_stage.run([recognition_res])
        
        if return_details:
            return final_result, [recognition_res]
        return final_result

    def process_transaction(
        self, 
        frames: List[Union[str, np.ndarray]], 
        camera_ids: List[str] = None, 
        lane_id: str = "LANE01",
        return_details: bool = False
    ) -> Union[TransactionResult, Tuple[TransactionResult, List[RecognitionStageResult]]]:
        """Processes a transaction containing multiple frames (e.g. video frames or multi-camera views)
        and runs voting aggregation to produce a highly accurate combined output.
        
        Args:
            frames: List of images (file paths or numpy arrays).
            camera_ids: Corresponding camera IDs for each frame (optional).
            lane_id: Lane ID.
            return_details: If True, returns both the TransactionResult and the intermediate RecognitionStageResult list.
            
        Returns:
            TransactionResult JSON-ready dataclass, or Tuple of (TransactionResult, List[RecognitionStageResult]).
        """
        if not frames:
            raise ValueError("Frames list cannot be empty for transaction processing.")
            
        if camera_ids is None:
            camera_ids = [f"CAM{i:02d}" for i in range(1, len(frames) + 1)]
        elif len(camera_ids) != len(frames):
            # Pad or truncate camera IDs
            if len(camera_ids) < len(frames):
                camera_ids += [camera_ids[-1] if camera_ids else "CAM01"] * (len(frames) - len(camera_ids))
            else:
                camera_ids = camera_ids[:len(frames)]
                
        recognition_results: List[RecognitionStageResult] = []
        
        # Run Stages 1-4 for each frame
        for img_input, cam_id in zip(frames, camera_ids):
            try:
                ingestion_res = self.ingestion_stage.run(img_input, cam_id, lane_id)
                detection_res = self.detection_stage.run(ingestion_res)
                transformation_res = self.transformation_stage.run(detection_res)
                recognition_res = self.recognition_stage.run(transformation_res)
                recognition_results.append(recognition_res)
            except Exception as e:
                logger.error(f"Error processing frame from camera {cam_id}: {e}")
                
        # Run Stage 5: Aggregate results across all frames
        final_result = self.aggregation_stage.run(recognition_results)
        
        if return_details:
            return final_result, recognition_results
        return final_result
