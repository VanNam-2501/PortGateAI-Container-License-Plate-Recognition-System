import os
import logging
import numpy as np
import cv2
from typing import Dict, Any
from app.schemas import BaseRecognizer, RecognitionResult

logger = logging.getLogger("SmartGatePipeline")


def _patch_paddlex_compatibility():
    """Patch PaddleX/PaddlePaddle compatibility issues.
    
    PaddleX 3.5.x calls config.set_optimization_level(3) which does not exist
    in PaddlePaddle 2.6.x. This monkey-patches the C++ binding class to add a
    no-op fallback so PaddleOCR 3.x can initialize on PaddlePaddle 2.6.x.
    
    Additionally patches enable_new_ir if missing, since PaddleX checks
    hasattr but the runner.py in PaddleX 3.5.2 also has code paths that
    may reference it.
    """
    try:
        import paddle.base.libpaddle as libpaddle
        AnalysisConfig = libpaddle.AnalysisConfig
        
        if not hasattr(AnalysisConfig, 'set_optimization_level'):
            logger.info("Patching AnalysisConfig.set_optimization_level (not available in paddle %s)",
                        _get_paddle_version())
            # Cannot set attributes directly on C++ pybind classes, so we
            # patch the runner source at import time instead.
            _patch_paddlex_runner_source()
    except Exception as e:
        logger.debug(f"Paddle compatibility patch skipped: {e}")


def _get_paddle_version():
    """Return installed paddlepaddle version string."""
    try:
        import paddle
        return paddle.__version__
    except Exception:
        return "unknown"


def _patch_paddlex_runner_source():
    """Patch PaddleX runner.py to guard set_optimization_level calls.
    
    This is idempotent — if already patched, it does nothing.
    """
    try:
        import paddlex
        runner_path = os.path.join(
            os.path.dirname(paddlex.__file__),
            "inference", "models", "runners", "paddle_static", "runner.py"
        )
        if not os.path.exists(runner_path):
            logger.debug(f"PaddleX runner.py not found at {runner_path}")
            return
        
        with open(runner_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check if already patched
        if 'hasattr(config, "set_optimization_level")' in content:
            logger.debug("PaddleX runner.py already patched.")
            return
        
        # Replace unguarded calls with guarded ones
        original = "                config.set_optimization_level(3)"
        replacement = '                if hasattr(config, "set_optimization_level"):\n                    config.set_optimization_level(3)'
        
        if original in content:
            patched = content.replace(original, replacement)
            with open(runner_path, "w", encoding="utf-8") as f:
                f.write(patched)
            logger.info("Successfully patched PaddleX runner.py for paddle compatibility.")
        else:
            logger.debug("PaddleX runner.py: no unguarded set_optimization_level found.")
    except Exception as e:
        logger.warning(f"Failed to auto-patch PaddleX runner.py: {e}")


class OcrRecognizer(BaseRecognizer):
    """Factory-like OCR Recognizer supporting multiple engines: mock, paddleocr, onnx."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("recognizer", {})
        self.engine_type = self.config.get("engine", "mock")
        
        self.engine = None
        self._paddleocr_version = None  # Track which PaddleOCR API version we loaded
        
        if self.engine_type == "paddleocr":
            self._init_paddleocr()
        elif self.engine_type == "onnx":
            self._init_onnx()
        else:
            logger.info("Initializing OCR in MOCK mode.")
            self.engine_type = "mock"

    def _init_paddleocr(self):
        # Step 1: Apply compatibility patches before importing PaddleOCR
        _patch_paddlex_compatibility()

        try:
            from paddleocr import PaddleOCR
            opts = self.config.get("paddleocr", {})
            logger.info("Loading PaddleOCR engine...")
            
            # Detect PaddleOCR version
            try:
                import paddleocr
                pocr_ver = paddleocr.__version__
                logger.info(f"PaddleOCR version: {pocr_ver}")
                major_ver = int(pocr_ver.split(".")[0])
            except Exception:
                major_ver = 3  # Assume 3.x if we can't detect
            
            if major_ver >= 3:
                self._init_paddleocr_v3(PaddleOCR, opts)
            else:
                self._init_paddleocr_v2(PaddleOCR, opts)
                
        except ImportError:
            logger.warning("paddleocr not installed. Falling back to MOCK OCR mode.")
            self.engine_type = "mock"
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            logger.warning("Falling back to MOCK OCR mode.")
            self.engine_type = "mock"
            self.engine = None

    def _init_paddleocr_v3(self, PaddleOCR, opts):
        """Initialize PaddleOCR 3.x (uses PaddleX internally)."""
        try:
            device = "gpu" if opts.get("use_gpu", True) else "cpu"
            self.engine = PaddleOCR(
                lang=opts.get("lang", "en"),
                device=device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            self._paddleocr_version = 3
            logger.info("PaddleOCR 3.x initialized successfully.")
        except Exception as e:
            logger.warning(f"PaddleOCR 3.x initialization failed: {e}")
            logger.info("Attempting PaddleOCR 2.x fallback...")
            self._init_paddleocr_v2(PaddleOCR, opts)

    def _init_paddleocr_v2(self, PaddleOCR, opts):
        """Initialize PaddleOCR 2.x."""
        try:
            self.engine = PaddleOCR(
                lang=opts.get("lang", "en"),
                use_gpu=opts.get("use_gpu", True),
                show_log=False,
                det=opts.get("det", False),
                rec=opts.get("rec", True),
                cls=opts.get("cls", False)
            )
            self._paddleocr_version = 2
            logger.info("PaddleOCR 2.x initialized successfully.")
        except Exception as e:
            logger.error(f"PaddleOCR 2.x initialization also failed: {e}")
            self.engine = None
            self.engine_type = "mock"
            logger.warning("Falling back to MOCK OCR mode.")

    def _init_onnx(self):
        try:
            import onnxruntime as ort
            opts = self.config.get("onnx", {})
            model_path = opts.get("model_path", "models/ocr_rec.onnx")
            
            if os.path.exists(model_path):
                logger.info(f"Loading ONNX OCR model from {model_path}")
                # We can configure execution providers (CUDA or CPU)
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.engine = ort.InferenceSession(model_path, providers=providers)
                
                # Load char dict keys
                char_dict_path = opts.get("char_dict_path", "models/ocr_keys.txt")
                self.chars = []
                if os.path.exists(char_dict_path):
                    with open(char_dict_path, "r", encoding="utf-8") as f:
                        self.chars = [line.strip("\r\n") for line in f.readlines()]
            else:
                logger.warning(f"ONNX OCR model weights not found at {model_path}. Falling back to MOCK OCR mode.")
                self.engine_type = "mock"
        except ImportError:
            logger.warning("onnxruntime not installed. Falling back to MOCK OCR mode.")
            self.engine_type = "mock"

    def recognize(self, crop: np.ndarray, class_name: str = None) -> RecognitionResult:
        """Runs recognition on the cropped image.
        
        Args:
            crop: Warped image crop.
            class_name: Optional class name to help mock/heuristic predictions.
            
        Returns:
            RecognitionResult.
        """
        if self.engine_type == "paddleocr" and self.engine is not None:
            return self._run_paddleocr(crop)
        elif self.engine_type == "onnx" and self.engine is not None:
            return self._run_onnx(crop)
        else:
            return self._run_mock(crop, class_name=class_name)

    def _run_paddleocr(self, crop: np.ndarray) -> RecognitionResult:
        try:
            if self._paddleocr_version == 3:
                return self._run_paddleocr_v3(crop)
            else:
                return self._run_paddleocr_v2(crop)
        except Exception as e:
            logger.error(f"PaddleOCR recognition failed: {e}")
            return RecognitionResult(text="", confidence=0.0)

    def _run_paddleocr_v3(self, crop: np.ndarray) -> RecognitionResult:
        """Run recognition using PaddleOCR 3.x API.
        
        PaddleOCR 3.x returns results via predict() which wraps PaddleX pipeline.
        The result objects have 'rec_texts' and 'rec_scores' attributes, or we can
        use the text_rec_model directly if available.
        """
        opts = self.config.get("paddleocr", {})
        use_det = opts.get("det", False)
        
        # Method 1: Use the text recognition model directly (bypass detection)
        # If use_det is enabled in configuration, we skip this to run the full detection + recognition pipeline.
        if not use_det and hasattr(self.engine, "paddlex_pipeline"):
            pipeline = self.engine.paddlex_pipeline
            
            # Try to access the text recognition sub-model directly
            text_rec_model = None
            if hasattr(pipeline, 'text_rec_model'):
                text_rec_model = pipeline.text_rec_model
            elif hasattr(pipeline, '_pipeline') and hasattr(pipeline._pipeline, 'text_rec_model'):
                text_rec_model = pipeline._pipeline.text_rec_model
            
            if text_rec_model is not None:
                try:
                    results = list(text_rec_model([crop]))
                    if results and len(results) > 0:
                        res = results[0]
                        # PaddleX model results are dict-like or have attributes
                        if hasattr(res, 'rec_text'):
                            text = res.rec_text
                            conf = getattr(res, 'rec_score', 0.0)
                        elif isinstance(res, dict):
                            text = res.get("rec_text", "")
                            conf = res.get("rec_score", 0.0)
                        else:
                            text = str(res)
                            conf = 0.0
                        return RecognitionResult(text=str(text), confidence=float(conf))
                except Exception as e:
                    logger.debug(f"Direct text_rec_model failed: {e}, falling back to predict()")

        # Method 2: Use the full predict() pipeline (includes detection + recognition)
        try:
            results = self.engine.predict(crop)
            if results and len(results) > 0:
                result = results[0]
                # PaddleOCR 3.x predict returns objects with rec_texts and rec_scores
                rec_texts = None
                rec_scores = None
                
                if hasattr(result, 'rec_texts'):
                    rec_texts = result.rec_texts
                    rec_scores = result.rec_scores
                elif isinstance(result, dict):
                    rec_texts = result.get('rec_texts', result.get('rec_text', []))
                    rec_scores = result.get('rec_scores', result.get('rec_score', []))
                
                if rec_texts:
                    # Concatenate all recognized text segments
                    if isinstance(rec_texts, list):
                        text = " ".join(str(t) for t in rec_texts)
                        conf = float(np.mean(rec_scores)) if rec_scores else 0.0
                    else:
                        text = str(rec_texts)
                        conf = float(rec_scores) if rec_scores else 0.0
                    return RecognitionResult(text=text.strip(), confidence=conf)
        except Exception as e:
            logger.debug(f"PaddleOCR 3.x predict() failed: {e}")
        
        return RecognitionResult(text="", confidence=0.0)

    def _run_paddleocr_v2(self, crop: np.ndarray) -> RecognitionResult:
        """Run recognition using PaddleOCR 2.x API."""
        logger.debug(f"Running PaddleOCR 2.x on crop of shape {crop.shape}")
        result = self.engine.ocr(crop, cls=False)
        logger.debug(f"PaddleOCR 2.x result: {result}")
        
        if result and result[0]:
            # PaddleOCR 2.x with det=False returns list of (text, conf) tuples
            if isinstance(result[0], (list, tuple)) and len(result[0]) == 2:
                text, conf = result[0]
                return RecognitionResult(text=str(text), confidence=float(conf))
            # PaddleOCR 2.x with det=True returns list of [box, (text, conf)]
            elif isinstance(result[0], list) and len(result[0]) > 0:
                texts = []
                confs = []
                for line in result[0]:
                    if isinstance(line, (list, tuple)) and len(line) == 2:
                        box, (text, conf) = line
                        texts.append(str(text))
                        confs.append(float(conf))
                if texts:
                    return RecognitionResult(
                        text=" ".join(texts),
                        confidence=float(np.mean(confs))
                    )
        return RecognitionResult(text="", confidence=0.0)

    def _run_onnx(self, crop: np.ndarray) -> RecognitionResult:
        try:
            # 1. Preprocess crop to match ONNX model input (usually reshape, normalize, gray/rgb)
            # This is a template placeholder logic for PyTorch CRNN / Paddle OCR ONNX models.
            # Usually input is Gray, shape: (1, 1, 32, 100) or similar.
            h, w = crop.shape[:2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            resized = cv2.resize(gray, (100, 32)) # Standard CRNN input size (W=100, H=32)
            
            # Normalize
            normalized = resized.astype(np.float32) / 255.0
            normalized = (normalized - 0.5) / 0.5  # Mean 0.5, Std 0.5
            
            # Add batch and channel dimensions
            input_tensor = np.expand_dims(normalized, axis=(0, 1)) # shape: (1, 1, 32, 100)
            
            # Get input names
            input_name = self.engine.get_inputs()[0].name
            
            # Run inference
            outputs = self.engine.run(None, {input_name: input_tensor})
            
            # Decode output indices to text (using CTC greedy decoder as typical for OCR)
            # outputs[0] usually has shape (sequence_length, batch_size, num_classes) or (batch_size, sequence_length, num_classes)
            preds = outputs[0]
            pred_indices = np.argmax(preds, axis=2)[0] # Shape: (seq_len,)
            
            # Simple CTC decoding (remove duplicates and blanks)
            decoded_text = ""
            prev_idx = -1
            blank_idx = len(self.chars) # Usually blank is the last class
            
            for idx in pred_indices:
                if idx != blank_idx and idx != prev_idx:
                    if idx < len(self.chars):
                        decoded_text += self.chars[idx]
                prev_idx = idx
                
            # Compute confidence dummy or mean probability
            probs = np.max(preds, axis=2)[0]
            conf = float(np.mean(probs))
            
            return RecognitionResult(text=decoded_text, confidence=conf)
        except Exception as e:
            logger.error(f"ONNX OCR inference failed: {e}")
            return RecognitionResult(text="", confidence=0.0)

    def _run_mock(self, crop: np.ndarray, class_name: str = None) -> RecognitionResult:
        """Returns mock OCR results depending on class_name or image characteristics (aspect ratio, etc.)."""
        if class_name == "plate":
            return RecognitionResult(text="59A-123.45", confidence=0.91)
        elif class_name == "container_code":
            return RecognitionResult(text="TGRU 892624-8", confidence=0.94)
        elif class_name == "iso_type":
            return RecognitionResult(text="45G1", confidence=0.88)

        h, w = crop.shape[:2]
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Heuristics to determine which class it likely is based on aspect ratio
        if aspect_ratio > 4.5:
            # Probably container code: e.g., "MSKU 123456 5"
            # Return slightly noisy text to test the container_validator clean/validate
            return RecognitionResult(text="MSKU 123456-5", confidence=0.94)
        elif 2.0 <= aspect_ratio <= 4.5:
            # Probably a long license plate or square plate
            # Return slightly noisy license plate
            return RecognitionResult(text="59A-123.45", confidence=0.91)
        else:
            # Probably ISO code (like "45R1", "22G1")
            return RecognitionResult(text="45R1", confidence=0.88)
