import time
import re
import logging
from typing import Dict, Any, List
from app.schemas import (
    BaseRecognizer, 
    TransformationResult, 
    RecognitionStageResult, 
    RecognizedObject, 
    RecognitionResult
)
from app.utils.validators.plate_validator import validate_plate, clean_plate_text, format_plate_text
from app.utils.validators.container_validator import validate_container_code, clean_container_text
from app.utils.validators.crop_validator import validate_crop_for_ocr
from app.utils.crop_preprocessor import preprocess_crop

logger = logging.getLogger("SmartGatePipeline")

class RecognitionStage:
    """Tầng 4: Recognition & Logic - Nhận diện ký tự và kiểm tra logic nghiệp vụ."""
    
    def __init__(self, recognizer: BaseRecognizer, config: Dict[str, Any]):
        self.recognizer = recognizer
        self.config = config
        self.gate_config = config.get("pre_ocr_gate", {})
        self.preprocess_config = config.get("preprocessing", {})
        self.correction_config = config.get("class_correction", {})

    def run(self, transformation_result: TransformationResult) -> RecognitionStageResult:
        """Runs OCR and validations on all warped crops.
        
        Pipeline cho mỗi ảnh cắt:
        1. Sửa lớp tự động trước OCR (Pre-OCR Correction) dựa trên Aspect Ratio
        2. Kiểm tra cổng xác thực (Pre-OCR Gate) → loại ảnh rác
        3. Tiền xử lý ảnh (Preprocessing) → nâng cao chất lượng
        4. Chạy OCR (Recognition) → nhận diện ký tự
        5. Sửa lớp tự động sau OCR (Post-OCR Correction) dựa trên text
        6. Hậu xử lý + Validation → kiểm tra logic nghiệp vụ
        
        Args:
            transformation_result: Result from the Transformation stage.
            
        Returns:
            RecognitionStageResult object.
        """
        start_time = time.perf_counter()
        recognized_objects: List[RecognizedObject] = []
        
        correction_enabled = self.correction_config.get("enabled", True)
        ar_opts = self.correction_config.get("aspect_ratio", {})
        
        for crop in transformation_result.crops:
            det = crop.detection
            class_name = det.class_name
            
            # ====== Bước 1: Sửa lớp tự động trước OCR (Pre-OCR Aspect Ratio Correction) ======
            if correction_enabled and crop.crop_image is not None:
                h, w = crop.crop_image.shape[:2]
                if h > 0:
                    ar = w / h
                    # Lấy cấu hình ngưỡng
                    iso_to_cont_th = ar_opts.get("iso_to_container_threshold", 4.2)
                    cont_to_iso_min = ar_opts.get("container_to_iso_min", 1.0)
                    cont_to_iso_max = ar_opts.get("container_to_iso_max", 3.8)
                    
                    # Quy tắc 1: Nếu YOLO nhận diện nhầm container_code thành iso_type do góc chụp hoặc nhầm lẫn
                    if class_name == "iso_type" and ar > iso_to_cont_th:
                        logger.info(
                            f"Pre-OCR Class Correction: Aspect ratio {ar:.2f} quá rộng cho 'iso_type'. "
                            f"Chuyển lớp từ 'iso_type' sang 'container_code'."
                        )
                        class_name = "container_code"
                        det.class_name = "container_code"  # Cập nhật trong object gốc
                        
                    # Quy tắc 2: Nếu YOLO nhận diện nhầm iso_type thành container_code
                    elif class_name == "container_code" and cont_to_iso_min <= ar <= cont_to_iso_max:
                        logger.info(
                            f"Pre-OCR Class Correction: Aspect ratio {ar:.2f} phù hợp với 'iso_type'. "
                            f"Chuyển lớp từ 'container_code' sang 'iso_type'."
                        )
                        class_name = "iso_type"
                        det.class_name = "iso_type"  # Cập nhật trong object gốc
            
            # ====== Bước 2: Cổng xác thực Pre-OCR Gate ======
            gate_passed, gate_reason = validate_crop_for_ocr(
                crop.crop_image, class_name, self.gate_config
            )
            
            if not gate_passed:
                # Ảnh không đạt → bỏ qua OCR, tạo kết quả rỗng
                logger.info(
                    f"Bỏ qua OCR cho '{class_name}' (conf={det.confidence:.2f}): {gate_reason}"
                )
                rec_result = RecognitionResult(
                    text="",
                    confidence=0.0,
                    is_valid=False,
                    validation_message=f"Skipped OCR: {gate_reason}"
                )
                recognized_objects.append(RecognizedObject(
                    detection=det,
                    recognition=rec_result,
                    crop_image=crop.crop_image
                ))
                continue
            
            # ====== Bước 2: Tiền xử lý ảnh (Preprocessing) ======
            h, w = crop.crop_image.shape[:2]
            is_square_plate = (class_name == "plate" and h > 0 and (w / h) < 1.8)
            
            if is_square_plate:
                logger.info(f"Detected 2-line square plate (AR={w/h:.2f}). Using dual-strategy OCR...")
                
                # Strategy 1: Try full-image OCR first (modern OCR engines can handle multi-line)
                full_prep = preprocess_crop(crop.crop_image, class_name, self.preprocess_config)
                full_rec = self.recognizer.recognize(full_prep, class_name=class_name)
                full_cleaned = re.sub(r'[^A-Z0-9]', '', full_rec.text.upper())
                
                # Check if full-image result looks like a valid plate
                full_is_valid, _ = validate_plate(full_cleaned)
                
                if full_is_valid and len(full_cleaned) >= 7:
                    # Full-image OCR succeeded
                    logger.info(f"Full-image OCR succeeded for square plate: '{full_rec.text}'")
                    raw_rec = full_rec
                else:
                    # Strategy 2: Split into 2 halves and concatenate
                    # Use adaptive split based on content analysis
                    # Top part (province + seri letter): ~45%, Bottom part (numbers): ~55%
                    split_ratio = 0.45
                    top_half = crop.crop_image[0:int(h * split_ratio), :]
                    bottom_half = crop.crop_image[int(h * split_ratio):, :]
                    
                    # Preprocess and recognize each half
                    top_prep = preprocess_crop(top_half, class_name, self.preprocess_config)
                    top_rec = self.recognizer.recognize(top_prep, class_name=class_name)
                    
                    bot_prep = preprocess_crop(bottom_half, class_name, self.preprocess_config)
                    bot_rec = self.recognizer.recognize(bot_prep, class_name=class_name)
                    
                    # Clean individual results before combining
                    top_clean = re.sub(r'[^A-Z0-9]', '', top_rec.text.upper())
                    bot_clean = re.sub(r'[^A-Z0-9]', '', bot_rec.text.upper())
                    
                    # Concatenate directly (no spaces) to form a single plate string
                    # e.g. top="51R" + bot="16694" → "51R16694"
                    combined_text = top_clean + bot_clean
                    combined_conf = (top_rec.confidence + bot_rec.confidence) / 2.0 if top_clean and bot_clean else max(top_rec.confidence, bot_rec.confidence, 0.0)
                    
                    logger.info(
                        f"Split OCR: top='{top_clean}', bottom='{bot_clean}', "
                        f"combined='{combined_text}', conf={combined_conf:.3f}"
                    )
                    
                    # Check which strategy gave a better result
                    split_is_valid, _ = validate_plate(combined_text)
                    if split_is_valid and len(combined_text) >= 7:
                        raw_rec = RecognitionResult(text=combined_text, confidence=combined_conf)
                    elif full_rec.text and len(full_cleaned) > len(combined_text):
                        # If neither is formally valid, prefer full OCR if it has more characters
                        raw_rec = full_rec
                    else:
                        raw_rec = RecognitionResult(text=combined_text, confidence=combined_conf)
            else:
                processed_image = preprocess_crop(
                    crop.crop_image, class_name, self.preprocess_config
                )
                
                # ====== Bước 3: Chạy OCR Engine ======
                raw_rec = self.recognizer.recognize(processed_image, class_name=class_name)
            
            logger.info(
                f"OCR '{class_name}': text='{raw_rec.text}', conf={raw_rec.confidence:.3f}"
            )
            
            # ====== Bước 5: Sửa lớp tự động sau OCR (Post-OCR Semantic Correction) ======
            text = raw_rec.text
            conf = raw_rec.confidence
            
            cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
            
            if correction_enabled:
                # Quy tắc 1 (Sửa về ISO Type): Nếu lớp hiện tại là container_code nhưng text có định dạng ISO Type
                if class_name == "container_code":
                    # Kiểm tra xem text có phải là ISO type hay không (4 ký tự: 2 số + 2 chữ/số)
                    is_iso_format = len(cleaned_text) == 4 and cleaned_text[0].isdigit() and cleaned_text[1].isdigit()
                    if is_iso_format:
                        logger.info(
                            f"Post-OCR Class Correction: Văn bản '{cleaned_text}' có dạng của 'iso_type'. "
                            f"Chuyển lớp từ 'container_code' sang 'iso_type'."
                        )
                        class_name = "iso_type"
                        det.class_name = "iso_type"
                        
                # Quy tắc 2 (Sửa về Container Code): Nếu lớp hiện tại là iso_type nhưng text có định dạng Container Code
                elif class_name == "iso_type":
                    # Kiểm tra xem text có độ dài >= 10 và bắt đầu bằng 3-4 chữ cái
                    is_container_format = len(cleaned_text) >= 10 and cleaned_text[:3].isalpha()
                    if is_container_format:
                        logger.info(
                            f"Post-OCR Class Correction: Văn bản '{cleaned_text}' có dạng của 'container_code'. "
                            f"Chuyển lớp từ 'iso_type' sang 'container_code'."
                        )
                        class_name = "container_code"
                        det.class_name = "container_code"
            
            # ====== Bước 6: Hậu xử lý + Validation nghiệp vụ ======
            is_valid = False
            val_msg = ""
            
            if class_name == "plate":
                # Clean and format license plate
                cleaned_text = clean_plate_text(text)
                is_valid, val_msg = validate_plate(cleaned_text)
                formatted_text = format_plate_text(cleaned_text) if is_valid else cleaned_text
                rec_result = RecognitionResult(
                    text=formatted_text,
                    confidence=conf,
                    is_valid=is_valid,
                    validation_message=val_msg
                )
            
            elif class_name == "container_code":
                # Clean and validate container ISO 6346 code
                cleaned_text = clean_container_text(text)
                is_valid, val_msg = validate_container_code(cleaned_text)
                rec_result = RecognitionResult(
                    text=cleaned_text,
                    confidence=conf,
                    is_valid=is_valid,
                    validation_message=val_msg
                )
                
            elif class_name == "iso_type":
                # ISO type is generally 4 characters representing container size/type (e.g. 45R1, 22G1)
                cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(cleaned_text) == 4 and cleaned_text[0].isdigit() and cleaned_text[1].isdigit():
                    is_valid = True
                    val_msg = "Valid 4-character ISO type format"
                else:
                    is_valid = False
                    val_msg = f"Invalid ISO type format: expected 4 characters (2 digits + 2 letters/digits), got '{cleaned_text}'"
                
                rec_result = RecognitionResult(
                    text=cleaned_text,
                    confidence=conf,
                    is_valid=is_valid,
                    validation_message=val_msg
                )
            else:
                # Fallback for unknown classes
                rec_result = RecognitionResult(
                    text=text,
                    confidence=conf,
                    is_valid=True,
                    validation_message="No validation rules for class"
                )
                
            recognized_objects.append(RecognizedObject(
                detection=det,
                recognition=rec_result,
                crop_image=crop.crop_image
            ))
            
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        return RecognitionStageResult(
            transformation_result=transformation_result,
            recognized_objects=recognized_objects,
            latency_ms=latency_ms
        )
