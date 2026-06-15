"""
Cổng xác thực ảnh cắt trước OCR (Pre-OCR Validation Gate).
Kiểm tra 3 lớp: tỷ lệ khung hình, kích thước tối thiểu, chất lượng (blur + contrast).
"""
import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger("SmartGatePipeline")

# Giá trị mặc định nếu không có cấu hình
DEFAULT_GATE_CONFIG = {
    "enabled": True,
    "aspect_ratio": {
        "plate": [0.5, 5.0],
        "container_code": [2.0, 15.0],
        "iso_type": [0.5, 4.0],
    },
    "min_crop_size": {
        "plate": [20, 10],
        "container_code": [30, 8],
        "iso_type": [10, 8],
    },
    "crop_blur_threshold": 15.0,
    "crop_min_contrast": 10.0,
}


def validate_crop_for_ocr(
    crop_image: np.ndarray, 
    class_name: str, 
    config: Dict[str, Any] = None
) -> Tuple[bool, str]:
    """Kiểm tra ảnh cắt có đủ chất lượng để chạy OCR hay không.
    
    Kiểm tra theo thứ tự từ nhẹ → nặng tính toán:
    1. Tỷ lệ khung hình (Aspect Ratio)
    2. Kích thước tối thiểu (Min Size)
    3. Độ mờ (Blur - Laplacian variance)
    4. Độ tương phản (Contrast - Std deviation)
    
    Args:
        crop_image: Ảnh cắt đã warp (numpy array BGR).
        class_name: Tên lớp đối tượng ('plate', 'container_code', 'iso_type').
        config: Cấu hình pre_ocr_gate từ settings.yaml.
        
    Returns:
        Tuple[bool, str]: (passed, reason)
        - passed=True: Ảnh đạt, cho phép chạy OCR.
        - passed=False: Ảnh không đạt, reason giải thích lý do.
    """
    if config is None:
        config = DEFAULT_GATE_CONFIG
    
    if not config.get("enabled", True):
        return True, "Pre-OCR gate disabled"
    
    h, w = crop_image.shape[:2]
    
    # --- Lớp 1: Kiểm tra tỷ lệ khung hình ---
    aspect_ratio_limits = config.get("aspect_ratio", {}).get(class_name)
    if aspect_ratio_limits and h > 0:
        ar = w / h
        ar_min, ar_max = aspect_ratio_limits
        if ar < ar_min or ar > ar_max:
            reason = (
                f"Aspect ratio {ar:.2f} ngoài dải hợp lệ [{ar_min}, {ar_max}] "
                f"cho lớp '{class_name}' (kích thước: {w}x{h} px)"
            )
            logger.warning(f"Pre-OCR Gate REJECT: {reason}")
            return False, reason
    
    # --- Lớp 2: Kiểm tra kích thước tối thiểu ---
    min_size = config.get("min_crop_size", {}).get(class_name)
    if min_size:
        min_w, min_h = min_size
        if w < min_w or h < min_h:
            reason = (
                f"Kích thước {w}x{h} px nhỏ hơn tối thiểu {min_w}x{min_h} px "
                f"cho lớp '{class_name}'"
            )
            logger.warning(f"Pre-OCR Gate REJECT: {reason}")
            return False, reason
    
    # --- Lớp 3a: Kiểm tra độ mờ (Laplacian variance) ---
    blur_threshold = config.get("crop_blur_threshold", 15.0)
    gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY) if len(crop_image.shape) == 3 else crop_image
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    if blur_score < blur_threshold:
        reason = (
            f"Ảnh quá mờ: blur_score={blur_score:.2f} < ngưỡng {blur_threshold} "
            f"cho lớp '{class_name}'"
        )
        logger.warning(f"Pre-OCR Gate REJECT: {reason}")
        return False, reason
    
    # --- Lớp 3b: Kiểm tra độ tương phản ---
    min_contrast = config.get("crop_min_contrast", 10.0)
    contrast = float(np.std(gray))
    
    if contrast < min_contrast:
        reason = (
            f"Ảnh thiếu tương phản: std={contrast:.2f} < ngưỡng {min_contrast} "
            f"cho lớp '{class_name}' (có thể là vùng nền trơn)"
        )
        logger.warning(f"Pre-OCR Gate REJECT: {reason}")
        return False, reason
    
    logger.debug(
        f"Pre-OCR Gate PASS: class={class_name}, size={w}x{h}, "
        f"AR={w/h:.2f}, blur={blur_score:.1f}, contrast={contrast:.1f}"
    )
    return True, "passed"
