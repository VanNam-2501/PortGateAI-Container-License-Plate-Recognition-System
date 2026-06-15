"""
Pipeline tiền xử lý ảnh cắt trước OCR (Crop Preprocessing).
Gồm 5 bước: Grayscale → CLAHE → Denoise → Upscale → Binarize.
Mỗi bước có thể bật/tắt và tùy chỉnh theo từng lớp đối tượng.
"""
import cv2
import numpy as np
import logging
from typing import Dict, Any
from copy import deepcopy

logger = logging.getLogger("SmartGatePipeline")

# Cấu hình mặc định
DEFAULT_PREPROCESS_CONFIG = {
    "enabled": True,
    "clahe": {
        "enabled": True,
        "clip_limit": 2.0,
        "tile_grid_size": [8, 8],
    },
    "denoise": {
        "enabled": True,
        "method": "bilateral",
        "d": 5,
        "sigma_color": 75,
        "sigma_space": 75,
    },
    "upscale": {
        "enabled": True,
        "target_height": 64,
        "max_scale": 4.0,
    },
    "binarize": {
        "enabled": True,
        "block_size": 11,
        "constant": 2,
    },
    "class_overrides": {},
}


def _merge_config(base: Dict, overrides: Dict) -> Dict:
    """Merge class-specific overrides vào cấu hình mặc định (deep merge)."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Bước 1: Chuyển sang ảnh xám."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _apply_clahe(image: np.ndarray, config: Dict) -> np.ndarray:
    """Bước 2: Cân bằng tương phản cục bộ CLAHE."""
    clahe_cfg = config.get("clahe", {})
    if not clahe_cfg.get("enabled", True):
        return image
    
    clip_limit = clahe_cfg.get("clip_limit", 2.0)
    tile_size = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    result = clahe.apply(image)
    
    logger.debug(f"CLAHE applied: clip_limit={clip_limit}, tile={tile_size}")
    return result


def _denoise(image: np.ndarray, config: Dict) -> np.ndarray:
    """Bước 3: Khử nhiễu (Bilateral Filter hoặc fastNlMeans)."""
    denoise_cfg = config.get("denoise", {})
    if not denoise_cfg.get("enabled", True):
        return image
    
    method = denoise_cfg.get("method", "bilateral")
    
    if method == "bilateral":
        d = denoise_cfg.get("d", 5)
        sigma_color = denoise_cfg.get("sigma_color", 75)
        sigma_space = denoise_cfg.get("sigma_space", 75)
        result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        logger.debug(f"Bilateral filter: d={d}, σ_color={sigma_color}, σ_space={sigma_space}")
    elif method == "fastNlMeans":
        h = denoise_cfg.get("h", 10)
        template_size = denoise_cfg.get("template_window_size", 7)
        search_size = denoise_cfg.get("search_window_size", 21)
        result = cv2.fastNlMeansDenoising(image, None, h, template_size, search_size)
        logger.debug(f"fastNlMeans: h={h}")
    else:
        logger.warning(f"Phương pháp khử nhiễu không hợp lệ: '{method}', bỏ qua")
        return image
    
    return result


def _upscale_if_needed(image: np.ndarray, config: Dict) -> np.ndarray:
    """Bước 4: Phóng to ảnh nếu quá nhỏ."""
    upscale_cfg = config.get("upscale", {})
    if not upscale_cfg.get("enabled", True):
        return image
    
    target_height = upscale_cfg.get("target_height", 64)
    max_scale = upscale_cfg.get("max_scale", 4.0)
    
    h, w = image.shape[:2]
    
    if h >= target_height:
        return image
    
    scale = target_height / h
    scale = min(scale, max_scale)  # Giới hạn để tránh phóng quá to
    
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    result = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    logger.debug(f"Upscale: {w}x{h} → {new_w}x{new_h} (scale={scale:.2f}x)")
    return result


def _binarize(image: np.ndarray, config: Dict) -> np.ndarray:
    """Bước 5: Nhị phân hóa thích ứng (Adaptive Gaussian Thresholding)."""
    bin_cfg = config.get("binarize", {})
    if not bin_cfg.get("enabled", True):
        return image
    
    block_size = bin_cfg.get("block_size", 11)
    constant = bin_cfg.get("constant", 2)
    
    # block_size phải là số lẻ và >= 3
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3
    
    result = cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        constant
    )
    logger.debug(f"Binarize: block_size={block_size}, constant={constant}")
    return result


def preprocess_crop(
    crop_image: np.ndarray, 
    class_name: str = None, 
    config: Dict[str, Any] = None
) -> np.ndarray:
    """Chạy pipeline tiền xử lý ảnh cắt trước khi đưa vào OCR.
    
    Pipeline: Grayscale → CLAHE → Denoise → Upscale → Binarize
    
    Args:
        crop_image: Ảnh cắt đã warp (numpy array, BGR hoặc grayscale).
        class_name: Tên lớp đối tượng ('plate', 'container_code', 'iso_type') 
                    để áp dụng cấu hình riêng nếu có.
        config: Cấu hình preprocessing từ settings.yaml.
        
    Returns:
        Ảnh đã tiền xử lý (grayscale hoặc binary), sẵn sàng cho OCR.
    """
    if config is None:
        config = deepcopy(DEFAULT_PREPROCESS_CONFIG)
    
    if not config.get("enabled", True):
        return crop_image
    
    # Merge class-specific overrides nếu có
    effective_config = deepcopy(config)
    if class_name and "class_overrides" in config:
        overrides = config["class_overrides"].get(class_name, {})
        if overrides:
            effective_config = _merge_config(effective_config, overrides)
            logger.debug(f"Áp dụng class_overrides cho '{class_name}'")
    
    # Bước 1: Grayscale
    image = _to_grayscale(crop_image)
    
    # Bước 2: CLAHE
    image = _apply_clahe(image, effective_config)
    
    # Bước 3: Denoise
    image = _denoise(image, effective_config)
    
    # Bước 4: Upscale
    image = _upscale_if_needed(image, effective_config)
    
    # Bước 5: Binarize
    image = _binarize(image, effective_config)
    
    # Đảm bảo đầu ra luôn có 3 kênh màu (BGR) để tương thích với các bộ OCR (đặc biệt là PaddleOCR 3.x)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    logger.debug(
        f"Preprocessing hoàn tất cho '{class_name}': "
        f"output shape={image.shape}"
    )
    return image
