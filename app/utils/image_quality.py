import cv2
import numpy as np
from typing import Tuple, Dict, Any

def check_blur(image: np.ndarray) -> float:
    """Calculates the focus measure of the image using Laplacian variance.
    Higher values indicate sharper images. Lower values (e.g. < 50) indicate blurry images.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(variance)


def check_brightness(image: np.ndarray) -> float:
    """Calculates the average brightness of the image (0 to 255)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mean_val = np.mean(gray)
    return float(mean_val)


def check_resolution(image: np.ndarray) -> Tuple[int, int]:
    """Returns the width and height of the image."""
    height, width = image.shape[:2]
    return width, height


def evaluate_image_quality(
    image: np.ndarray,
    blur_threshold: float = 50.0,
    min_brightness: float = 40.0,
    max_brightness: float = 235.0,
    min_res: Tuple[int, int] = (640, 480)
) -> Tuple[bool, Dict[str, Any]]:
    """Evaluates multiple image quality factors and returns a pass/fail status
    with detailed metrics.
    """
    metrics = {}
    passed = True
    
    # 1. Check Resolution
    w, h = check_resolution(image)
    metrics["resolution"] = [w, h]
    if w < min_res[0] or h < min_res[1]:
        passed = False
        metrics["resolution_error"] = f"Resolution {w}x{h} below minimum required {min_res[0]}x{min_res[1]}"
        
    # 2. Check Blur (only if resolution passed to save computation)
    if passed:
        blur_val = check_blur(image)
        metrics["blur_score"] = blur_val
        if blur_val < blur_threshold:
            passed = False
            metrics["blur_error"] = f"Image is blurry. Score: {blur_val:.2f} (threshold: {blur_threshold})"
            
    # 3. Check Brightness
    if passed:
        brightness_val = check_brightness(image)
        metrics["brightness"] = brightness_val
        if brightness_val < min_brightness:
            passed = False
            metrics["brightness_error"] = f"Image too dark: {brightness_val:.2f} (min: {min_brightness})"
        elif brightness_val > max_brightness:
            passed = False
            metrics["brightness_error"] = f"Image too bright: {brightness_val:.2f} (max: {max_brightness})"
            
    return passed, metrics
