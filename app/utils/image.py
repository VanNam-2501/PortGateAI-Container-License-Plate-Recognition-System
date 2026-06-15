import cv2
import numpy as np
from typing import List, Tuple

def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """Sorts 4 points of an OBB in clockwise order:
    [top-left, top-right, bottom-right, bottom-left].
    
    Args:
        pts: A numpy array of shape (4, 2) containing coordinates.
        
    Returns:
        Sorted numpy array of shape (4, 2).
    """
    assert pts.shape == (4, 2), "Points array must have shape (4, 2)"
    
    # Sort points based on their x-coordinates
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    
    # Grab the left-most and right-most points
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    
    # Left-most points: top-left has smaller y-coordinate, bottom-left has larger
    left_most_sorted = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_most_sorted[0], left_most_sorted[1]
    
    # Right-most points: top-right has smaller y-coordinate, bottom-right has larger
    right_most_sorted = right_most[np.argsort(right_most[:, 1]), :]
    tr, br = right_most_sorted[0], right_most_sorted[1]
    
    return np.array([tl, tr, br, bl], dtype="float32")


def perspective_warp(image: np.ndarray, pts: np.ndarray, padding_px: int = 5) -> np.ndarray:
    """Applies perspective warp to crop a rotated bounding box and make it a flat rectangle.
    
    Args:
        image: Original high-resolution image.
        pts: Four coordinates of the OBB (shape: 4x2).
        padding_px: Outer padding in pixels around the bounding box.
        
    Returns:
        Warped and cropped image.
    """
    rect = order_points_clockwise(pts)
    (tl, tr, br, bl) = rect
    
    # Compute the width of the new image
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    # Compute the height of the new image
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    # Destination points for perspective transform
    # We add padding to avoid cutting off edge characters during OCR.
    dst = np.array([
        [padding_px, padding_px],
        [max_width + padding_px - 1, padding_px],
        [max_width + padding_px - 1, max_height + padding_px - 1],
        [padding_px, max_height + padding_px - 1]
    ], dtype="float32")
    
    # Add padding to original points as well (extrapolate outwards)
    # A simple way to do it is finding the center and expanding the coordinates slightly.
    center = np.mean(rect, axis=0)
    expanded_rect = rect.copy()
    
    # Determine scaling factor
    scale_x = (max_width + 2 * padding_px) / max_width if max_width > 0 else 1.0
    scale_y = (max_height + 2 * padding_px) / max_height if max_height > 0 else 1.0
    
    for i in range(4):
        vector = rect[i] - center
        expanded_rect[i] = center + vector * np.array([scale_x, scale_y])
        
    # Get the perspective transform matrix and warp the image
    m = cv2.getPerspectiveTransform(expanded_rect.astype("float32"), dst)
    warped = cv2.warpPerspective(image, m, (max_width + 2 * padding_px, max_height + 2 * padding_px))
    
    return warped
