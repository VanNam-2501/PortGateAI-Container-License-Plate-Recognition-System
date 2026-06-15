import cv2
import numpy as np
import os
import json
import yaml
from typing import List, Tuple, Union

from app.schemas import RecognitionStageResult, TransactionResult

def draw_text_with_bg(
    img: np.ndarray, 
    text: str, 
    x: int, 
    y: int, 
    text_color: Tuple[int, int, int], 
    accent_color: Tuple[int, int, int],
    anchor: str = 'bottom',
    bg_color: Tuple[int, int, int] = (20, 20, 20), 
    alpha: float = 0.85, 
    font_scale: float = 0.42, 
    thickness: int = 1
) -> int:
    """Draws text with a semi-transparent background, a small left accent bar, 
    and handles border bounds nicely.
    
    anchor: 'bottom' means y is the bottom edge of the label box.
            'top' means y is the top edge of the label box.
            
    Returns the height of the drawn label so we can stack them.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    pad_x = 6
    pad_y = 4
    
    # Calculate box height
    box_height = h + baseline + 2 * pad_y
    
    # Calculate y1 and y2 based on anchor
    if anchor == 'bottom':
        y2 = y
        y1 = y2 - box_height
    else: # anchor == 'top'
        y1 = y
        y2 = y1 + box_height
        
    x1 = x
    x2 = x + w + pad_x + 4 # 4px extra for accent bar
    
    img_h, img_w = img.shape[:2]
    
    # If the label goes out of right boundary, shift it left
    if x2 > img_w:
        shift_x = x2 - img_w + 5
        x1 -= shift_x
        x2 -= shift_x
        
    # Clip coordinates to image boundaries
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))
    
    # Draw background rectangle
    if x2 > x1 and y2 > y1:
        sub_img = img[y1:y2, x1:x2]
        bg_rect = np.full_like(sub_img, bg_color)
        cv2.addWeighted(bg_rect, alpha, sub_img, 1.0 - alpha, 0, dst=sub_img)
        
        # Draw accent bar
        cv2.rectangle(img, (x1, y1), (x1 + 3, y2), accent_color, -1)
        
    # Draw text
    text_x = x1 + 6
    text_y = y2 - baseline - pad_y
    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)
    
    return box_height + 2

def draw_pipeline_result(
    image: Union[str, np.ndarray], 
    stage_results: List[RecognitionStageResult], 
    output_path: str,
    config_path: str = "config/settings.yaml"
):
    """Draws bounding boxes, classes, OCR texts, and confidences on the image.
    
    Args:
        image: Path to the original image or numpy array (BGR).
        stage_results: List containing RecognitionStageResult for the frames.
        output_path: Path to save the visualized image.
        config_path: Path to the settings configuration file.
    """
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            print(f"Error: Could not load image from {image}")
            return
    else:
        img = image.copy()
        
    img_h, img_w = img.shape[:2]
    
    # Load and draw ROI polygon if enabled in configuration
    roi_polygon = None
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            roi_cfg = config.get("roi", {})
            if roi_cfg.get("enabled", False):
                roi_file = roi_cfg.get("config_file", "config/roi_config.json")
                if os.path.exists(roi_file):
                    with open(roi_file, "r", encoding="utf-8") as f:
                        roi_data = json.load(f)
                    
                    camera_id = "polygon"
                    if stage_results and len(stage_results) > 0:
                        try:
                            # Safely extract camera_id from stage results structure
                            camera_id = stage_results[0].transformation_result.detection_result.ingestion.camera_id
                        except AttributeError:
                            pass
                    
                    pts = None
                    if camera_id in roi_data:
                        pts = roi_data[camera_id]
                    elif "polygon" in roi_data:
                        pts = roi_data["polygon"]
                        
                    if pts and len(pts) >= 3:
                        roi_polygon = np.array(pts, dtype=np.int32)
    except Exception as e:
        print(f"Warning: Could not load ROI for visualization: {e}")

    if roi_polygon is not None:
        # Draw semi-transparent filled polygon overlay (orange/gold tint)
        roi_overlay = img.copy()
        cv2.fillPoly(roi_overlay, [roi_polygon], (0, 140, 255))
        cv2.addWeighted(roi_overlay, 0.15, img, 0.85, 0, dst=img)
        # Draw thicker dashed-style outline
        cv2.polylines(img, [roi_polygon], isClosed=True, color=(0, 110, 220), thickness=2, lineType=cv2.LINE_AA)
        
        # Draw ROI tag at first point of polygon
        label_x, label_y = int(roi_polygon[0][0]), int(roi_polygon[0][1])
        draw_text_with_bg(
            img, "ROI Active", label_x, label_y,
            text_color=(255, 255, 255), accent_color=(0, 110, 220),
            anchor='bottom', font_scale=0.38, thickness=1
        )
        
    # Nice BGR Colors for classes
    CLASS_COLORS = {
        "plate": (80, 220, 100),         # Emerald Green
        "container_code": (255, 150, 0), # Ocean Blue
        "iso_type": (0, 120, 255)        # Bright Amber/Orange
    }
    
    # Collect all objects and their bounding boxes
    flat_objects = []
    detection_boxes = []
    for res in stage_results:
        for obj in res.recognized_objects:
            flat_objects.append(obj)
            points = np.array(obj.detection.obb.points, dtype=np.int32)
            x_min = int(np.min(points[:, 0]))
            y_min = int(np.min(points[:, 1]))
            x_max = int(np.max(points[:, 0]))
            y_max = int(np.max(points[:, 1]))
            detection_boxes.append((x_min, y_min, x_max, y_max))
            
    # Keep track of already drawn label boxes to avoid drawing labels on top of other labels
    drawn_label_boxes = []
    
    for i, obj in enumerate(flat_objects):
        det = obj.detection
        recog = obj.recognition
        cls_name = det.class_name
        
        box_color = CLASS_COLORS.get(cls_name, (180, 180, 180))
        
        # Draw OBB (Oriented Bounding Box)
        points = np.array(det.obb.points, dtype=np.int32)
        cv2.polylines(img, [points], isClosed=True, color=box_color, thickness=2)
        
        # Prepare texts
        display_class = cls_name.replace("_", " ").title()
        cls_text = f"{display_class} ({det.confidence:.2f})"
        
        if recog.text:
            ocr_text = f"OCR: {recog.text}"
            if recog.confidence > 0:
                ocr_text += f" ({recog.confidence:.2f})"
        else:
            ocr_text = "OCR: None"
            
        x_min, y_min, x_max, y_max = detection_boxes[i]
        
        # Determine OCR text color based on validity
        ocr_text_color = (120, 250, 90) if recog.is_valid else (80, 160, 245) # Soft green / Soft orange
        ocr_accent_color = ocr_text_color
        
        # Measure label sizes to evaluate placement
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1
        pad_x = 6
        pad_y = 4
        
        (w_cls, h_cls), base_cls = cv2.getTextSize(cls_text, font, font_scale, thickness)
        (w_ocr, h_ocr), base_ocr = cv2.getTextSize(ocr_text, font, font_scale, thickness)
        
        box_w = max(w_cls, w_ocr) + pad_x + 4
        box_h_cls = h_cls + base_cls + 2 * pad_y
        box_h_ocr = h_ocr + base_ocr + 2 * pad_y
        total_h = box_h_cls + box_h_ocr + 2
        
        # Build other obstacles: other detection boxes + already drawn label boxes
        obstacles = [box for j, box in enumerate(detection_boxes) if j != i] + drawn_label_boxes
        
        # Evaluate ABOVE placement
        y2_above = y_min - 4
        y1_above = y2_above - total_h
        x1_above = x_min
        x2_above = x_min + box_w
        # Slide horizontal position if it overflows right
        if x2_above > img_w:
            shift = x2_above - img_w + 5
            x1_above -= shift
            x2_above -= shift
        above_box = (x1_above, y1_above, x2_above, y2_above)
        
        # Evaluate BELOW placement
        y1_below = y_max + 4
        y2_below = y1_below + total_h
        x1_below = x_min
        x2_below = x_min + box_w
        # Slide horizontal position if it overflows right
        if x2_below > img_w:
            shift = x2_below - img_w + 5
            x1_below -= shift
            x2_below -= shift
        below_box = (x1_below, y1_below, x2_below, y2_below)
        
        # Function to compute overlap penalty
        def get_penalty(label_box):
            lx1, ly1, lx2, ly2 = label_box
            penalty = 0
            
            # Check overlap with obstacles
            for ox1, oy1, ox2, oy2 in obstacles:
                ix1 = max(lx1, ox1)
                iy1 = max(ly1, oy1)
                ix2 = min(lx2, ox2)
                iy2 = min(ly2, oy2)
                if ix2 > ix1 and iy2 > iy1:
                    overlap_area = (ix2 - ix1) * (iy2 - iy1)
                    penalty += 20000 + overlap_area
                    
            # Off screen penalty
            if ly1 < 0:
                penalty += abs(ly1) * 10
            if ly2 > img_h:
                penalty += (ly2 - img_h) * 10
            if lx1 < 0:
                penalty += abs(lx1) * 10
            if lx2 > img_w:
                penalty += (lx2 - img_w) * 10
                
            return penalty
            
        above_penalty = get_penalty(above_box)
        below_penalty = get_penalty(below_box)
        
        # Choose the best placement
        if above_penalty <= below_penalty:
            # Draw above
            y_pos = y_min - 4
            ocr_height = draw_text_with_bg(
                img, ocr_text, x_min, y_pos, 
                text_color=ocr_text_color, accent_color=ocr_accent_color,
                anchor='bottom', font_scale=font_scale, thickness=thickness
            )
            y_pos -= ocr_height
            draw_text_with_bg(
                img, cls_text, x_min, y_pos, 
                text_color=(255, 255, 255), accent_color=box_color,
                anchor='bottom', font_scale=font_scale, thickness=thickness
            )
            # Record drawn label box
            drawn_label_boxes.append(above_box)
        else:
            # Draw below
            y_pos = y_max + 4
            cls_height = draw_text_with_bg(
                img, cls_text, x_min, y_pos, 
                text_color=(255, 255, 255), accent_color=box_color,
                anchor='top', font_scale=font_scale, thickness=thickness
            )
            y_pos += cls_height
            draw_text_with_bg(
                img, ocr_text, x_min, y_pos, 
                text_color=ocr_text_color, accent_color=ocr_accent_color,
                anchor='top', font_scale=font_scale, thickness=thickness
            )
            # Record drawn label box
            drawn_label_boxes.append(below_box)
            
    # Save the output image
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Saved visualization to: {output_path}")
