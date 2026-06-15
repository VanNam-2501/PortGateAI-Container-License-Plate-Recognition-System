#!/usr/bin/env python3
import cv2
import numpy as np
import os
import json
import argparse
import sys

# Ensure the root project directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def draw_roi_interactive(image_path_or_array, save_path="config/roi_config.json", camera_id=None):
    """Opens an interactive OpenCV window to let the user draw a polygon ROI.
    
    Args:
        image_path_or_array: Path to the image or loaded numpy image.
        save_path: Path to the json file where coordinates will be saved.
        camera_id: Key under which to save the ROI in the json.
    """
    if isinstance(image_path_or_array, str):
        if not os.path.exists(image_path_or_array):
            print(f"Error: Image path '{image_path_or_array}' does not exist.")
            return False
        img = cv2.imread(image_path_or_array)
        if img is None:
            print(f"Error: Could not decode image from '{image_path_or_array}'.")
            return False
    else:
        img = image_path_or_array.copy()

    h, w = img.shape[:2]
    points = []
    window_name = "Draw ROI (Press 's' to Save, 'q' to Cancel)"
    
    # Check if we already have a saved ROI for this camera or default
    if os.path.exists(save_path):
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            key = camera_id if camera_id else "polygon"
            existing_points = data.get(key, [])
            # Convert existing points to list of tuples and validate bounds
            for pt in existing_points:
                if len(pt) == 2:
                    px, py = int(pt[0]), int(pt[1])
                    if 0 <= px <= w and 0 <= py <= h:
                        points.append((px, py))
            if points:
                print(f"Loaded {len(points)} existing points from {save_path}.")
        except Exception as e:
            print(f"Warning: Could not read existing ROI from {save_path}: {e}")

    # Callback function for mouse click events
    def mouse_callback(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            # Limit coordinate bounds just in case
            points[-1] = (max(0, min(x, w - 1)), max(0, min(y, h - 1)))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Calculate sensible default window size
    aspect = w / h
    win_w = 1280
    win_h = int(win_w / aspect)
    cv2.resizeWindow(window_name, win_w, win_h)
    
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n" + "="*60)
    print(" INSTRUCTIONS FOR DRAWING REGION OF INTEREST (ROI)")
    print("="*60)
    print(" - LEFT CLICK: Add a point (vertex) for the ROI polygon")
    print(" - RIGHT CLICK / 'z': Undo the last point")
    print(" - 'c': Clear all points")
    print(" - 's': Save coordinates and exit")
    print(" - 'q' / ESC: Quit without saving changes")
    print("="*60 + "\n")

    while True:
        display_img = img.copy()
        
        # Draw UI details on the canvas
        overlay = display_img.copy()
        
        # Renders the ROI polygon
        if len(points) > 0:
            # Draw lines and circles
            for i in range(len(points)):
                cv2.circle(display_img, points[i], 5, (0, 0, 255), -1)
                if i > 0:
                    cv2.line(display_img, points[i-1], points[i], (0, 255, 255), 2)
            
            # Draw closing line if we have >= 3 points
            if len(points) >= 3:
                cv2.line(display_img, points[-1], points[0], (0, 255, 255), 2)
                # Fill polygon semi-transparently
                pts_arr = np.array(points, dtype=np.int32)
                cv2.fillPoly(overlay, [pts_arr], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, display_img, 0.7, 0, dst=display_img)
        
        # Add visual instructions text overlay at the top left of the screen
        text_bg = np.zeros_like(display_img[:40, :])
        cv2.putText(display_img, "Left Click: Add | Right Click/z: Undo | c: Clear | s: Save | q: Quit", 
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(20) & 0xFF
        
        # 'q' or Esc to Quit
        if key == 27 or key == ord('q'):
            print("ROI drawing canceled. No changes saved.")
            break
        # 'z' to Undo
        elif key == ord('z'):
            if points:
                points.pop()
        # 'c' to Clear
        elif key == ord('c'):
            points = []
            print("Cleared all points.")
        # 's' to Save
        elif key == ord('s'):
            if len(points) < 3 and len(points) > 0:
                print("Error: Polygon must have at least 3 points to form a region!")
                continue
            
            # Load existing config or create new
            config_data = {}
            if os.path.exists(save_path):
                try:
                    with open(save_path, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                except Exception:
                    pass
            
            save_key = camera_id if camera_id else "polygon"
            config_data[save_key] = [[pt[0], pt[1]] for pt in points]
            
            # Ensure folder exists
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2)
                
            print(f"Successfully saved ROI ({len(points)} points) to {save_path} under key '{save_key}':")
            print(config_data[save_key])
            break

    cv2.destroyAllWindows()
    return True

def main():
    parser = argparse.ArgumentParser(description="Interactive OpenCV ROI Drawer")
    parser.add_argument("--image", type=str, required=True, help="Path to image file to draw ROI on")
    parser.add_argument("--camera", type=str, default=None, help="Camera ID / Key to save the ROI under (e.g. CAM01)")
    parser.add_argument("--config", type=str, default="config/roi_config.json", help="Path to save JSON configuration")
    
    args = parser.parse_args()
    draw_roi_interactive(args.image, args.config, args.camera)

if __name__ == "__main__":
    main()
