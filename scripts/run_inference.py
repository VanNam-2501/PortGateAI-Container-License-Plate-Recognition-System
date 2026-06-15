#!/usr/bin/env python
import argparse
import json
import os
import sys
import glob
from dataclasses import asdict

# Ensure the root project directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline.gate_pipeline import SmartGatePipeline

def datetime_serializer(obj):
    """JSON serializer for datetime objects."""
    import datetime
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def main():
    parser = argparse.ArgumentParser(description="Smart Port Gate OCR Pipeline Inference CLI")
    parser.add_argument("--image", type=str, help="Path to a single image file")
    parser.add_argument("--folder", type=str, help="Path to a directory containing images")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Path to configuration file")
    parser.add_argument("--visualize", action="store_true", help="Draw bounding boxes and OCR results on images and save them")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save visualized images")
    parser.add_argument("--draw-roi", action="store_true", help="Draw or edit the ROI polygon interactively before running inference")
    
    args = parser.parse_args()
    
    if not args.image and not args.folder:
        parser.print_help()
        sys.exit(1)
        
    # Interactive ROI drawing
    config_path = args.config
    temp_config_created = False
    
    if args.draw_roi:
        draw_img = None
        if args.image:
            draw_img = args.image
        elif args.folder:
            # Find the first image to draw ROI on
            extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
            for ext in extensions:
                found = glob.glob(os.path.join(args.folder, ext)) + glob.glob(os.path.join(args.folder, ext.upper()))
                if found:
                    draw_img = found[0]
                    break
                    
        if draw_img:
            # Read ROI config file setting from main config
            roi_config_file = "config/roi_config.json"
            import yaml
            try:
                if os.path.exists(args.config):
                    with open(args.config, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f) or {}
                    roi_config_file = cfg.get("roi", {}).get("config_file", "config/roi_config.json")
            except Exception:
                pass
                
            print(f"Opening interactive ROI drawing tool on: {draw_img}")
            from scripts.draw_roi import draw_roi_interactive
            draw_roi_interactive(draw_img, save_path=roi_config_file)
            
            # Create a temporary config that forces 'roi.enabled: true' for this run
            import tempfile
            try:
                if os.path.exists(args.config):
                    with open(args.config, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f) or {}
                else:
                    cfg = {}
                    
                if "roi" not in cfg:
                    cfg["roi"] = {}
                cfg["roi"]["enabled"] = True
                
                # Write to temp file in config directory
                temp_fd, temp_path = tempfile.mkstemp(suffix=".yaml", prefix="settings_temp_", dir="config")
                os.close(temp_fd)
                with open(temp_path, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f)
                config_path = temp_path
                temp_config_created = True
                print("ROI dynamically enabled for this run.")
            except Exception as e:
                print(f"Warning: Could not enable ROI dynamically: {e}")
        else:
            print("Warning: --draw-roi provided but no images found to draw on.")

    try:
        # Initialize pipeline
        print(f"Initializing pipeline with config: {config_path}")
        pipeline = SmartGatePipeline(config_path=config_path)
        
        if args.image:
            print(f"\n--- Running inference on single image: {args.image} ---")
            try:
                if args.visualize:
                    result, stage_results = pipeline.process_single(args.image, return_details=True)
                    # Output to visualization folder
                    if args.output_dir:
                        os.makedirs(args.output_dir, exist_ok=True)
                        out_path = os.path.join(args.output_dir, "visualized_" + os.path.basename(args.image))
                    else:
                        # Save directly to the current working directory (abcproject root)
                        out_path = "visualized_" + os.path.basename(args.image)
                    try:
                        from app.utils.visualize import draw_pipeline_result
                        draw_pipeline_result(args.image, stage_results, out_path, config_path=config_path)
                    except Exception as ve:
                        print(f"Failed to generate visualization: {ve}")
                else:
                    result = pipeline.process_single(args.image)
                    
                # Print JSON formatted output
                print(json.dumps(asdict(result), default=datetime_serializer, indent=4, ensure_ascii=False))
            except Exception as e:
                print(f"Error processing image {args.image}: {e}", file=sys.stderr)
                
        elif args.folder:
            print(f"\n--- Running batch transaction inference on folder: {args.folder} ---")
            # Support common image formats
            extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
            image_paths = []
            for ext in extensions:
                image_paths.extend(glob.glob(os.path.join(args.folder, ext)))
                image_paths.extend(glob.glob(os.path.join(args.folder, ext.upper())))
                
            if not image_paths:
                print(f"No images found in folder: {args.folder}")
                sys.exit(1)
                
            print(f"Found {len(image_paths)} images. Grouping them as a single multi-frame transaction...")
            
            try:
                if args.visualize:
                    result, stage_results = pipeline.process_transaction(image_paths, return_details=True)
                    for i, (img_path, stage_res) in enumerate(zip(image_paths, stage_results)):
                        if args.output_dir:
                            os.makedirs(args.output_dir, exist_ok=True)
                            out_path = os.path.join(args.output_dir, "visualized_" + os.path.basename(img_path))
                        else:
                            out_path = os.path.join(os.path.dirname(img_path), "visualized_" + os.path.basename(img_path))
                        try:
                            from app.utils.visualize import draw_pipeline_result
                            draw_pipeline_result(img_path, [stage_res], out_path, config_path=config_path)
                        except Exception as ve:
                            print(f"Failed to generate visualization for {img_path}: {ve}")
                else:
                    result = pipeline.process_transaction(image_paths)
                print(json.dumps(asdict(result), default=datetime_serializer, indent=4, ensure_ascii=False))
            except Exception as e:
                print(f"Error processing transaction: {e}", file=sys.stderr)
    finally:
        if temp_config_created and os.path.exists(config_path):
            try:
                os.remove(config_path)
            except Exception:
                pass

if __name__ == "__main__":
    main()
