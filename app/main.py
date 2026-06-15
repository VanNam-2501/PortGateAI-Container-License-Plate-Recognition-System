import os
import json
import uvicorn
import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

from app.config import STATIC_DIR, UPLOADS_DIR, OUTPUTS_DIR, RESULTS_DIR, ensure_runtime_dirs
from app.core.config import settings
from app.pipeline.gate_pipeline import SmartGatePipeline
from app.utils.logger import get_logger, setup_logging
from app.utils.video_simulator import VideoSimulator

# Setup logging
setup_logging()
logger = get_logger()

app = FastAPI(title="Smart Port Gate OCR Simulation API")

# Setup directories
ensure_runtime_dirs()

# Mount static files (so the dashboard can easily load results and uploaded videos)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

# Global Pipeline and Simulator instances
pipeline = SmartGatePipeline(config_path=settings.APP_CONFIG_PATH)
simulator = VideoSimulator(
    pipeline,
    static_dir=STATIC_DIR,
    results_dir=RESULTS_DIR,
    results_url_prefix="/outputs/results",
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error occurred: {exc}\n{traceback.format_exc()}")
    detail = str(exc) if settings.DEBUG else "Internal Server Error"
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": detail}
    )

class RoiConfig(BaseModel):
    polygon: List[List[int]]

@app.get("/", response_class=HTMLResponse)
async def read_index():
    """Serves the main single-page dashboard."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dashboard index.html not found! Please compile static files.</h1>")

@app.get("/api/stream")
async def get_stream():
    """MJPEG stream yielding processed display frames in real-time."""
    if not simulator.is_running:
        # If simulation isn't active, return a placeholder image or empty stream
        # Actually, it's better to yield a static frame saying "SIMULATION STOPPED"
        pass
        
    return StreamingResponse(
        simulator.get_frame_bytes(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/roi")
async def get_roi():
    """Fetches the current active ROI polygon configuration."""
    config_file = pipeline.config.get("roi", {}).get("config_file", "config/roi_config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read ROI config: {e}")
    return {"polygon": []}

@app.post("/api/roi")
async def save_roi(roi: RoiConfig):
    """Saves a new ROI polygon configuration and reloads it dynamically."""
    config_file = pipeline.config.get("roi", {}).get("config_file", "config/roi_config.json")
    try:
        os.makedirs(os.path.dirname(os.path.abspath(config_file)), exist_ok=True)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump({"polygon": roi.polygon}, f, indent=2)
        # Reload in both pipeline stages and simulator
        pipeline.detection_stage.roi_enabled = True
        simulator.load_roi()
        return {"status": "success", "polygon": roi.polygon}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save ROI config: {e}")

@app.post("/api/simulation/start")
async def start_simulation(video_path: str = Form(...)):
    """Starts the real-time simulation on a specific video file."""
    # Check if absolute path or inside uploads
    target_path = video_path
    if not os.path.isabs(target_path):
        target_path = os.path.join(UPLOADS_DIR, video_path)
        
    if not os.path.exists(target_path):
        raise HTTPException(status_code=400, detail=f"Video file not found: {video_path}")
        
    success = simulator.start(target_path)
    if success:
        return {"status": "started", "video": video_path}
    else:
        raise HTTPException(status_code=500, detail="Failed to start simulation.")

@app.post("/api/simulation/stop")
async def stop_simulation():
    """Stops the active video simulation."""
    simulator.stop()
    return {"status": "stopped"}

@app.get("/api/results")
async def get_results():
    """Returns the history of transaction results."""
    with simulator.results_lock:
        return simulator.results_history

@app.get("/api/status")
async def get_status():
    """Returns current simulator state for the dashboard."""
    with simulator.results_lock:
        results_count = len(simulator.results_history)
    return {
        "is_running": simulator.is_running,
        "state": simulator.state,
        "status_message": simulator.status_message,
        "tx_id": simulator.tx_id,
        "transaction_frames": len(simulator.transaction_frames),
        "empty_frames": simulator.consecutive_empty_frames,
        "results_count": results_count,
    }

@app.get("/api/videos")
async def list_videos():
    """Lists all uploaded test video files."""
    try:
        videos = []
        for file in os.listdir(UPLOADS_DIR):
            if file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.h264')):
                videos.append(file)
        return {"videos": videos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list videos: {e}")

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Uploads a test video file to the server uploads folder."""
    try:
        filename = file.filename
        dest_path = os.path.join(UPLOADS_DIR, filename)
        with open(dest_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"Uploaded video saved to: {dest_path}")
        return {"status": "success", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload video: {e}")

def main():
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
        workers=settings.MAX_WORKERS if not settings.DEBUG else 1
    )

if __name__ == "__main__":
    main()
