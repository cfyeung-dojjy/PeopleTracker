import logging
import os
from datetime import datetime
from pathlib import Path
import subprocess

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from ultralytics import YOLO

# Load .env values into environment variables before reading them.
load_dotenv()

app = FastAPI()


LOG_DIR = Path("log")
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
MODEL_WEIGHTS = os.getenv("YOLO_MODEL", "yolov8n.pt")
TRACKER_NAME = os.getenv("TRACKER_NAME", "bytetrack.yaml")
LOG_FILE = LOG_DIR / "app.log"

LOG_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

try:
    logger.info("Loading YOLO model: %s", MODEL_WEIGHTS)
    model = YOLO(MODEL_WEIGHTS)
    logger.info("YOLO model loaded successfully")
except Exception as exc:
    logger.exception("Failed to load YOLO model")
    model = None

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Ultralytics FastAPI video API is running"}

def avi_to_mp4(avi_path: Path) -> Path:
    """Convert .avi video to .mp4 using ffmpeg in the same directory as the input file.

    Assumes the filenames are unique

    Parameters
    ----------
    avi_path : Path
        Ultralytics saved .avi video

    Returns
    -------
    Path
        The converted .mp4 video path

    Raises
    ------
    subprocess.CalledProcessError
        If ffmpeg command fails
    """
    mp4_path = avi_path.with_suffix(".mp4")
    command = ["ffmpeg", "-i", str(avi_path), "-y", str(mp4_path)]
    subprocess.run(command, check=True)
    return mp4_path

@app.post("/process-video")
def process_video(video_path: str):
    if model is None:
        logger.error("Model is not available")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": "YOLO model failed to initialize"},
        )

    if not video_path or not isinstance(video_path, str):
        logger.warning("Missing or invalid video_path: %s", video_path)
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "`video_path` is required and must be a string"},
        )

    source_path = Path(video_path)
    if source_path.is_absolute():
        logger.warning("Absolute video_path is not allowed: %s", video_path)
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Absolute paths are not allowed. Use a path relative to input/."},
        )

    source_path = INPUT_DIR / source_path

    if not source_path.exists() or not source_path.is_file():
        logger.warning("Video file not found: %s", source_path)
        raise HTTPException(
            status_code=404,
            detail={"status": "error", "message": f"Video file not found: {source_path}"},
        )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / timestamp

    logger.info("Processing video: %s", source_path)
    logger.info("Output path: %s", output_dir)
    logger.info("Using tracker: %s", TRACKER_NAME)

    try:
        results = model.track(
            source=str(source_path),
            save=True,
            save_dir=output_dir,
            classes=[0],
            conf=0.25,
            # persist=True, # needed if we track the video frame-by-frame, i.e. from a streaming video source
            tracker=TRACKER_NAME,
        )

        if not results:
            raise RuntimeError("No track results returned")

        # Ultralytics saves the processed video as {save_dir}/{source_path.stem}.avi
        # We have to convert them to mp4 by ourselves
        output_path = output_dir / f"{source_path.stem}.avi"
        output_path = avi_to_mp4(output_path)

        return {
            "status": "success",
            "video_path": str(source_path.resolve()),
            "output_path": str(output_path.resolve()),
        }
    except Exception as exc:
        logger.exception("Error processing video %s", source_path)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to process video. See log/app.log for details.",
            },
        )
    except subprocess.CalledProcessError as e:
        logger.error("Failed to convert AVI to MP4: %s", e)
        return {
            "status": "avi file generated but failed to convert to mp4",
            "video_path": str(source_path.resolve()),
            "output_path": str(output_path.resolve()),
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
