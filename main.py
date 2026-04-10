import logging
import os
from datetime import datetime
from pathlib import Path

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


def _resolve_output_filename(output_filename: str | None) -> str:
    """Prepare file name for the processed file
    if output_filename is provided, add a timestamp as subfix to avoid the following issues:
    1. 2 filenames is the same
    2. we process the same file multiple times, and we want to keep all the processed files instead of overwriting them.

    Parameters
    ----------
    output_filename : str | None
        The output filename provided by the user. It can be None or a string.

    Returns
    -------
    str
        The resolved output filename with a timestamp subfix if output_filename is provided, otherwise just a timestamp as the filename.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_filename:
        base = Path(output_filename).stem
        return f"{base}_{timestamp}.mp4"
    return f"{timestamp}.mp4"


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Ultralytics FastAPI video API is running"}


@app.post("/process-video")
def process_video(video_path: str, output_filename: str | None = None):
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

    output_name = _resolve_output_filename(output_filename)
    output_path = OUTPUT_DIR / output_name

    logger.info("Processing video: %s", source_path)
    logger.info("Output path: %s", output_path)
    logger.info("Using tracker: %s", TRACKER_NAME)

    try:
        results = model.track(
            source=str(source_path),
            save=True,
            save_dir=str(OUTPUT_DIR),
            name=output_path.stem,
            classes=[0],
            conf=0.25,
            tracker=TRACKER_NAME,
        )

        if not results:
            raise RuntimeError("No track results returned")

        result = results[0]
        saved_path = getattr(result, "path", None)
        if saved_path:
            saved_path = Path(saved_path)
            if saved_path.is_file():
                output_path = saved_path
        elif output_path.exists():
            logger.info("Using fallback output path: %s", output_path)
        else:
            logger.warning("Expected output file does not exist after tracking: %s", output_path)

        logger.info("Completed processing: %s", output_path)
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
