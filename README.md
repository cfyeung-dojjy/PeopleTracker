# Project Setup and Build Guide

<mark><b>Visit [here](https://drive.google.com/file/d/1Pb2eaShuP_8nBKSGXwkgqkYXcx-Cwb0p/view) for the processed MOT16-08 video</b></mark>

## Building the Docker Image

### Prerequisites

Before building the Docker image, you need to prepare the environment and download required model files.

The final image size should be ~10GB, mainly due to ultralytics requiring Pytorch and Opencv. I have found no way to reduce the image size by a meaningful amount. Except if we install the CPU-only pytorch packages.

### Step 1: Prepare the .env File

Create a `.env` file in the project root directory with the following configuration:

```bash
cp .env.example .env
```

Adjust the values according to your deployment environment.

### Step 2: Download Required Model Files

Although the assignment ask me to down the model weights within the container, I prefer downloading the weights in the source dir, and mount them to the container with read-only mode.

Download the necessary model files and place them in the models directory:

We use the latest YOLOv26x models by default, yolov26x for person detection and yolov26x-cls for ReID.

```bash
# Download model files to ./models directory
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt -O models/yolo26x.pt
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-cls.pt -O models/yolo26x-cls.pt
```

### Step 3: Build and run the Docker Image

Build the Docker image using the following command:

Make sure `nvidia-container-toolkit` is available on your platform.

```bash
docker compose up -d --build
```

## How to use this tool

1. Put the source video files under `input/`
2. Send an HTTP request to `http//:http://127.0.0.1:8000/process-video` along with the video filename, i.e. `MOT16-08-raw.mp4`
3. Alternatively , go to `http://127.0.0.1:8000/docs` and use the web interface there.
4. The API will save the outputs under `output/{timestamp}/`. It will first generate an .avi file (forced by Ultralytics), we then convert it to mp4 file using ffmpeg.

## Architectural Choices

I have chosen to build an API using FastAPI and Ultralytics.

With FastAPI, we could easily add an API endpoint for frame-by-frame inference, i.e. from a real time stream.

Ultralytics is a great library that provides out-of-the-box support for YOLO models and object tracking with ReID.

For this assignment, I used the largest and latest model available for the best performance during both person detection (yolo26x) and ReID (yolo26x-cls). nano models could be used if we are running the project within a device with limited computing power.

BoT-SORT is used because it is the only one that supports ReID in Ultralytics toolkit.
