import cv2
import subprocess
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from threading import Thread
import shutil
import numpy as np

from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- CONFIG ----------
HLS_DIR = "hls"
STREAM_NAME = "stream.m3u8"
VIDEO_SOURCE = "https://ai-search-video.s3.us-east-1.amazonaws.com/ai_search_videos/Vid.mp4"
# ----------------------------

os.makedirs(HLS_DIR, exist_ok=True)

# Load YOLO on GPU
model = YOLO("yolov8n.pt").to("cuda")

cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30

ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-f", "rawvideo",
    "-pix_fmt", "rgb24",          # INPUT format
    "-s", f"{width}x{height}",
    "-r", str(fps),
    "-i", "-",
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-tune", "zerolatency",
    "-g", str(fps),
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-pix_fmt", "yuv420p",        # OUTPUT format for HLS
    "-f", "hls",
    "-hls_time", "1",
    "-hls_list_size", "10",
    "-hls_flags", "delete_segments+append_list+temp_file",
    os.path.join(HLS_DIR, STREAM_NAME)
]

ffmpeg = None
streaming = False

def yolo_loop():
    global ffmpeg, streaming
    ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    streaming = True
    while streaming:
        ret, frame = cap.read()
        if not ret:
            print("Video ended or cannot read frame.")
            break

        # YOLO inference
        results = model(frame, device=0, verbose=False)
        for r in results:
            frame = r.plot()

        # Convert BGR -> RGB before sending to FFmpeg
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            ffmpeg.stdin.write(rgb_frame.tobytes())
        except BrokenPipeError:
            print("FFmpeg closed. Stopping loop.")
            break

    streaming = False
    if ffmpeg:
        ffmpeg.stdin.close()
        ffmpeg.terminate()
        ffmpeg = None

@app.post("/start")
def start_stream():
    global streaming
    if not streaming:
        Thread(target=yolo_loop, daemon=True).start()
        return {"status": "YOLO streaming started"}
    return {"status": "Already streaming"}

@app.post("/stop")
def stop_stream():
    global streaming, ffmpeg
    streaming = False
    if ffmpeg:
        ffmpeg.stdin.close()
        ffmpeg.terminate()
        ffmpeg = None

    # Clear HLS folder
    if os.path.exists(HLS_DIR):
        for filename in os.listdir(HLS_DIR):
            file_path = os.path.join(HLS_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    return {"status": "Streaming stopped and HLS folder cleared"}

# Serve HLS files
app.mount("/hls", StaticFiles(directory=HLS_DIR), name="hls")
# Serve frontend
app.mount("/", StaticFiles(directory=".", html=True), name="frontend")
