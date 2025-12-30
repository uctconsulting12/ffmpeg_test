import cv2
import subprocess
import os
import time
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from threading import Thread
from queue import Queue, Full, Empty
from fastapi.middleware.cors import CORSMiddleware

# ---------------- FASTAPI ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- CONFIG ----------------
HLS_DIR = "hls"
STREAM_NAME = "stream.m3u8"
VIDEO_SOURCE = "https://ai-search-video.s3.us-east-1.amazonaws.com/ai_search_videos/Vid.mp4"

os.makedirs(HLS_DIR, exist_ok=True)

# ---------------- YOLO ----------------
model = YOLO("yolov8n.pt").to("cuda")

# ---------------- VIDEO CAPTURE ----------------
cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30

# ---------------- FRAME QUEUE ----------------
frame_queue = Queue(maxsize=100)

# ---------------- FFMPEG (LIVE HLS) ----------------
ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-fflags", "nobuffer",
    "-f", "rawvideo",
    "-pix_fmt", "rgb24",
    "-s", f"{width}x{height}",
    "-r", str(fps),
    "-i", "-",

    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-profile:v", "baseline",
    "-level", "3.0",
    "-pix_fmt", "yuv420p",

    "-g", str(fps),
    "-keyint_min", str(fps),
    "-sc_threshold", "0",

    "-f", "hls",
    "-hls_time", "1",
    "-hls_list_size", "10",
    "-hls_playlist_type", "event",
    "-hls_flags", "delete_segments+append_list+independent_segments",
    "-hls_allow_cache", "0",

    os.path.join(HLS_DIR, STREAM_NAME)
]

ffmpeg = None
streaming = False

# ---------------- VIDEO READER (REAL-TIME) ----------------
def reader_loop():
    global streaming
    frame_interval = 1.0 / fps

    while streaming:
        start = time.time()

        ret, frame = cap.read()

        # STOP WHEN VIDEO ENDS
        if not ret:
            streaming = False
            break

        try:
            frame_queue.put_nowait(frame)
        except Full:
            pass

        elapsed = time.time() - start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

# ---------------- YOLO + FFMPEG ----------------
def yolo_loop():
    global ffmpeg, streaming

    ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    while streaming or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=1)
        except Empty:
            continue

        results = model(frame, device=0, verbose=False)
        for r in results:
            frame = r.plot()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            ffmpeg.stdin.write(rgb_frame.tobytes())
        except BrokenPipeError:
            break

    if ffmpeg:
        ffmpeg.stdin.close()
        ffmpeg.terminate()
        ffmpeg = None

# ---------------- API ----------------
@app.post("/start")
def start_stream():
    global streaming

    if not streaming:
        streaming = True
        Thread(target=reader_loop, daemon=True).start()
        Thread(target=yolo_loop, daemon=True).start()
        return {"status": "YOLO HLS streaming started"}

    return {"status": "Already streaming"}

@app.post("/stop")
def stop_stream():
    global streaming, ffmpeg

    streaming = False

    if ffmpeg:
        ffmpeg.stdin.close()
        ffmpeg.terminate()
        ffmpeg = None

    for f in os.listdir(HLS_DIR):
        try:
            os.remove(os.path.join(HLS_DIR, f))
        except:
            pass

    return {"status": "Streaming stopped"}

# ---------------- STATIC FILES ----------------
app.mount("/hls", StaticFiles(directory=HLS_DIR), name="hls")
app.mount("/", StaticFiles(directory=".", html=True), name="frontend")
