import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import subprocess
import time
from threading import Thread
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import socket

# ---------------- FASTAPI ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- STREAM CONFIG ----------------
RTSP_URL = os.environ.get(
    "RTSP_URL",
    "rtsp://admin:industry4@192.168.88.100:554/cam/realmonitor?channel=1&subtype=0"
)

RTMP_HOST = os.environ.get("RTMP_HOST", "mediamtx")
RTMP_PORT = int(os.environ.get("RTMP_PORT", 1935))
RTMP_URL = os.environ.get(
    "RTMP_URL",
    f"rtmp://{RTMP_HOST}:{RTMP_PORT}/stream"
)

# ---------------- YOLO (GPU) ----------------
model = YOLO("yolov8n.pt").to("cuda")
streaming = False

# ---------------- UTILS ----------------
def wait_for_rtmp(host, port, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            s = socket.socket()
            s.settimeout(1)
            s.connect((host, port))
            s.close()
            print(f"✅ RTMP server {host}:{port} reachable")
            return True
        except Exception:
            print(f"⏳ Waiting for RTMP server {host}:{port}...")
            time.sleep(1)
    print("❌ RTMP server not reachable")
    return False

# ---------------- STREAM LOOP ----------------
def stream_loop():
    global streaming

    if not wait_for_rtmp(RTMP_HOST, RTMP_PORT):
        streaming = False
        return

    while streaming:
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        wait_count = 0
        while not cap.isOpened() and streaming:
            print(f"⏳ Waiting for RTSP stream... ({wait_count})")
            time.sleep(1)
            wait_count += 1
            cap.open(RTSP_URL, cv2.CAP_FFMPEG)

        if not streaming:
            break

        # Get stream properties (fallback if zero)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS) or 15

        print(f"📷 RTSP Stream: {width}x{height} @ {fps} FPS")

        # FFmpeg command (libx264, safe presets)
        ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-loglevel", "error",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{width}x{height}",
    "-r", str(fps),  # match video fps
    "-i", "-",
    "-c:v", "libx264",
    "-preset", "ultrafast",  # faster encoding
    "-tune", "zerolatency",
    "-pix_fmt", "yuv420p",
    "-g", str(fps),
    "-f", "flv",
    RTMP_URL
]


        ffmpeg_out = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        try:
            while streaming:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Frame grab failed, reconnecting...")
                    break

                # YOLO inference
                results = model(frame, device=0, verbose=False)
                for r in results:
                    frame = r.plot()

                try:
                    ffmpeg_out.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    print("⚠️ RTMP connection lost, retrying...")
                    break

        except Exception as e:
            print("🔥 Stream error:", e)

        finally:
            cap.release()
            try:
                ffmpeg_out.stdin.close()
            except:
                pass
            ffmpeg_out.terminate()
            if streaming:
                print("⏱ Reconnecting in 3 seconds...")
                time.sleep(3)

    print("🛑 Stream stopped")

# ---------------- API ENDPOINTS ----------------
@app.post("/start")
def start_stream():
    global streaming
    if not streaming:
        streaming = True
        Thread(target=stream_loop, daemon=True).start()
        return {"status": "RTSP YOLO STREAM STARTED"}
    return {"status": "Already streaming"}

@app.post("/stop")
def stop_stream():
    global streaming
    streaming = False
    return {"status": "Streaming stopped"}
