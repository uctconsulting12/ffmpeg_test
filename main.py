import os
import cv2
import subprocess
import time
from threading import Thread
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

# ---------------- ENV CONFIG ----------------
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

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

# ---------------- HLS CONFIG ----------------
HLS_DIR = os.environ.get("HLS_DIR", "./hls")
HLS_PLAYLIST = os.path.join(HLS_DIR, "stream.m3u8")
os.makedirs(HLS_DIR, exist_ok=True)

# ---------------- YOLO (GPU) ----------------
model = YOLO("yolov8n.pt").to("cuda")
streaming = False

# ---------------- MOUNT HLS ----------------
app.mount(
    "/hls",
    StaticFiles(directory=HLS_DIR, html=False),
    name="hls"
)

# ---------------- STREAM LOOP ----------------
def stream_loop():
    global streaming

    while streaming:
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Wait for stream
        wait_count = 0
        while not cap.isOpened() and streaming:
            print(f"⏳ Waiting for RTSP stream... ({wait_count})")
            time.sleep(1)
            wait_count += 1
            cap.open(RTSP_URL, cv2.CAP_FFMPEG)

        if not streaming:
            break

        # Stream properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = round(cap.get(cv2.CAP_PROP_FPS)) or 15

        print(f"📷 RTSP Stream: {width}x{height} @ {fps} FPS")

        # ---------------- FFMPEG PIPE ----------------
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "h264_nvenc",
            "-preset", "llhp",
            "-rc", "vbr_hq",
            "-b:v", "4M",
            "-maxrate", "6M",
            "-bufsize", "8M",
            "-pix_fmt", "yuv420p",
            "-fflags", "+genpts",
            "-err_detect", "ignore_err",
            "-vsync", "0",
            "-f", "hls",
            "-hls_time", "4",
            "-hls_list_size", "10",
            "-hls_flags", "delete_segments+append_list",
            "-hls_segment_filename", f"{HLS_DIR}/seg_%03d.ts",
            HLS_PLAYLIST
        ]

        ffmpeg_out = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        retry_count = 0
        try:
            while streaming:
                ret, frame = cap.read()
                if not ret:
                    retry_count += 1
                    if retry_count > 5:
                        print("Too many failed frames, reconnecting...")
                        break
                    time.sleep(0.1)
                    continue
                retry_count = 0

                # YOLO inference (non-blocking if using stream=True)
                results = model.predict(frame, device=0, verbose=False, batch=16)
                for r in results:
                    frame = r.plot()

                # Send to FFmpeg
                try:
                    ffmpeg_out.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    print("⚠️ FFmpeg pipe broken, restarting...")
                    break

        except Exception as e:
            print("🔥 Stream error:", e)

        finally:
            cap.release()
            try:
                ffmpeg_out.stdin.close()
            except Exception:
                pass
            ffmpeg_out.terminate()

            if streaming:
                print("Reconnecting in 3 seconds...")
                time.sleep(3)

    print("🛑 Stream stopped")

# ---------------- API ENDPOINTS ----------------
@app.post("/start")
def start_stream():
    global streaming
    if not streaming:
        streaming = True
        Thread(target=stream_loop, daemon=True).start()
        return {"status": "RTSP → YOLO → HLS STREAM STARTED"}
    return {"status": "Already streaming"}

@app.post("/stop")
def stop_stream():
    global streaming
    streaming = False
    return {"status": "Streaming stopped"}
