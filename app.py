import os
import uuid
import threading
import subprocess
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO
import shutil

app = FastAPI()

HLS_ROOT = "hls_out"
os.makedirs(HLS_ROOT, exist_ok=True)

# Serve HLS folders
app.mount("/hls", StaticFiles(directory=HLS_ROOT), name="hls")

# Load YOLO once (GPU will be used if available)
model = YOLO("yolov8n.pt")

# Keep track of running streams
streams = {}  # stream_id -&gt; {"thread":..., "stop": threading.Event(), "proc":..., "out_dir":...}

class StartRequest(BaseModel):
    source: str              # RTSP URL or file path
    use_nvenc: bool = True


def stream_worker(stream_id: str, source: str, stop_event: threading.Event,
                  use_nvenc: bool):

    out_dir = os.path.join(HLS_ROOT, stream_id)
    os.makedirs(out_dir, exist_ok=True)
    out_m3u8 = os.path.join(out_dir, "stream.m3u8")

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        # If you want, we can add an FFmpeg-based RTSP reader for better reliability
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps
    print(f"--------------------------------------------{fps}")
    vcodec = "libx264"
    bitrate_k=4000
    hls_time=4
    hls_list_size=100

    # Keyframe interval ~2 seconds for HLS
    gop = int(fps * 2)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",

        "-c:v", vcodec,
        "-pix_fmt", "yuv420p",
        "-g", str(gop),
        "-keyint_min", str(gop),

        "-b:v", f"{bitrate_k}k",
        "-maxrate", f"{int(bitrate_k * 1.1)}k",
        "-bufsize", f"{bitrate_k * 3}k",

        "-f", "hls",
        "-hls_time", str(hls_time),
        "-hls_list_size", str(hls_list_size),
        "-hls_flags", "delete_segments+append_list",
        out_m3u8
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    streams[stream_id]["proc"] = proc

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            annotated = results[0].plot()

            try:
                proc.stdin.write(annotated.tobytes())
            except BrokenPipeError:
                break
    finally:
        cap.release()
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass


@app.post("/streams/start")
def start_stream(req: StartRequest):
    stream_id = str(uuid.uuid4())
    stop_event = threading.Event()

    streams[stream_id] = {"stop": stop_event, "thread": None, "proc": None}

    t = threading.Thread(
        target=stream_worker,
        args=(stream_id, req.source, stop_event, req.use_nvenc),
        daemon=True
    )
    streams[stream_id]["thread"] = t
    t.start()

    return {
        "stream_id": stream_id,
        "hls_url": f"/hls/{stream_id}/stream.m3u8"
    }


@app.post("/streams/stop/{stream_id}")
async def stop_stream(stream_id: str):
    if stream_id not in streams:
        raise HTTPException(status_code=404, detail="stream_id not found")
    streams[stream_id]["stop"].wait()
    streams[stream_id]["stop"].set()

    
    proc = streams[stream_id].get("proc")
    if proc:
        try:
           await proc.terminate()
        except Exception:
            pass

      # delete directory named as stream_id
    dir_path = os.path.join("hls_out", stream_id) # or f"/app/hls_out/{stream_id}"

    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete directory: {e}")

    return {"status": "stopping", "stream_id": stream_id}