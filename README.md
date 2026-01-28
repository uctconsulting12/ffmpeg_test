ffmpeg_cmd = [
    "ffmpeg", "-y",

    # INPUT
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{w}x{h}",
    "-r", str(fps),
    "-i", "-",

    # ENCODER
    "-c:v", vcodec,                 # libx264
    "-preset", "veryfast",           
    "-pix_fmt", "yuv420p",

    # GOP (important for segment size)
    "-g", str(gop),
    "-keyint_min", str(gop),
    "-sc_threshold", "0",

    # RATE CONTROL (KVS-like)
    "-crf", "28",                    
    "-maxrate", f"{bitrate_k}k",
    "-bufsize", f"{bitrate_k * 2}k",

    # HLS → MP4 segments
    "-f", "hls",
    "-hls_time", str(hls_time),
    "-hls_list_size", str(hls_list_size),
    "-hls_segment_type", "fmp4",     
    "-hls_flags", "delete_segments+append_list",

    out_m3u8
]




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