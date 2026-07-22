# fire-smoke
For each camera, every polling cycle:

Frame capture — ffmpeg connects to the camera's RTSP stream and pulls a single frame back as a JPEG.
YOLO detection — the frame is run through a custom-trained YOLO model (me (1).pt). Any box above the confidence threshold moves to the next stage.
Florence-2 captioning — for each qualifying box, the full frame is captioned by Florence-2 (<DETAILED_CAPTION> prompt), producing a natural language description of the scene.
Semantic verification — the caption is embedded with all-MiniLM-L6-v2 and compared (cosine similarity) against two banks of reference phrases (FIRE_PHRASES, SMOKE_PHRASES). If either max similarity clears SIMILARITY_THRESHOLD, the detection is accepted.
Decision — PASS detections (real fire/smoke) and REJECT detections (YOLO false positives Florence didn't confirm) are saved to separate folders with the box drawn on the frame, and every detection is logged to a CSV.
Requirements
bash
pip install ultralytics transformers sentence-transformers opencv-python-headless pillow psutil

ffmpeg must also be installed and on PATH (used for RTSP frame capture). Colab ships with it already; elsewhere:

bash
apt-get install -y ffmpeg

A CUDA GPU is strongly recommended — Florence-2 in particular is slow on CPU.

Configuration

All of the following are constants near the top of the script:

Constant	Purpose
RTSP_STREAMS	List of camera RTSP URLs to poll
POLL_INTERVAL_SECONDS	Delay between full sweeps of all cameras
YOLO_MODEL_PATH	Path to the trained YOLO weights file
YOLO_CONF_THRESHOLD	Minimum YOLO confidence to trigger Florence verification
SIMILARITY_THRESHOLD	Minimum cosine similarity (fire or smoke) to accept a detection
FIRE_PHRASES / SMOKE_PHRASES	Reference phrases the caption is compared against
FFMPEG_CAPTURE_TIMEOUT_SECONDS	Max time to wait for ffmpeg to connect + return a frame
Output
semantic_passed/ — annotated frames for confirmed fire/smoke events
semantic_rejected/ — annotated frames YOLO flagged but Florence didn't confirm
semantic_results.csv — one row per YOLO detection, with class, confidence, caption, similarity scores, decision, and per-stage timing
Running it
bash
python semantic_pipeline.py

Runs until interrupted (Ctrl+C), sweeping all cameras once per POLL_INTERVAL_SECONDS.

Notes on frame capture

RTSP reading goes through the ffmpeg binary directly rather than cv2.VideoCapture:

ffmpeg -rtsp_transport tcp -fflags nobuffer -flags low_delay \
       -i <url> -an -frames:v 1 -f image2 -vcodec mjpeg -q:v 3 pipe:1

The frame comes back as a single JPEG on stdout and is decoded with cv2.imdecode — no ffprobe step, no raw-video/resolution handling. That matters here specifically because these cameras have a slow RTSP handshake (10+ seconds observed); adding a second connection (e.g. for probing resolution) roughly doubles per-camera latency and risks the timeout. If a camera doesn't respond in time or returns a corrupt/partial frame, that camera is skipped for the cycle rather than raising — check the log for ffmpeg timed out / Failed to read frame warnings if a specific camera is consistently unreachable.

Memory management

torch.cuda.empty_cache() and gc.collect() run after every camera's processing (not just once per full sweep), to keep VRAM from creeping up over a long-running, many-camera session.

Known limitations / next steps
Cameras are polled sequentially, one at a time. With 31 cameras and a YOLO + Florence pass per detection, a cycle can take a while — if POLL_INTERVAL_SECONDS is too short for the number of cameras, cycles will effectively run back-to-back.
Florence-2 captions the full frame, not the cropped detection box. Cropping to the box before captioning would likely be faster and more precise, at the cost of losing surrounding context.
Each RTSP connection is opened fresh every cycle. If the camera handshake time becomes the dominant cost, a persistent background-thread frame grabber (one long-lived connection per camera, always holding the latest frame) would remove that wait entirely — this is a bigger architectural change than the current single-shot-per-cycle design.
