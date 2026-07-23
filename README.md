# Semantic Fire & Smoke Detection Pipeline

This project is a real-time Fire and Smoke Detection pipeline designed for RTSP camera streams.

The pipeline keeps all RTSP streams alive in memory and processes the latest frame from each camera at a fixed interval.

## Pipeline

RTSP Cameras
        ↓
Persistent RTSP Reader
        ↓
Latest Frame in Memory
        ↓
YOLO Fire & Smoke Detection
        ↓
Florence-2 Scene Captioning
        ↓
Sentence Similarity Verification
        ↓
PASS / REJECT Decision
        ↓
CSV Logging + Annotated Image Saving

---

## Main Script

Run the following script:

```bash
python "fast rtsp read tester.py"
```

This is the primary script of the project.

It performs the complete pipeline:

- Opens all RTSP streams only once.
- Keeps every stream alive using background threads.
- Stores the latest frame in memory.
- Processes every camera every 10 seconds.
- Runs YOLO Fire & Smoke Detection.
- Verifies detections using Florence-2.
- Performs semantic similarity verification.
- Saves PASS and REJECT images.
- Stores results in a CSV file.

---

## Features

- Persistent RTSP streaming
- Background frame grabbing
- Round-robin camera scheduler
- YOLO-based Fire & Smoke Detection
- Florence-2 VLM verification
- SentenceTransformer semantic validation
- CSV logging
- CPU/RAM/VRAM monitoring
- Automatic RTSP reconnection
- Annotated image saving

---

## Project Files

```
fast rtsp read tester.py      # Main execution script
florence_semantic_tester.py   # Florence testing
requirements.txt              # Python dependencies
README.md
```

---

## Install

```bash
pip install -r requirements.txt
```

---

## Configuration

Edit the following variables inside

```python
fast rtsp read tester.py
```

- RTSP_STREAMS
- YOLO_MODEL_PATH
- POLL_INTERVAL_SECONDS
- Confidence thresholds

---

## Output

The project automatically creates:

```
semantic_passed/
semantic_rejected/
semantic_results.csv
```

---

## Requirements

- Python 3.10+
- CUDA GPU (Recommended)
- PyTorch
- OpenCV
- Ultralytics YOLO
- Transformers
- SentenceTransformers

---

## Notes

- All RTSP streams are opened only once.
- Frames are continuously updated in memory.
- No reconnect is performed unless a stream goes down.
- Every processing cycle uses the latest available frame.
- Florence-2 is only used after YOLO detects Fire or Smoke.
