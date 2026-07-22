# fire-smoke
# Semantic Fire & Smoke Verification Pipeline

## Overview

This project implements a two-stage AI pipeline for industrial fire and smoke detection using RTSP camera streams.

The pipeline combines:

- YOLO for real-time object detection
- Microsoft Florence-2 Large Vision Language Model (VLM) for scene understanding
- SentenceTransformer semantic similarity verification

The objective is to reduce false positives by validating YOLO detections using natural language scene descriptions instead of relying solely on object detection confidence.

---

# Pipeline Architecture

```
                RTSP Camera Stream
                        │
                        ▼
                 Read Current Frame
                        │
                        ▼
                YOLO Fire/Smoke Detector
                        │
             Detection Above Threshold?
                │                 │
               No                Yes
                │                 ▼
             Skip Frame     Florence-2 Caption
                                  │
                                  ▼
                     Sentence Embedding Model
                                  │
                                  ▼
                  Semantic Similarity Matching
                 (Fire / Smoke Trigger Phrases)
                                  │
                                  ▼
                     PASS / REJECT Decision
                                  │
                 ┌────────────────┴──────────────┐
                 ▼                               ▼
          Save Image                     Log Results
```

---

# Features

- Supports multiple RTSP camera streams
- YOLO-based fire and smoke detection
- Florence-2 scene caption generation
- Semantic verification using SentenceTransformer
- Automatic PASS / REJECT decision
- CSV logging of every prediction
- Resource monitoring
    - CPU Usage
    - RAM Usage
    - GPU VRAM
- Saves annotated images
- Detailed timing statistics for every pipeline stage

---

# Components

## 1. YOLO Detector

Responsible for detecting:

- Fire
- Smoke

Only detections above the configured confidence threshold are processed further.

Configuration:

```python
YOLO_CONF_THRESHOLD = 0.35
```

---

## 2. Florence-2 Large

Generates a detailed caption describing the current camera scene.

Example:

```
A factory floor with dense black smoke rising from machinery.
```

---

## 3. Semantic Verification

The generated caption is compared against predefined fire and smoke phrases using SentenceTransformer.

Fire trigger examples:

- visible flames
- industrial fire
- burning machinery
- active combustion

Smoke trigger examples:

- dense smoke
- heavy smoke
- smoke plume
- black smoke

The maximum cosine similarity is used for verification.

---

## Decision Logic

```
if

Fire Similarity >= Threshold

OR

Smoke Similarity >= Threshold

↓

PASS

Else

↓

REJECT
```

Similarity threshold:

```python
SIMILARITY_THRESHOLD = 0.5
```

---

# Folder Structure

```
project/

│
├── semantic_passed/
│      Verified detections
│
├── semantic_rejected/
│      Rejected detections
│
├── semantic_results.csv
│
├── main.py
│
└── YOLO Model
```

---

# Output
 Rejected Or Accepted


## CSV Output

Each processed detection contains:

- Camera ID
- YOLO Class
- YOLO Confidence
- Florence Caption
- Fire Similarity
- Smoke Similarity
- Decision
- Timestamp
- Frame Read Time
- YOLO Inference Time
- Florence Time
- Embedding Time
- Total Pipeline Time
- CPU Usage
- RAM Usage
- VRAM Usage

---

# Performance Metrics

The following timing statistics are recorded:

- RTSP Frame Read
- YOLO Inference
- Florence Caption Generation
- Sentence Embedding
- Total Pipeline Time

System metrics include:

- CPU Utilization
- Python Process RAM
- Total System RAM
- GPU VRAM Allocation

---

# Requirements

Python 3.10+

Install dependencies:

```bash
pip install ultralytics
pip install transformers==4.41.2
pip install sentence-transformers
pip install opencv-python-headless
pip install pillow
pip install psutil
```

---

# Configuration

Edit the following variables:

```python
RTSP_STREAMS
```

Add all RTSP camera URLs.

---

YOLO model

```python
YOLO_MODEL_PATH
```

---

Detection threshold

```python
YOLO_CONF_THRESHOLD
```

---

Semantic threshold

```python
SIMILARITY_THRESHOLD
```

---

Polling interval

```python
POLL_INTERVAL_SECONDS
```

---

# Current Processing Flow

Every polling cycle:

```
For each camera

↓

Open RTSP stream

↓

Read one frame

↓

Close stream

↓

Run YOLO

↓

Generate Florence caption

↓

Semantic similarity verification

↓

PASS / REJECT

↓

Save image

↓

Log CSV
```

---

# Future Improvements

- Persistent FFmpeg-based RTSP streaming to eliminate repeated connection overhead.
- Continuous frame buffering with latest-frame retrieval.
- Quantized Florence model for faster inference.
- Region-of-interest (ROI) caption generation instead of full-frame captioning.
- Asynchronous pipeline execution for higher throughput.
- Batch processing across multiple camera streams.
- Automatic RTSP reconnection and health monitoring.

---

# License

This project is intended for research, industrial AI deployment, and intelligent fire & smoke monitoring systems.
