# ============================================================================
# RUN THESE INSTALLATIONS IN A COLAB CELL FIRST:
#
# !pip install ultralytics transformers sentence-transformers opencv-python-headless pillow psutil
# !pip install ultralytics transformers==4.41.2 sentence-transformers opencv-python-headless pillow psutil
# ============================================================================

import os
import sys
import csv
import time
import signal
import threading
import logging
import types
import importlib.machinery
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import psutil
from PIL import Image
from ultralytics import YOLO

# Safely mock flash_attn so transformers sees it as installed
dummy_flash = types.ModuleType("flash_attn")
dummy_flash.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
sys.modules["flash_attn"] = dummy_flash

import transformers
import transformers.dynamic_module_utils
transformers.dynamic_module_utils.check_imports = lambda filename: []

# Patch EncoderDecoderCache for transformers/sentence-transformers compatibility on Colab
if not hasattr(transformers, "EncoderDecoderCache"):
    try:
        from transformers.cache_utils import EncoderDecoderCache
        setattr(transformers, "EncoderDecoderCache", EncoderDecoderCache)
    except Exception:
        class DummyEncoderDecoderCache: pass
        setattr(transformers, "EncoderDecoderCache", DummyEncoderDecoderCache)

from transformers import AutoProcessor, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------------
# Setup Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("SemanticPipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)-8s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
logger.propagate = False

# Initialize psutil CPU measurements baseline
psutil.cpu_percent(interval=None)

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------
RTSP_STREAMS = [
  #enter your rtsp streams
  "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=2&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=3&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=4&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=5&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=6&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=7&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=8&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=9&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=10&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=11&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=12&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=13&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=14&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=15&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=16&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=17&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=18&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=19&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=20&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=21&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=22&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=23&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=24&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=25&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=26&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=27&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=28&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=29&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=30&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=31&subtype=0",
    "rtsp://admin:admin%40123@103.245.34.90:554/cam/realmonitor?channel=32&subtype=0",
]

POLL_INTERVAL_SECONDS = 10
YOLO_MODEL_PATH = "me (1).pt"
YOLO_CONF_THRESHOLD = 0.35
SIMILARITY_THRESHOLD = 0.5

FIRE_PHRASES = [
    "visible flames", "orange flames", "yellow flames", "burning object",
    "burning machinery", "active combustion", "fire spreading", "large fire",
    "small fire", "industrial fire", "open flames", "flames rising",
    "fire outbreak", "burning material", "intense flames"
]

SMOKE_PHRASES = [
    "black smoke", "dense black smoke", "gray smoke", "white smoke",
    "thick smoke", "heavy smoke", "smoke plume", "smoke rising",
    "billowing smoke", "industrial smoke", "smoke emission", "visible smoke",
    "continuous smoke", "dense smoke", "dark smoke"
]

PASSED_DIR = "semantic_passed"
REJECTED_DIR = "semantic_rejected"
CSV_FILE = "semantic_results.csv"

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def get_resource_stats() -> Dict[str, float]:
    """Retrieve system CPU load, system RAM usage, Python process RAM usage, and GPU VRAM usage."""
    cpu_pct = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    sys_ram_mb = vm.used / (1024 * 1024)
    sys_ram_pct = vm.percent
    
    proc = psutil.Process()
    proc_ram_mb = proc.memory_info().rss / (1024 * 1024)
    
    vram_mb = (torch.cuda.memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
    return {
        "cpu_pct": cpu_pct,
        "proc_ram_mb": proc_ram_mb,
        "sys_ram_mb": sys_ram_mb,
        "sys_ram_pct": sys_ram_pct,
        "vram_mb": vram_mb
    }

# ---------------------------------------------------------------------------
# CSV Manager
# ---------------------------------------------------------------------------
class CSVLogger:
    def __init__(self, filepath: str = CSV_FILE):
        self.filepath = filepath
        self.headers = [
            "Camera ID", "YOLO Class", "YOLO Confidence", "Florence Caption",
            "Fire Similarity", "Smoke Similarity", "Decision", "Timestamp",
            "Frame Read (ms)", "YOLO Time (ms)", "Florence Time (ms)", "Embedder Time (ms)", "Total Pipeline (ms)",
            "CPU Load (%)", "Script RAM (MB)", "System RAM (MB)", "System RAM (%)", "VRAM Alloc (MB)"
        ]
        self._init_csv()

    def _init_csv(self):
        try:
            if not os.path.exists(self.filepath):
                with open(self.filepath, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.headers)
        except PermissionError:
            logger.warning("Could not initialize %s (File is locked/open in another program).", self.filepath)

    def append(self, row: List[str]):
        try:
            with open(self.filepath, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except PermissionError:
            logger.warning("Could not write to %s (File is locked/open in another program). Skipping CSV row.", self.filepath)
        except Exception as e:
            logger.warning("Failed to write to CSV: %s", e)

# ---------------------------------------------------------------------------
# Core Classes
# ---------------------------------------------------------------------------
class FlorenceVLM:
    def __init__(self, model_name: str = "microsoft/Florence-2-large", device: str = "cuda"):
        self.device = device
        dtype_str = "FP16 (Half Precision)" if device == "cuda" else "FP32 (Full Precision)"
        logger.info("Loading Florence-2 VLM (%s) in %s on %s...", model_name, dtype_str, device)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation="eager",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)

        # Patch _supports_sdpa attribute for compatibility with transformers >= 4.42
        if not hasattr(self.model, '_supports_sdpa'):
            setattr(self.model, '_supports_sdpa', False)

    def generate_caption(self, image_np: np.ndarray, prompt: str = "<DETAILED_CAPTION>") -> str:
        pil_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")
        inputs = {
            k: v.to(self.device, dtype=torch.float16) if torch.is_floating_point(v) and self.device == "cuda" else v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                early_stopping=False,
                do_sample=False,
                num_beams=1,
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(pil_image.width, pil_image.height)
        )
        return parsed_answer[prompt]


class YOLOVerifier:
    def __init__(self, model_path: str = YOLO_MODEL_PATH, conf_threshold: float = YOLO_CONF_THRESHOLD, device: str = "cuda"):
        self.device = device
        self.conf_threshold = conf_threshold
        dtype_str = "FP16 (Half Precision)" if device == "cuda" else "FP32"
        logger.info("Loading YOLO Model (%s) in %s on %s...", model_path, dtype_str, device)
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray):
        results = self.model(frame, verbose=False, device=self.device)
        return results


class SemanticEmbedder:
    def __init__(self, fire_phrases: List[str], smoke_phrases: List[str], model_name: str = "all-MiniLM-L6-v2", device: str = "cuda"):
        self.device = device
        logger.info("Loading SentenceTransformer (%s) on %s...", model_name, device)
        self.embedder = SentenceTransformer(model_name, device=device)
        logger.info("Precomputing trigger phrase embeddings...")
        self.fire_embeddings = self.embedder.encode(fire_phrases, convert_to_tensor=True)
        self.smoke_embeddings = self.embedder.encode(smoke_phrases, convert_to_tensor=True)

    def compute_similarity(self, caption: str) -> Tuple[float, float]:
        caption_emb = self.embedder.encode(caption, convert_to_tensor=True)
        fire_sims = util.cos_sim(caption_emb, self.fire_embeddings)[0]
        smoke_sims = util.cos_sim(caption_emb, self.smoke_embeddings)[0]
        return float(torch.max(fire_sims)), float(torch.max(smoke_sims))

## ---------------------------------------------------------------------------
# Background RTSP Reader
# ---------------------------------------------------------------------------
_shutdown_event = threading.Event()

class RTSPFrameGrabber:

    def __init__(self, url: str):
        self._url = url
        self._frame = None
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def get_latest_frame(self):

        with self._lock:

            if self._frame is None:
                return None

            return self._frame.copy()

    def _loop(self):

        while self._running and not _shutdown_event.is_set():

            logger.info(f"Connecting : {self._url}")

            cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                logger.warning(f"Failed to connect : {self._url}")
                time.sleep(2)
                continue

            logger.info(f"Connected : {self._url}")

            while self._running and not _shutdown_event.is_set():

                ret, frame = cap.read()

                if not ret:
                    logger.warning(f"Stream Lost : {self._url}")
                    break

                with self._lock:
                    self._frame = frame.copy()

            cap.release()

            logger.info("Reconnecting...")
            time.sleep(1)

# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------
class SemanticPipeline:
    def __init__(
        self,
        yolo: YOLOVerifier,
        vlm: FlorenceVLM,
        embedder: SemanticEmbedder,
        csv_logger: CSVLogger
    ):
        self.yolo = yolo
        self.vlm = vlm
        self.embedder = embedder
        self.csv_logger = csv_logger

        os.makedirs(PASSED_DIR, exist_ok=True)
        os.makedirs(REJECTED_DIR, exist_ok=True)

    def process_camera(self, cam_idx: int, grabber: RTSPFrameGrabber):
        logger.info(f"\nCapturing from Camera {cam_idx + 1}...")
        # ------------------------------------------------------------------
# 0. Get Latest Frame From Persistent RTSP Grabber
# ------------------------------------------------------------------
        t_start_frame = time.perf_counter()

        frame = grabber.get_latest_frame()

        t_end_frame = time.perf_counter()
        frame_read_ms = (t_end_frame - t_start_frame) * 1000.0

        if frame is None:
            logger.warning(f"No frame available yet for Camera {cam_idx + 1}")
            return

        annotated_frame = frame.copy()


        # 1. YOLO Detection
        t_start_yolo = time.perf_counter()
        results = self.yolo.detect(frame)
        t_end_yolo = time.perf_counter()
        yolo_ms = (t_end_yolo - t_start_yolo) * 1000.0

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf >= YOLO_CONF_THRESHOLD:
                    cls_id = int(box.cls[0])
                    cls_name = self.yolo.model.names[cls_id]

                    # 2. Florence-2 VLM Caption
                    t_start_florence = time.perf_counter()
                    caption = self.vlm.generate_caption(frame)
                    t_end_florence = time.perf_counter()
                    florence_ms = (t_end_florence - t_start_florence) * 1000.0

                    # 3. Sentence Embedding & Similarity
                    t_start_embed = time.perf_counter()
                    max_fire_sim, max_smoke_sim = self.embedder.compute_similarity(caption)
                    t_end_embed = time.perf_counter()
                    embed_ms = (t_end_embed - t_start_embed) * 1000.0

                    total_pipeline_ms = frame_read_ms + yolo_ms + florence_ms + embed_ms
                    sys_stats = get_resource_stats()

                    # 4. Decision
                    if max_fire_sim >= SIMILARITY_THRESHOLD or max_smoke_sim >= SIMILARITY_THRESHOLD:
                        decision = "PASS"
                        save_dir = PASSED_DIR
                    else:
                        decision = "REJECT"
                        save_dir = REJECTED_DIR

                    # Terminal Output Formatting
                    print("=" * 65)
                    print(f" Camera {cam_idx + 1} | YOLO Class: {cls_name} | Conf: {conf:.2f}")
                    print("-" * 65)
                    print(f" Florence Caption: {caption}")
                    print(f" Fire Similarity : {max_fire_sim:.4f} | Smoke Similarity: {max_smoke_sim:.4f}")
                    print(f" Decision        : {decision}")
                    print("-" * 65)
                    print(" STAGE TIMING STATS:")
                    print(f"  • RTSP Frame Read : {frame_read_ms:8.2f} ms")
                    print(f"  • YOLO Inference  : {yolo_ms:8.2f} ms")
                    print(f"  • Florence-2 VLM  : {florence_ms:8.2f} ms")
                    print(f"  • Embedder & Sim  : {embed_ms:8.2f} ms")
                    print(f"  • Total Pipeline  : {total_pipeline_ms:8.2f} ms")
                    print("-" * 65)
                    print(" SYSTEM RESOURCE UTILIZATION:")
                    print(f"  • CPU Utilization  : {sys_stats['cpu_pct']:5.1f}%")
                    print(f"  • Script RAM Load  : {sys_stats['proc_ram_mb']:7.1f} MB (Current Python Process)")
                    print(f"  • Total System RAM : {sys_stats['sys_ram_mb']:7.1f} MB ({sys_stats['sys_ram_pct']:.1f}% Overall System)")
                    print(f"  • VRAM Allocated   : {sys_stats['vram_mb']:7.1f} MB")
                    print("=" * 65)

                    timestamp_csv = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")

                    csv_row = [
                        f"Cam_{cam_idx+1}", cls_name, f"{conf:.2f}", caption,
                        f"{max_fire_sim:.4f}", f"{max_smoke_sim:.4f}", decision, timestamp_csv,
                        f"{frame_read_ms:.2f}", f"{yolo_ms:.2f}", f"{florence_ms:.2f}",
                        f"{embed_ms:.2f}", f"{total_pipeline_ms:.2f}",
                        f"{sys_stats['cpu_pct']:.1f}", f"{sys_stats['proc_ram_mb']:.1f}",
                        f"{sys_stats['sys_ram_mb']:.1f}", f"{sys_stats['sys_ram_pct']:.1f}", f"{sys_stats['vram_mb']:.1f}"
                    ]
                    self.csv_logger.append(csv_row)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 255, 0) if decision == "PASS" else (0, 0, 255)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{cls_name} {conf:.2f}", (x1, max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    img_name = f"cam{cam_idx+1}_{cls_name}_{timestamp_file}.jpg"
                    img_path = os.path.join(save_dir, img_name)
                    cv2.imwrite(img_path, annotated_frame)

# ---------------------------------------------------------------------------
# Execution Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    def sig_handler(signum, frame):
        _shutdown_event.set()

    signal.signal(signal.SIGINT, sig_handler)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_str = "FP16 (Half Precision)" if device == "cuda" else "FP32 (Full Precision)"

    print("=" * 65)
    print(f" DEVICE INITIALIZATION : {device.upper()}")
    if device == "cuda":
        print(f" GPU DEVICE DETECTED   : {torch.cuda.get_device_name(0)}")
        print(f" FLORENCE-2 PRECISION  : {dtype_str}")
        print(f" YOLO PRECISION        : {dtype_str}")
    else:
        print(" WARNING: CUDA NOT DETECTED! Running on CPU in FP32.")
    print("=" * 65)

    # ------------------------------------------------------------
    # Load Models
    # ------------------------------------------------------------
    yolo = YOLOVerifier(
        model_path=YOLO_MODEL_PATH,
        conf_threshold=YOLO_CONF_THRESHOLD,
        device=device,
    )

    vlm = FlorenceVLM(device=device)

    embedder = SemanticEmbedder(
        fire_phrases=FIRE_PHRASES,
        smoke_phrases=SMOKE_PHRASES,
        device=device,
    )

    csv_logger = CSVLogger(filepath=CSV_FILE)

    pipeline = SemanticPipeline(
        yolo=yolo,
        vlm=vlm,
        embedder=embedder,
        csv_logger=csv_logger,
    )

    logger.info("Starting Semantic Fire & Smoke Testing Pipeline...")

    # ------------------------------------------------------------
    # Create Persistent RTSP Grabbers (ONLY ONCE)
    # ------------------------------------------------------------
    grabbers = []

    for url in RTSP_STREAMS:
        grabber = RTSPFrameGrabber(url)
        grabber.start()
        grabbers.append(grabber)

    logger.info("Waiting 5 seconds for RTSP streams to warm up...")
    time.sleep(5)

    # ------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------
    try:

        while not _shutdown_event.is_set():

            logger.info(
                "--- Starting new polling cycle at %s ---",
                datetime.now().strftime("%H:%M:%S"),
            )

            for cam_idx, grabber in enumerate(grabbers):

                if _shutdown_event.is_set():
                    break

                pipeline.process_camera(cam_idx, grabber)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(
                "Sleeping for %d seconds...",
                POLL_INTERVAL_SECONDS,
            )

            for _ in range(POLL_INTERVAL_SECONDS):

                if _shutdown_event.is_set():
                    break

                time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user.")

    finally:

        logger.info("Stopping RTSP Grabbers...")

        _shutdown_event.set()

        for grabber in grabbers:
            grabber.stop()

        logger.info("Pipeline shut down cleanly.")