# ============================================================================
# RUN THESE INSTALLATIONS IN A COLAB CELL FIRST:
#
# !pip install ultralytics transformers sentence-transformers opencv-python-headless pillow psutil
# !pip install ultralytics transformers==4.41.2 sentence-transformers opencv-python-headless pillow psutil
# ============================================================================

import os
import sys
# Configure FFmpeg RTSP flags before importing OpenCV (zero latency, TCP transport, no buffer, 5s timeout)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000|timeout;5000000|stimeout;5000000"
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
from collections import deque

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
 #Enter you rtsp streams 
 ]

POLL_INTERVAL_SECONDS = 10
YOLO_MODEL_PATH = "me (2).pt"

# Confidence & Area Filtering Thresholds
FIRE_CONF_THRESH = 0.35
SMOKE_CONF_THRESH = 0.40

MIN_BOX_AREA = 1500
MAX_SMOKE_AREA_RATIO = 0.70

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
    def __init__(self, model_path: str = YOLO_MODEL_PATH, conf_threshold: float = min(FIRE_CONF_THRESH, SMOKE_CONF_THRESH), device: str = "cuda"):
        self.device = device
        self.conf_threshold = conf_threshold
        dtype_str = "FP16 (Half Precision)" if device == "cuda" else "FP32"
        logger.info("Loading YOLO Model (%s) in %s on %s...", model_path, dtype_str, device)
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray):
        results = self.model(frame, conf=self.conf_threshold, verbose=False, device=self.device)
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

# ---------------------------------------------------------------------------
# Persistent RTSP Stream Reader (background thread per stream)
# ---------------------------------------------------------------------------
_shutdown_event = threading.Event()


class RTSPFrameGrabber:
    """
    Opens an RTSP stream ONCE and keeps it alive in a background daemon thread.
    Continuously reads frames so the latest frame is always in memory.
    Calling get_latest_frame() is a near-instant memory copy (~0ms).
    Auto-reconnects on failure.
    """

    def __init__(self, cam_idx: int, url: str):
        self.cam_idx = cam_idx
        self._url = url
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._connected = False
        self._last_frame_time: float = 0.0

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def has_frame(self) -> bool:
        with self._lock:
            return self._frame is not None

    def start(self):
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True, name=f"RTSP-Cam{self.cam_idx + 1}")
        t.start()

    def stop(self):
        self._running = False

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Returns the latest frame from memory (near-instant, just a numpy copy)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _loop(self):
        """Background loop: connect → read frames continuously → reconnect on failure."""
        # During initial warmup, retry fast (2s) so all streams load quickly.
        # After the first successful frame, switch to 60s retry on disconnect.
        WARMUP_RETRY_DELAY = 2    # seconds — fast retry during initial load
        NORMAL_RETRY_DELAY = 60   # seconds — slow retry after stream drops
        has_ever_connected = False

        while self._running and not _shutdown_event.is_set():
            cap = None
            retry_delay = WARMUP_RETRY_DELAY if not has_ever_connected else NORMAL_RETRY_DELAY

            try:
                cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    logger.warning(
                        f"Camera {self.cam_idx + 1}: Failed to connect, "
                        f"retrying in {retry_delay}s..."
                    )
                    self._connected = False
                    self._interruptible_sleep(retry_delay)
                    continue

                self._connected = True
                has_ever_connected = True
                logger.info(f"Camera {self.cam_idx + 1}: Stream connected ✓")

                while self._running and not _shutdown_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(
                            f"Camera {self.cam_idx + 1}: Lost stream, "
                            f"will retry in {NORMAL_RETRY_DELAY}s..."
                        )
                        self._connected = False
                        break
                    with self._lock:
                        self._frame = frame
                        self._last_frame_time = time.perf_counter()

            except Exception as e:
                logger.error(f"Camera {self.cam_idx + 1}: Stream error: {e}")
                self._connected = False
            finally:
                if cap is not None:
                    cap.release()

            # Wait before reconnecting (60s for established streams)
            if self._running and not _shutdown_event.is_set():
                self._interruptible_sleep(NORMAL_RETRY_DELAY)

    def _interruptible_sleep(self, seconds: float):
        """Sleep in 1-second chunks so we can respond to shutdown quickly."""
        for _ in range(int(seconds)):
            if not self._running or _shutdown_event.is_set():
                return
            time.sleep(1)


# ---------------------------------------------------------------------------
# Round-Robin Scheduler
# ---------------------------------------------------------------------------
class CameraScheduler:
    """
    Round-robin scheduler that cycles through all cameras evenly.
    Ensures each camera gets processed before any camera is processed again.
    Tracks per-camera timing stats.
    """

    def __init__(self, grabbers: List[RTSPFrameGrabber]):
        self._grabbers = grabbers
        self._num_cameras = len(grabbers)
        self._current_idx = 0
        self._cycle_count = 0

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    def get_next_batch(self) -> List[Tuple[int, RTSPFrameGrabber]]:
        """
        Returns the full list of cameras for a complete round-robin cycle.
        Starting from the current index, wraps around so every camera is visited once.
        """
        batch = []
        for i in range(self._num_cameras):
            idx = (self._current_idx + i) % self._num_cameras
            batch.append((idx, self._grabbers[idx]))
        self._current_idx = (self._current_idx + self._num_cameras) % self._num_cameras
        self._cycle_count += 1
        return batch


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

    def process_camera(self, cam_idx: int, frame: np.ndarray, frame_read_ms: float):
        """Process a single pre-fetched frame through YOLO → Florence → Embedder pipeline."""
        logger.info(f"\nProcessing Camera {cam_idx + 1}...")

        annotated_frame = frame.copy()

        # 1. YOLO Detection
        t_start_yolo = time.perf_counter()
        results = self.yolo.detect(frame)
        t_end_yolo = time.perf_counter()
        yolo_ms = (t_end_yolo - t_start_yolo) * 1000.0

        frame_h, frame_w = frame.shape[:2]
        frame_area = float(frame_h * frame_w)

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = str(self.yolo.model.names[cls_id])
                cls_name_lower = cls_name.lower()

                # 1. Class-specific confidence threshold check
                if "fire" in cls_name_lower:
                    conf_thresh = FIRE_CONF_THRESH
                elif "smoke" in cls_name_lower:
                    conf_thresh = SMOKE_CONF_THRESH
                else:
                    conf_thresh = min(FIRE_CONF_THRESH, SMOKE_CONF_THRESH)

                if conf < conf_thresh:
                    continue

                # 2. Bounding Box Area and Ratio Filtering
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_w = max(0, x2 - x1)
                box_h = max(0, y2 - y1)
                box_area = box_w * box_h

                if box_area < MIN_BOX_AREA:
                    logger.info(
                        f"Camera {cam_idx + 1}: Skipping {cls_name} box "
                        f"(conf: {conf:.2f}, area: {box_area} < min: {MIN_BOX_AREA})"
                    )
                    continue

                if "smoke" in cls_name_lower:
                    smoke_ratio = box_area / frame_area if frame_area > 0 else 0.0
                    if smoke_ratio > MAX_SMOKE_AREA_RATIO:
                        logger.info(
                            f"Camera {cam_idx + 1}: Skipping smoke box "
                            f"(conf: {conf:.2f}, smoke area ratio: {smoke_ratio:.2f} > max: {MAX_SMOKE_AREA_RATIO})"
                        )
                        continue

                # 3. Florence-2 VLM Caption
                t_start_florence = time.perf_counter()
                caption = self.vlm.generate_caption(frame)
                t_end_florence = time.perf_counter()
                florence_ms = (t_end_florence - t_start_florence) * 1000.0

                # 4. Sentence Embedding & Similarity
                t_start_embed = time.perf_counter()
                max_fire_sim, max_smoke_sim = self.embedder.compute_similarity(caption)
                t_end_embed = time.perf_counter()
                embed_ms = (t_end_embed - t_start_embed) * 1000.0

                total_pipeline_ms = frame_read_ms + yolo_ms + florence_ms + embed_ms
                sys_stats = get_resource_stats()

                # 5. Decision
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
                print(f"  • RTSP Frame Read : {frame_read_ms:8.2f} ms  (from memory — stream pre-loaded)")
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

                color = (0, 255, 0) if decision == "PASS" else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f"{cls_name} {conf:.2f}", (x1, max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                img_name = f"cam{cam_idx+1}_{cls_name}_{timestamp_file}.jpg"
                img_path = os.path.join(save_dir, img_name)

                # Save annotated frame for both PASS and REJECT decisions
                cv2.imwrite(img_path, annotated_frame)
                if decision == "REJECT":
                    logger.info(f"Saved rejected frame with bounding box to {REJECTED_DIR}: {img_path}")

# ---------------------------------------------------------------------------
# Execution Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # STEP 0: GPU Check — print GPU status BEFORE anything else loads
    # -----------------------------------------------------------------------
    GPU_AVAILABLE = torch.cuda.is_available()
    print("=" * 65)
    print(f" GPU AVAILABLE : {GPU_AVAILABLE}")
    if GPU_AVAILABLE:
        print(f" GPU DEVICE    : {torch.cuda.get_device_name(0)}")
        print(f" CUDA VERSION  : {torch.version.cuda}")
        print(f" USING GPU     : TRUE ✓")
    else:
        print(" USING GPU     : FALSE ✗")
        print(" WARNING: No CUDA GPU detected! Running on CPU (FP32).")
        print(" Performance will be significantly slower.")
    print("=" * 65)

    def sig_handler(signum, frame):
        _shutdown_event.set()
    signal.signal(signal.SIGINT, sig_handler)

    device = "cuda" if GPU_AVAILABLE else "cpu"
    dtype_str = "FP16 (Half Precision)" if device == "cuda" else "FP32 (Full Precision)"

    print("=" * 65)
    print(f" DEVICE INITIALIZATION : {device.upper()}")
    if device == "cuda":
        print(f" GPU DEVICE DETECTED   : {torch.cuda.get_device_name(0)}")
        print(f" FLORENCE-2 PRECISION  : {dtype_str}")
        print(f" YOLO PRECISION        : FP32 (Full Precision / Original)")
    else:
        print(" WARNING: CUDA NOT DETECTED! Running on CPU in FP32.")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # STEP 1: Launch ALL RTSP streams at once (persistent background threads)
    # -----------------------------------------------------------------------
    print("=" * 65)
    print(f" LOADING ALL {len(RTSP_STREAMS)} RTSP STREAMS INTO MEMORY...")
    print(" Each stream runs in its own background thread.")
    print(" Streams stay connected — frame reads will be near-instant.")
    print("=" * 65)

    grabbers: List[RTSPFrameGrabber] = []
    for idx, url in enumerate(RTSP_STREAMS):
        grabber = RTSPFrameGrabber(cam_idx=idx, url=url)
        grabber.start()
        grabbers.append(grabber)

    # -----------------------------------------------------------------------
    # Wait until ALL streams have at least one frame loaded in memory.
    # Poll every 2 seconds, show live progress. Max wait = 120 seconds.
    # Pipeline will NOT start until all frames are in memory.
    # -----------------------------------------------------------------------
    WARMUP_POLL_INTERVAL = 2     # seconds between progress checks
    WARMUP_MAX_TIMEOUT   = 120   # max seconds to wait before giving up on stragglers
    total_streams = len(RTSP_STREAMS)

    logger.info(f"Waiting for ALL {total_streams} streams to load first frame into memory...")
    warmup_start = time.perf_counter()

    while True:
        ready_count = sum(1 for g in grabbers if g.has_frame)
        connected_count = sum(1 for g in grabbers if g.is_connected)
        elapsed = time.perf_counter() - warmup_start

        print(f"\r  ⏳ Streams ready: {ready_count}/{total_streams} | "
              f"Connected: {connected_count}/{total_streams} | "
              f"Elapsed: {elapsed:.0f}s", end="", flush=True)

        if ready_count == total_streams:
            # All streams have frames in memory!
            print()  # newline after \r progress
            break

        if elapsed >= WARMUP_MAX_TIMEOUT:
            print()  # newline after \r progress
            logger.warning(
                f"Warmup timeout ({WARMUP_MAX_TIMEOUT}s) reached. "
                f"{ready_count}/{total_streams} streams ready. "
                f"Proceeding with available streams."
            )
            break

        time.sleep(WARMUP_POLL_INTERVAL)

    warmup_elapsed_ms = (time.perf_counter() - warmup_start) * 1000.0

    # Final status report — all streams loaded
    ready_count = sum(1 for g in grabbers if g.has_frame)
    connected_count = sum(1 for g in grabbers if g.is_connected)
    print("=" * 65)
    print(f" ALL STREAMS LOADED INTO MEMORY")
    print(f"  • Warmup Time   : {warmup_elapsed_ms:.0f} ms")
    print(f"  • Frames Ready  : {ready_count}/{total_streams}")
    print(f"  • Connected     : {connected_count}/{total_streams}")
    print("-" * 65)
    for i, g in enumerate(grabbers):
        status = "✓ FRAME IN MEMORY" if g.has_frame else ("⚠ CONNECTED (no frame)" if g.is_connected else "✗ DISCONNECTED")
        print(f"  • Camera {i+1:2d}  : {status}")
    print("=" * 65)

    if ready_count == 0:
        logger.error("No streams have frames loaded! Check network/RTSP URLs. Exiting.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # STEP 2: Load ML models
    # -----------------------------------------------------------------------
    min_conf_thresh = min(FIRE_CONF_THRESH, SMOKE_CONF_THRESH)
    yolo = YOLOVerifier(model_path=YOLO_MODEL_PATH, conf_threshold=min_conf_thresh, device=device)
    vlm = FlorenceVLM(device=device)
    embedder = SemanticEmbedder(fire_phrases=FIRE_PHRASES, smoke_phrases=SMOKE_PHRASES, device=device)
    csv_logger = CSVLogger(filepath=CSV_FILE)

    pipeline = SemanticPipeline(yolo=yolo, vlm=vlm, embedder=embedder, csv_logger=csv_logger)

    # -----------------------------------------------------------------------
    # STEP 3: Create scheduler and start processing
    # -----------------------------------------------------------------------
    scheduler = CameraScheduler(grabbers)

    logger.info("Starting Semantic Fire & Smoke Testing Pipeline...")
    logger.info(f"Scheduler: Round-robin across {len(RTSP_STREAMS)} cameras, poll interval {POLL_INTERVAL_SECONDS}s")

    try:
        while not _shutdown_event.is_set():
            cycle_num = scheduler.cycle_count + 1
            logger.info("--- Cycle %d starting at %s ---", cycle_num, datetime.now().strftime("%H:%M:%S"))

            batch = scheduler.get_next_batch()
            cameras_processed = 0
            cameras_skipped = 0

            for cam_idx, grabber in batch:
                if _shutdown_event.is_set():
                    break

                # Grab latest frame from memory (near-instant)
                t_read_start = time.perf_counter()
                frame = grabber.get_latest_frame()
                t_read_end = time.perf_counter()
                frame_read_ms = (t_read_end - t_read_start) * 1000.0

                if frame is None:
                    cameras_skipped += 1
                    logger.warning(f"Camera {cam_idx + 1}: No frame in memory (stream down?), skipping.")
                    continue

                cameras_processed += 1
                pipeline.process_camera(cam_idx, frame, frame_read_ms)

            logger.info(
                "Cycle %d complete: %d processed, %d skipped",
                cycle_num, cameras_processed, cameras_skipped
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Sleeping for %d seconds...", POLL_INTERVAL_SECONDS)
            for _ in range(POLL_INTERVAL_SECONDS):
                if _shutdown_event.is_set():
                    break
                time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user.")
    finally:
        logger.info("Stopping all stream grabbers...")
        for g in grabbers:
            g.stop()
        logger.info("Pipeline shut down cleanly.")er.info("Stopping all stream grabbers...")
        for g in grabbers:
            g.stop()
        logger.info("Pipeline shut down cleanly.")
