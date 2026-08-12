"""
object_detection.py — BlindAssist Project (CORRECTED)
=======================================================
YOLOv8 streaming inference with generator.
Processes frames asynchronously. No blocking per-frame.
"""

import json
import cv2
import signal
import sys
import logging
import numpy as np
from pathlib import Path
from collections import Counter, deque

logger = logging.getLogger("ObjectDetection")

# ── CONFIG ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "settings.json"


def _load_settings() -> dict:
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


_settings = _load_settings()

CONFIDENCE = float(_settings.get("yolo_confidence", 0.75))
INFER_SIZE = int(_settings.get("yolo_imgsz", 640))
# Configurable so it can be pointed away from the OCR camera on the Pi, the
# same way gesture_camera_index already is.
CAMERA_INDEX = int(_settings.get("object_detection_camera_index", 0))

DISPLAY_WINDOW = bool(_settings.get("object_detection_display", False))


# ── MODEL PATH RESOLUTION ──────────────────────────────────
def _resolve_model_path() -> str:
    """Locate YOLO weights that actually exist on this device.

    The previous version searched only for `yolov8m.pt`. This Pi ships
    yolov8n.pt and yolov8s.pt, and the configured path pointed at a directory
    that does not exist, so every candidate missed and the function returned
    the bad absolute path anyway — ultralytics cannot auto-download to an
    arbitrary absolute path, so Mode 6 failed on first use. We now accept any
    yolov8 weight present, preferring the smallest (nano is the only size that
    runs at a usable frame rate on a Pi 5 CPU).
    """
    configured = _settings.get("yolo_model_path")
    if configured:
        p = Path(configured)
        if not p.is_absolute():
            p = BASE_DIR / configured
        if p.exists():
            return str(p)
        logger.warning(f"yolo_model_path {configured!r} not found; searching for weights.")

    search_dirs = [BASE_DIR / "models_local", BASE_DIR / "models", BASE_DIR,
                   Path(__file__).parent]
    # Nano first: on a Pi 5 CPU, yolov8n runs several times faster than yolov8m
    # and Mode 6 announces objects continuously, so latency matters more than
    # a few points of mAP.
    for name in ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"):
        for directory in search_dirs:
            try:
                candidate = directory / name
                if candidate.exists():
                    logger.info(f"Using YOLO weights: {candidate}")
                    return str(candidate)
            except OSError:
                continue          # e.g. models/ symlink to an unmounted USB

    # Nothing on disk: a bare filename lets ultralytics download it on demand.
    logger.warning("No YOLO weights found locally; ultralytics will try to download yolov8n.pt.")
    return "yolov8n.pt"


MODEL_PATH = _resolve_model_path()

# ── MODEL (lazy-loaded, not at import time) ─────────────────
_model = None


def _get_model():
    """Lazy-load YOLO model on first use to prevent import-time crashes."""
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO
            logger.info(f"Loading {MODEL_PATH}...")
            _model = YOLO(MODEL_PATH)
            logger.info("Model ready.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    return _model


# ── PREPROCESSING ───────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def preprocess(frame: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = _clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


# ── STREAMING INFERENCE ─────────────────────────────────────
def stream_detect(source=0, callback=None, max_frames=None, stop_event=None):
    """
    Streaming object detection.
    Reads frames from capture and runs YOLOv8 inference.

    Stopping (previously impossible on a headless device — see main.py Mode 6):
      - set `stop_event` from another thread, or
      - return False from `callback`, or
      - pass `max_frames`, or
      - press Q, but only when object_detection_display is enabled.
    """
    model = _get_model()

    # CAP_V4L2 is a Linux-only backend; requesting it on macOS/Windows makes
    # VideoCapture fail to open at all, so only ask for it where it exists.
    if hasattr(cv2, "CAP_V4L2") and sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        cap.release()
        logger.error(f"Object detection camera failed to open (source={source}).")
        raise RuntimeError("Camera not available for object detection.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Discard warmup frames so auto-exposure settles before the first
    # announcement (the same convention ocr.py and gesture_control.py use).
    for _ in range(5):
        cap.read()

    logger.info(f"Streaming detection active (display={DISPLAY_WINDOW})")

    frame_count = 0
    try:
        while cap.isOpened():
            if stop_event is not None and stop_event.is_set():
                logger.info("Object detection stopped by stop_event.")
                break

            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                source=frame,
                conf=CONFIDENCE,
                imgsz=INFER_SIZE,
                verbose=False,
                device='cpu'
            )
            result = results[0]
            detections = []

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names[cls]

                detections.append({
                    'name': name,
                    'conf': conf,
                    'box': (x1, y1, x2, y2),
                    'center_x': (x1 + x2) / 2
                })

                if DISPLAY_WINDOW:
                    color = (0, 255, 0) if conf > 0.5 else (0, 180, 255)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{name} {conf:.2f}", (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            h, w = frame.shape[:2]
            descriptions = []
            for d in detections:
                pos = "left" if d['center_x'] < w / 3 else ("right" if d['center_x'] > 2 * w / 3 else "center")
                descriptions.append(f"a {d['name']} on your {pos}")

            if descriptions:
                text = "I see: " + ", ".join(descriptions)
            else:
                text = "I don't see anything clearly."

            if DISPLAY_WINDOW:
                cv2.putText(frame, text[:80], (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Object Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if callback:
                # A callback returning False means "stop" — same convention as
                # gesture_control.detect_gesture.
                if callback(text, detections) is False:
                    logger.info("Object detection stopped by callback.")
                    break

            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break

    finally:
        cap.release()
        if DISPLAY_WINDOW:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


# ── SINGLE SCAN API ─────────────────────────────────────────
def scan_frame(frame: np.ndarray) -> str:
    """Fast single-frame scan for TTS feedback."""
    model = _get_model()
    processed = preprocess(frame)
    results = model(processed, conf=CONFIDENCE, imgsz=INFER_SIZE, verbose=False)

    h, w = frame.shape[:2]
    descriptions = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        name = model.names[int(box.cls[0])]
        center_x = (x1 + x2) / 2
        pos = "left" if center_x < w / 3 else ("right" if center_x > 2 * w / 3 else "center")
        descriptions.append(f"a {name} on your {pos}")

    if not descriptions:
        return "I don't see anything clearly."
    return "I see: " + ", ".join(descriptions)


def run_detection(callback=None, max_frames=None, stop_event=None):
    """
    Legacy API wrapper used by main.py's Mode 6.
    """
    logger.info("Object detection starting.")
    stream_detect(
        source=CAMERA_INDEX,
        callback=callback,
        max_frames=max_frames,
        stop_event=stop_event,
    )

if __name__ == '__main__':
    import time
    import subprocess

    last_speech_time = 0

    def test_callback(text, detections):
        global last_speech_time
        
        # 1. Force the text to print in your terminal
        print(f"\n[AI SEES]: {text}")
        
        # 2. Force the speaker to talk (throttled to every 4 seconds so it doesn't overlap and crash)
        current_time = time.time()
        if text != "I don't see anything clearly." and (current_time - last_speech_time > 4):
            try:
                # Uses the standard Linux voice engine to speak the text
                subprocess.Popen(['espeak', text])
                last_speech_time = current_time
            except Exception as e:
                print(f"[AUDIO ERROR] Could not play sound: {e}")

    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    
    print("Starting Object Detection Test... (Will exit quickly after 40 frames)")
    # Reduced max_frames to 40 so it doesn't run for a whole minute
    run_detection(callback=test_callback, max_frames=40)
