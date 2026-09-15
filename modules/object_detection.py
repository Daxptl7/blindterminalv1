"""
object_detection.py — BlindAssist product object announcer
==========================================================
Local YOLO inference with USB/Camera Module 3 capture, temporal confirmation,
bounded speech output, and Raspberry Pi-friendly frame scheduling.
"""

import json
import cv2
import os
import platform
import select
import shutil
import signal
import subprocess
import sys
import logging
import time
import numpy as np
from pathlib import Path
from collections import deque
from statistics import median
from typing import Optional

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


def _bounded_float(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(_settings.get(name, default))
    except (TypeError, ValueError):
        value = default
    return min(max(value, low), high)


def _bounded_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(_settings.get(name, default))
    except (TypeError, ValueError):
        value = default
    return min(max(value, low), high)


CONFIDENCE = _bounded_float("yolo_confidence", 0.50, 0.05, 0.95)
INFER_SIZE = _bounded_int("yolo_imgsz", 416, 160, 1280)
INFERENCE_INTERVAL_S = _bounded_float(
    "object_detection_interval_s", 0.40, 0.0, 5.0
)
STABILITY_WINDOW = _bounded_int("object_detection_stability_window", 5, 1, 30)
STABILITY_MIN_HITS = _bounded_int(
    "object_detection_stability_min_hits", 3, 1, STABILITY_WINDOW
)
MAX_ANNOUNCED = _bounded_int("object_detection_max_announced", 2, 1, 10)
CAMERA_BACKEND = str(
    _settings.get("object_detection_camera_backend", "auto")
).strip().lower()
ALLOW_MODEL_DOWNLOAD = bool(_settings.get("yolo_allow_download", False))
ENHANCE_LOW_LIGHT = bool(
    _settings.get("object_detection_enhance_low_light", False)
)
YOLO_DEVICE = str(_settings.get("yolo_device", "cpu")).strip() or "cpu"
# Configurable so it can be pointed away from the OCR camera on the Pi, the
# same way gesture_camera_index already is.
try:
    CAMERA_INDEX = int(_settings.get("object_detection_camera_index", 0))
except Exception:
    CAMERA_INDEX = None

DISPLAY_WINDOW = bool(_settings.get("object_detection_display", False))


class ObjectDetectionSetupError(RuntimeError):
    """A startup failure that can be explained to a non-technical user."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# Raspberry Pi exposes ISP/codecs/metadata as /dev/video* nodes. Some open
# successfully but never produce frames, so object detection must probe actual
# images instead of trusting the configured number.
_NON_CAMERA_NODE_HINTS = ("unicam", "bcm2835-isp", "rpivid", "pispbe",
                          "codec", "hevc", "isp", "stat", "meta")


def _v4l2_nodes() -> list:
    nodes = []
    base = "/sys/class/video4linux"
    if not os.path.isdir(base):
        return nodes

    for entry in sorted(os.listdir(base)):
        if not entry.startswith("video"):
            continue
        try:
            index = int(entry[len("video"):])
        except ValueError:
            continue

        name = ""
        try:
            with open(os.path.join(base, entry, "name")) as f:
                name = f.read().strip().lower()
        except OSError:
            pass
        if any(hint in name for hint in _NON_CAMERA_NODE_HINTS):
            continue

        try:
            bus = os.path.realpath(os.path.join(base, entry, "device")).lower()
        except OSError:
            bus = ""
        nodes.append(("usb" in bus, index, name))

    nodes.sort(key=lambda item: (not item[0], item[1]))
    return nodes


def _video_capture(source):
    if hasattr(cv2, "CAP_V4L2") and sys.platform.startswith("linux") and isinstance(source, int):
        return cv2.VideoCapture(source, cv2.CAP_V4L2)
    return cv2.VideoCapture(source)


def _find_rpicam_binary() -> Optional[str]:
    """Return the Raspberry Pi camera video command, including its old name."""
    return shutil.which("rpicam-vid") or shutil.which("libcamera-vid")


class _RpicamMjpegCapture:
    """Small VideoCapture-compatible adapter around rpicam-vid MJPEG stdout.

    OCR intentionally uses rpicam-still because it needs one high-resolution
    photograph. Object detection needs a continuous, low-resolution stream.
    Keeping this adapter command-line based avoids the common Pi problem where
    Picamera2 is installed for the system Python but cannot be imported from the
    separate Python 3.11 virtual environment used by MediaPipe.
    """

    def __init__(self, binary: str, width: int = 640, height: int = 480,
                 fps: int = 15, camera: int = 0, read_timeout_s: float = 3.0):
        self._buffer = bytearray()
        self._timeout = max(float(read_timeout_s), 0.25)
        self._closed = False
        command = [
            binary,
            "--timeout", "0",
            "--nopreview",
            "--codec", "mjpeg",
            "--width", str(width),
            "--height", str(height),
            "--framerate", str(fps),
            "--quality", "80",
            "--camera", str(camera),
            "--output", "-",
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except Exception:
            self._closed = True
            raise

    def isOpened(self):
        return (
            not self._closed
            and self._process.poll() is None
            and self._process.stdout is not None
        )

    def set(self, *_args):
        # Resolution and frame rate are fixed when rpicam-vid starts.
        return False

    def read(self):
        if not self.isOpened():
            return False, None

        deadline = time.monotonic() + self._timeout
        stdout = self._process.stdout
        while time.monotonic() < deadline and self.isOpened():
            start = self._buffer.find(b"\xff\xd8")
            end = self._buffer.find(b"\xff\xd9", start + 2 if start >= 0 else 0)
            if start >= 0 and end >= 0:
                jpeg = bytes(self._buffer[start:end + 2])
                del self._buffer[:end + 2]
                frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None and frame.size:
                    return True, frame
                continue

            remaining = max(0.0, deadline - time.monotonic())
            try:
                ready, _, _ = select.select([stdout], [], [], min(remaining, 0.5))
            except (OSError, ValueError):
                return False, None
            if not ready:
                continue
            try:
                chunk = os.read(stdout.fileno(), 65536)
            except OSError:
                return False, None
            if not chunk:
                return False, None
            self._buffer.extend(chunk)
            # A damaged stream must not grow forever while searching for JPEG
            # markers. Keep enough for multiple normal 640x480 frames.
            if len(self._buffer) > 8 * 1024 * 1024:
                del self._buffer[:-2 * 1024 * 1024]
        return False, None

    def release(self):
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        if process.stdout is not None:
            process.stdout.close()


def _probe_camera(source):
    cap = _video_capture(source)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    for _ in range(6):
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0):
            return cap
    cap.release()
    return None


def _open_opencv_camera(source):
    if source is not None and not isinstance(source, int):
        cap = _probe_camera(source)
        if cap is None:
            logger.error(f"Object detection camera failed to open (source={source}).")
        return cap

    nodes = _v4l2_nodes()
    usb_indices = [index for is_usb, index, _ in nodes if is_usb]
    other_indices = [index for is_usb, index, _ in nodes if not is_usb]

    if nodes:
        logger.info("Object detection video nodes: " + ", ".join(
            f"video{index}{'(usb)' if is_usb else ''}"
            f"{' ' + name if name else ''}" for is_usb, index, name in nodes))

    order = []
    if source is not None and source >= 0:
        order.append(source)
    order += [index for index in usb_indices if index not in order]
    order += [index for index in other_indices if index not in order]
    sysfs_present = sys.platform.startswith("linux") and os.path.isdir(
        "/sys/class/video4linux"
    )
    if not nodes and not sysfs_present:
        order += [index for index in range(0, 11) if index not in order]

    previous_level = None
    try:
        previous_level = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        previous_level = None

    try:
        for index in order:
            cap = _probe_camera(index)
            if cap is None:
                continue
            if index != source:
                logger.warning(
                    f"object_detection_camera_index={source} is not a working "
                    f"camera; using index {index}. Set object_detection_camera_index "
                    f"to {index} in settings.json to skip this search.")
            else:
                logger.info(f"Object detection camera ready on index {index}.")
            return cap
    finally:
        if previous_level is not None:
            try:
                cv2.utils.logging.setLogLevel(previous_level)
            except Exception:
                pass

    logger.warning("No working OpenCV camera found for object detection.")
    return None


def _open_rpicam_camera():
    binary = _find_rpicam_binary()
    if not binary:
        return None
    camera = _bounded_int("object_detection_rpicam_camera", 0, 0, 10)
    fps = _bounded_int("object_detection_camera_fps", 15, 1, 60)
    try:
        cap = _RpicamMjpegCapture(binary, fps=fps, camera=camera)
    except Exception as exc:
        logger.warning(f"Could not start Raspberry Pi camera stream: {exc}")
        return None
    for _ in range(3):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            logger.info(f"Object detection camera ready through {Path(binary).name}.")
            return cap
    cap.release()
    logger.warning(f"{Path(binary).name} started but returned no usable frames.")
    return None


def _open_camera(source, backend: Optional[str] = None):
    """Open USB/OpenCV or Camera Module 3, depending on configuration."""
    selected = (backend or CAMERA_BACKEND or "auto").lower()
    if selected not in {"auto", "opencv", "usb", "rpicam"}:
        logger.warning(f"Unknown object_detection_camera_backend={selected!r}; using auto.")
        selected = "auto"

    if selected in {"auto", "opencv", "usb"}:
        cap = _open_opencv_camera(source)
        if cap is not None:
            return cap
        if selected != "auto":
            return None

    if selected in {"auto", "rpicam"}:
        cap = _open_rpicam_camera()
        if cap is not None:
            return cap

    logger.error(
        "No working camera found for object detection. Check the USB webcam, "
        "or install rpicam-apps for Camera Module 3."
    )
    return None


# ── MODEL PATH RESOLUTION ──────────────────────────────────
def _resolve_model_path(settings: Optional[dict] = None,
                        search_dirs: Optional[list] = None) -> Optional[str]:
    """Locate YOLO weights that actually exist on this device.

    The previous version searched only for `yolov8m.pt`. This Pi ships
    yolov8n.pt and yolov8s.pt, and the configured path pointed at a directory
    that does not exist, so every candidate missed and the function returned
    the bad absolute path anyway — ultralytics cannot auto-download to an
    arbitrary absolute path, so Mode 6 failed on first use. We now accept any
    yolov8 weight present, preferring the smallest (nano is the only size that
    runs at a usable frame rate on a Pi 5 CPU).
    """
    settings = _settings if settings is None else settings
    configured = os.environ.get("BLINDASSIST_YOLO_MODEL") or settings.get(
        "yolo_model_path"
    )
    if configured:
        p = Path(configured)
        if not p.is_absolute():
            p = BASE_DIR / configured
        if p.exists():
            return str(p)
        logger.warning(f"yolo_model_path {configured!r} not found; searching for weights.")

    if search_dirs is None:
        search_dirs = [
            BASE_DIR / "models_local" / "yolo",
            BASE_DIR / "models_local",
            BASE_DIR / "models",
            BASE_DIR,
            Path(__file__).parent,
        ]
    # Nano first: on a Pi 5 CPU, yolov8n runs several times faster than yolov8m
    # and Mode 6 announces objects continuously, so latency matters more than
    # a few points of mAP.
    arm = platform.machine().lower() in {"aarch64", "arm64", "armv7l"}
    optimized = (
        "yolo11n_ncnn_model", "yolov8n_ncnn_model",
        "yolo11n.onnx", "yolov8n.onnx",
    )
    pytorch = (
        "yolo11n.pt", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"
    )
    names = optimized + pytorch if arm else pytorch + optimized
    for name in names:
        for directory in search_dirs:
            try:
                candidate = directory / name
                if candidate.exists():
                    logger.info(f"Using YOLO weights: {candidate}")
                    return str(candidate)
            except OSError:
                continue          # e.g. models/ symlink to an unmounted USB

    # Product operation must not depend on internet availability. Downloads are
    # opt-in for development and the preparation script installs the result in
    # a deterministic local directory.
    allow_download = bool(settings.get("yolo_allow_download", ALLOW_MODEL_DOWNLOAD))
    if allow_download:
        logger.warning(
            "No YOLO weights found locally; development download is enabled."
        )
        return "yolov8n.pt"
    logger.error(
        "No local YOLO model found. Run scripts/prepare_object_detection.py "
        "while online, then retry."
    )
    return None


MODEL_PATH = _resolve_model_path()

# ── MODEL (lazy-loaded, not at import time) ─────────────────
_model = None


def _get_model():
    """Lazy-load YOLO model on first use to prevent import-time crashes."""
    global _model, MODEL_PATH
    if _model is None:
        # A model may have been provisioned after this module was imported.
        if MODEL_PATH is None:
            MODEL_PATH = _resolve_model_path()
        if MODEL_PATH is None:
            raise ObjectDetectionSetupError(
                "MODEL_MISSING",
                "The object detection model is not installed. Run "
                "scripts/prepare_object_detection.py while connected to the internet."
            )
        try:
            from ultralytics import YOLO
            logger.info(f"Loading {MODEL_PATH}...")
            _model = YOLO(MODEL_PATH)
            logger.info("Model ready.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise ObjectDetectionSetupError(
                "MODEL_LOAD_FAILED", f"The object detection model could not load: {e}"
            ) from e
    return _model


def prepare_model():
    """Load the model before the live-mode stop listener is armed."""
    return _get_model()


# ── PREPROCESSING ───────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def preprocess(frame: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = _clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def _position_for(center_x: float, width: int) -> str:
    if center_x < width / 3:
        return "left"
    if center_x > 2 * width / 3:
        return "right"
    return "center"


def _box_area(detection: dict) -> float:
    try:
        x1, y1, x2, y2 = detection["box"]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
    except (KeyError, TypeError, ValueError):
        return 0.0


class TemporalDetectionFilter:
    """Require repeated evidence before an object reaches the speech layer."""

    def __init__(self, window: int = STABILITY_WINDOW,
                 min_hits: int = STABILITY_MIN_HITS):
        self.window = max(1, int(window))
        self.min_hits = min(max(1, int(min_hits)), self.window)
        self._history = deque(maxlen=self.window)

    @staticmethod
    def _group(frame_detections):
        grouped = {}
        for detection in frame_detections:
            key = (detection.get("name", "object"),
                   detection.get("position", "center"))
            grouped.setdefault(key, []).append(detection)
        return grouped

    def update(self, detections: list) -> list:
        grouped_now = self._group(detections)
        self._history.append(grouped_now)
        stable = []

        # Requiring the object in the newest frame prevents a stale object from
        # being spoken after it has left the camera view.
        for key, current_items in grouped_now.items():
            counts = [len(frame.get(key, ())) for frame in self._history]
            if sum(count > 0 for count in counts) < self.min_hits:
                continue
            stable_count = max(1, int(round(median(counts))))
            candidates = sorted(
                current_items,
                key=lambda item: (float(item.get("conf", 0.0)), _box_area(item)),
                reverse=True,
            )
            stable.extend(candidates[:stable_count])
        return stable


_DEFAULT_PRIORITY = (
    "car", "truck", "bus", "train", "motorcycle", "bicycle", "person", "dog",
    "chair", "bench", "traffic light", "stop sign",
)


def describe_detections(detections: list,
                        max_items: int = MAX_ANNOUNCED) -> str:
    """Create a short spatial description suitable for speech."""
    if not detections:
        return "I don't see anything clearly."

    configured = _settings.get("object_detection_priority_classes")
    if isinstance(configured, list):
        priority = tuple(str(name).lower() for name in configured)
    else:
        priority = _DEFAULT_PRIORITY
    priority_rank = {name: index for index, name in enumerate(priority)}

    grouped = {}
    for detection in detections:
        key = (str(detection.get("name", "object")),
               str(detection.get("position", "center")))
        group = grouped.setdefault(key, {"items": [], "score": None})
        group["items"].append(detection)

    def group_score(entry):
        (name, position), data = entry
        largest = max((_box_area(item) for item in data["items"]), default=0.0)
        confidence = max(
            (float(item.get("conf", 0.0)) for item in data["items"]), default=0.0
        )
        return (
            priority_rank.get(name.lower(), len(priority_rank) + 1),
            0 if position == "center" else 1,
            -largest,
            -confidence,
        )

    plural = {"person": "people", "mouse": "mice"}
    phrases = []
    for (name, position), data in sorted(grouped.items(), key=group_score)[:max_items]:
        count = len(data["items"])
        spoken_name = name if count == 1 else plural.get(name, name + "s")
        where = "ahead" if position == "center" else f"on your {position}"
        if count == 1:
            phrases.append(f"a {spoken_name} {where}")
        else:
            phrases.append(f"{count} {spoken_name} {where}")
    return "I see " + ", and ".join(phrases) + "."


# ── STREAMING INFERENCE ─────────────────────────────────────
def stream_detect(source=None, callback=None, max_frames=None, stop_event=None):
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

    if source is None:
        source = CAMERA_INDEX

    cap = _open_camera(source)
    if cap is None:
        logger.error(f"Object detection camera failed to open (source={source}).")
        raise RuntimeError("Camera not available for object detection.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Discard warmup frames so auto-exposure settles before the first
    # announcement (the same convention ocr.py and gesture_control.py use).
    for _ in range(5):
        cap.read()

    logger.info(f"Streaming detection active (display={DISPLAY_WINDOW})")

    inference_count = 0
    failed_reads = 0
    last_inference_at = float("-inf")
    temporal_filter = TemporalDetectionFilter()
    try:
        while cap.isOpened():
            if stop_event is not None and stop_event.is_set():
                logger.info("Object detection stopped by stop_event.")
                break

            ret, frame = cap.read()
            if not ret:
                failed_reads += 1
                if failed_reads >= 5:
                    raise ObjectDetectionSetupError(
                        "CAMERA_READ_FAILED",
                        "The object detection camera stopped returning images."
                    )
                continue
            failed_reads = 0

            now = time.monotonic()
            if now - last_inference_at < INFERENCE_INTERVAL_S:
                if DISPLAY_WINDOW:
                    cv2.imshow("Object Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue
            last_inference_at = now

            inference_frame = preprocess(frame) if ENHANCE_LOW_LIGHT else frame
            results = model.predict(
                source=inference_frame,
                conf=CONFIDENCE,
                imgsz=INFER_SIZE,
                verbose=False,
                device=YOLO_DEVICE,
            )
            result = results[0]
            detections = []

            boxes = getattr(result, "boxes", None)
            for box in ([] if boxes is None else boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names[cls]
                center_x = (x1 + x2) / 2

                detections.append({
                    'name': name,
                    'conf': conf,
                    'box': (x1, y1, x2, y2),
                    'center_x': center_x,
                    'position': _position_for(center_x, frame.shape[1]),
                })

                if DISPLAY_WINDOW:
                    color = (0, 255, 0) if conf > 0.5 else (0, 180, 255)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{name} {conf:.2f}", (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            h, _ = frame.shape[:2]
            stable_detections = temporal_filter.update(detections)
            text = describe_detections(stable_detections)

            if DISPLAY_WINDOW:
                cv2.putText(frame, text[:80], (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Object Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if callback:
                # A callback returning False means "stop" — same convention as
                # gesture_control.detect_gesture.
                if callback(text, stable_detections) is False:
                    logger.info("Object detection stopped by callback.")
                    break

            inference_count += 1
            if max_frames is not None and inference_count >= max_frames:
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
    processed = preprocess(frame) if ENHANCE_LOW_LIGHT else frame
    results = model.predict(
        source=processed,
        conf=CONFIDENCE,
        imgsz=INFER_SIZE,
        verbose=False,
        device=YOLO_DEVICE,
    )

    _, w = frame.shape[:2]
    detections = []

    boxes = getattr(results[0], "boxes", None)
    for box in ([] if boxes is None else boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        name = model.names[int(box.cls[0])]
        center_x = (x1 + x2) / 2
        detections.append({
            "name": name,
            "conf": float(box.conf[0]),
            "box": (x1, y1, x2, y2),
            "center_x": center_x,
            "position": _position_for(center_x, w),
        })

    return describe_detections(detections)


def diagnostics(probe_camera: bool = False, load_model: bool = False) -> dict:
    """Return non-destructive readiness information for selftest and support."""
    global MODEL_PATH
    MODEL_PATH = _resolve_model_path()
    model_error = None
    camera_error = None
    model_ready = bool(MODEL_PATH and Path(MODEL_PATH).exists())
    if load_model:
        try:
            prepare_model()
            model_ready = True
        except Exception as exc:
            model_error = str(exc)

    camera_ready = None
    if probe_camera:
        cap = _open_camera(CAMERA_INDEX)
        camera_ready = cap is not None
        if cap is not None:
            cap.release()
        else:
            camera_error = "No USB/OpenCV or rpicam camera returned a frame."

    return {
        "model_path": MODEL_PATH,
        "model_ready": model_ready,
        "model_error": model_error,
        "camera_backend": CAMERA_BACKEND,
        "rpicam_available": bool(_find_rpicam_binary()),
        "camera_ready": camera_ready,
        "camera_error": camera_error,
        "confidence": CONFIDENCE,
        "image_size": INFER_SIZE,
        "inference_interval_s": INFERENCE_INTERVAL_S,
        "stability_window": STABILITY_WINDOW,
        "stability_min_hits": STABILITY_MIN_HITS,
        "device": YOLO_DEVICE,
    }


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
