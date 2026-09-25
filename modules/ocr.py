"""
ocr.py — BlindAssist Project
==============================
Surya OCR 2  ·  Pi Camera Module 3 (full-range autofocus)  ·  USB camera fallback
All 5 bugs from the previous version have been corrected.

SURYA OCR 2 API FIX (this version): the previous SuryaOCREngine constructed
RecognitionPredictor() with no arguments and called it with a languages list
as a second positional argument. Per Surya OCR 2's actual current API,
SuryaInferenceManager IS required, and the predictor takes a single argument
(no langs) — it is a full-page vision-language model, not a line-by-line
language-hinted recognizer. Results now come from `.blocks[].html` instead
of `.text_lines[].text`. See the SuryaOCREngine class docstring below.
"""

import cv2
import io
import logging
import numpy as np
import os
import re
import html as html_lib
import shutil
import subprocess
import tempfile
import threading
import time
import json
import math

from dataclasses import dataclass
from pathlib import Path
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
LOG_PATH    = BASE_DIR / "logs" / "ocr.log"
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("OCRModule")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# ── rpicam-still binary detection ────────────────────────────────────────────
RPICAM_BIN = (
    shutil.which("rpicam-still")
    or shutil.which("libcamera-still")
    or "/usr/bin/rpicam-still"
)

# ── Language maps ─────────────────────────────────────────────────────────────
LANG_MAP = {
    "eng": "en",
    "hin": "hi",
    "guj": "gu",
    "en":  "en",
    "hi":  "hi",
    "gu":  "gu",
}

TESSERACT_LANG_MAP = {
    "eng": "eng",
    "hin": "hin",
    "guj": "guj",
    "en":  "eng",
    "hi":  "hin",
    "gu":  "guj",
}


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────
def _load_config() -> dict:
    defaults = {
        "ocr_model_name":   "datalab-to/surya-ocr-2",
        "ocr_default_lang": "eng",
        "ocr_engine":       "auto",
        "ocr_surya_fallback": False,
        "ocr_preload":      False,
        "ocr_capture_frames": 3,
        # Camera Module 3 native still resolution. Full-page textbook print
        # needs the sensor detail; 2304x1728 threw much of it away before OCR.
        "ocr_capture_width": 4608,
        "ocr_capture_height": 2592,
        "ocr_capture_quality": 95,
        "ocr_autofocus_mode": "auto",
        "ocr_autofocus_range": "full",
        "ocr_autofocus_speed": "normal",
        "ocr_autofocus_settle_ms": 2500,
        "ocr_autofocus_window": "0.05,0.05,0.90,0.90",
        "ocr_tile_size": 1800,
        "ocr_tile_overlap": 180,
        "ocr_gemini_first": True,
        "ocr_gemini_image_format": "JPEG",
        "ocr_gemini_image_quality": 95,
        "ocr_gemini_media_resolution": "high",
        "ocr_gemini_timeout_s": 20,
        "ocr_gemini_min_words": 12,
        "ocr_gemini_coverage_check_min_pixels": 2000000,
        "ocr_tesseract_min_confidence": 45,
        "ocr_tesseract_psm_modes": "3,6,4,11",
        "gemini_api_key":   "",
        "gemini_model_name": "gemini-3.5-flash-lite",
        "gemini_timeout_s": 8,
    }
    from modules.ocr_guidance import DEFAULTS
    defaults.update(DEFAULTS)
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            for key in defaults:
                if key in cfg:
                    defaults[key] = cfg[key]
    except Exception as e:
        logger.warning(f"Config load warning: {e}")
    return defaults


_config = _load_config()


def _config_int(key: str, default: int, min_value: int, max_value: int) -> int:
    try:
        value = int(_config.get(key, default))
    except Exception:
        value = default
    return max(min_value, min(max_value, value))


def _sharpness_score(frame: np.ndarray) -> float:
    """Higher score means text edges are sharp across most of the frame.

    A single whole-frame Laplacian score is easily dominated by one sharp page
    edge, a hand, or a textured background while the textbook itself is soft.
    The median of a 3x3 grid rewards captures whose detail is distributed.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        scores = []
        for row in range(3):
            for col in range(3):
                y0, y1 = row * height // 3, (row + 1) * height // 3
                x0, x1 = col * width // 3, (col + 1) * width // 3
                region = gray[y0:y1, x0:x1]
                if region.size:
                    scores.append(float(cv2.Laplacian(region, cv2.CV_64F).var()))
        return float(np.median(scores)) if scores else 0.0
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Low-level rpicam-still capture
# ─────────────────────────────────────────────────────────────────────────────
def _rpicam_capture() -> "cv2 frame | None":
    """
    Captures one frame via rpicam-still with document-wide autofocus.
    Returns a BGR numpy array, or None on failure.
    """
    if not RPICAM_BIN or not os.path.exists(RPICAM_BIN):
        logger.error(f"rpicam-still not found at: {RPICAM_BIN}")
        return None

    tmp = os.path.join(tempfile.gettempdir(), f"aet_ocr_cap_{os.getpid()}_{time.time_ns()}.jpg")
    width = str(_config_int("ocr_capture_width", 4608, 640, 4608))
    height = str(_config_int("ocr_capture_height", 2592, 480, 2592))
    quality = str(_config_int("ocr_capture_quality", 95, 75, 100))
    settle_ms = str(_config_int("ocr_autofocus_settle_ms", 2500, 500, 8000))
    cmd = [
        RPICAM_BIN,
        "-o", tmp,
        "-n",                        # no preview window
        "-t", settle_ms,
        "--width",   width,
        "--height",  height,
        "--quality", quality,
    ]

    # These autofocus controls are supported by the newer rpicam-still binary.
    if "rpicam-still" in RPICAM_BIN:
        af_mode = str(_config.get("ocr_autofocus_mode", "auto")).lower()
        af_range = str(_config.get("ocr_autofocus_range", "full")).lower()
        af_speed = str(_config.get("ocr_autofocus_speed", "normal")).lower()
        if af_mode not in {"auto", "continuous", "manual"}:
            af_mode = "auto"
        if af_range not in {"normal", "macro", "full"}:
            af_range = "full"
        if af_speed not in {"normal", "fast"}:
            af_speed = "normal"
        cmd += [
            "--autofocus-mode", af_mode,
            "--autofocus-range", af_range,
            "--autofocus-speed", af_speed,
        ]

        # rpicam accepts a normalized x,y,width,height autofocus region. Keep
        # it opt-in because camera mounting and page framing vary by device.
        af_window = str(_config.get(
            "ocr_autofocus_window", "0.05,0.05,0.90,0.90"
        )).strip()
        if re.fullmatch(r"(?:0(?:\.\d+)?|1(?:\.0+)?)(?:,(?:0(?:\.\d+)?|1(?:\.0+)?)){3}", af_window):
            cmd += ["--autofocus-window", af_window]

    try:
        subprocess.run(cmd, capture_output=True, timeout=10, check=True)
        frame = cv2.imread(tmp)
        if frame is None:
            logger.error("rpicam-still wrote nothing or file is unreadable.")
        else:
            logger.info("Captured Pi still at %sx%s", frame.shape[1], frame.shape[0])
        return frame
    except subprocess.TimeoutExpired:
        logger.error("rpicam-still timed out after 10 seconds.")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"rpicam-still returned non-zero exit: {e.stderr.decode(errors='replace')}")
        return None
    except Exception as e:
        logger.error(f"rpicam-still unexpected error: {e}")
        return None
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Camera Manager  (singleton)
# ─────────────────────────────────────────────────────────────────────────────
class LatestPreview:
    """Drain camera frames continuously; expose only a fresh latest frame."""
    def __init__(self, cap=None, binary=None):
        self.cap = cap
        self.process = None
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.timestamp = 0
        if binary:
            # Same 16:9 aspect as the default full-resolution still. The final
            # image is rechecked because the sensor mode may change its crop.
            command = [binary, "--timeout", "0", "--nopreview", "--codec", "mjpeg",
                       "--width", "1280", "--height", "720", "--framerate", "5",
                       "--output", "-"]
            if "rpicam" in binary:
                command += ["--autofocus-mode", "continuous", "--autofocus-range", "full"]
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL, bufsize=0)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _publish(self, frame):
        with self.lock:
            self.latest = frame
            self.timestamp = time.monotonic()

    def _read(self):
        try:
            if self.process:
                buffer = bytearray()
                while not self.stopped.is_set():
                    chunk = self.process.stdout.read(65536)
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    while True:
                        start = buffer.find(b"\xff\xd8")
                        end = buffer.find(b"\xff\xd9", max(start, 0))
                        if start < 0 or end < 0:
                            break
                        frame = cv2.imdecode(np.frombuffer(bytes(buffer[start:end+2]),
                                                          dtype=np.uint8), cv2.IMREAD_COLOR)
                        del buffer[:end+2]
                        if frame is not None:
                            self._publish(frame)
                    if len(buffer) > 4_000_000:
                        buffer.clear()
            else:
                while not self.stopped.is_set():
                    ok, frame = self.cap.read()
                    if not ok:
                        break
                    self._publish(frame)
        except Exception:
            logger.exception("OCR preview reader failed")

    def take(self):
        with self.lock:
            frame, self.latest = self.latest, None
            return frame if time.monotonic() - self.timestamp < 2 else None

    def close(self):
        self.stopped.set()
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
        self.thread.join(timeout=2)
        if self.process:
            self.process.stdout.close()
        if self.thread.is_alive():
            # Do not permit a still read concurrently with a stuck USB reader.
            raise RuntimeError("Camera preview did not stop; reconnect the camera.")


class CameraManager:
    """
    Singleton that keeps the camera handle alive across multiple scan calls.
    Priority: rpicam-still (Pi Camera 3)  →  USB / OpenCV fallback
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._cam_type = None   # "rpicam" | "usb" | None
                    obj._preview = None
                    obj._usb_cap  = None   # cv2.VideoCapture (usb only)
                    cls._instance = obj
        return cls._instance

    # ── open ──────────────────────────────────────────────────────────────────
    def open(self) -> bool:
        """
        Detects and opens the best available camera.
        Returns True if a camera is ready.
        """
        if self._cam_type is not None:
            return True   # already open

        # 1. Try rpicam-still (Pi Camera Module 3 via CSI)
        if RPICAM_BIN and os.path.exists(RPICAM_BIN):
            self._cam_type = "rpicam"
            logger.info(f"Camera ready via {RPICAM_BIN} (macro autofocus mode)")
            return True

        # 2. Fallback: USB / webcam via OpenCV
        for index in (0, 1, 2):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self._usb_cap  = cap
                self._cam_type = "usb"
                logger.info(f"Camera ready via USB OpenCV index {index}")
                return True
            cap.release()

        logger.error("No camera found (rpicam-still not found, no USB camera).")
        return False

    # ── capture ───────────────────────────────────────────────────────────────
    def capture(self, stop_check=None) -> "cv2 frame | None":
        """Captures one BGR frame from whichever camera is active."""
        capture_frames = _config_int("ocr_capture_frames", 3, 1, 5)

        if self._cam_type == "rpicam":
            best = None
            best_score = -1.0
            for _ in range(capture_frames):
                if stop_check and stop_check():
                    return None
                frame = _rpicam_capture()
                if frame is None:
                    continue
                score = _sharpness_score(frame)
                if score > best_score:
                    best = frame
                    best_score = score
            if best is not None:
                logger.info(f"Selected sharpest Pi camera capture | score={best_score:.1f}")
            return best

        if self._cam_type == "usb" and self._usb_cap:
            best = None
            best_score = -1.0
            for _ in range(5):            # warm up exposure/focus
                self._usb_cap.read()
            for _ in range(capture_frames):
                if stop_check and stop_check():
                    return None
                time.sleep(0.12)
                ret, frame = self._usb_cap.read()
                if not ret:
                    continue
                score = _sharpness_score(frame)
                if score > best_score:
                    best = frame
                    best_score = score
            if best is not None:
                logger.info(f"Selected sharpest USB camera capture | score={best_score:.1f}")
                return best
            logger.error("USB camera read() returned no usable frame.")
            return None

        logger.error("capture() called but no camera is open.")
        return None

    def start_preview(self):
        self.stop_preview()
        if self._cam_type == "usb":
            self._preview = LatestPreview(self._usb_cap)
        elif self._cam_type == "rpicam":
            binary = shutil.which("rpicam-vid") or shutil.which("libcamera-vid")
            if not binary:
                return False
            try:
                self._preview = LatestPreview(binary=binary)
            except OSError:
                logger.exception("Could not start OCR preview")
                return False
        else:
            return False
        return True

    def preview_frame(self):
        return self._preview.take() if self._preview else None

    def stop_preview(self):
        if self._preview:
            self._preview.close()
            self._preview = None

    # ── release ───────────────────────────────────────────────────────────────
    def release(self):
        self.stop_preview()
        if self._usb_cap:
            try:
                self._usb_cap.release()
            except Exception:
                pass
        self._usb_cap  = None
        self._cam_type = None
        logger.info("Camera released.")


# ─────────────────────────────────────────────────────────────────────────────
# Surya OCR 2 Engine  (singleton) — CORRECTED
# ─────────────────────────────────────────────────────────────────────────────
class SuryaOCREngine:
    """
    Lazy-loading singleton for Surya OCR 2.

    CORRECTED vs the previous version, per Surya OCR 2's actual current API:
      1. SuryaInferenceManager DOES exist and IS required — RecognitionPredictor
         must be constructed as RecognitionPredictor(manager), not with no
         arguments. The earlier comment claiming it was removed was wrong.
      2. Surya OCR 2 is a full-page vision-language model. The predictor is
         called with a single argument — predictor([pil_image]) — it does
         NOT take a languages list as a second positional argument. Passing
         one (as the previous version did) gets misinterpreted internally,
         which is what produced "'list' object has no attribute 'bboxes'".
      3. Results come back as a list of page objects with a `.blocks`
         attribute (not `.text_lines`). Each block has `.html` (the
         recognized content as an HTML fragment), `.bbox`, `.confidence`,
         and `.skipped` (True for non-text blocks like pictures). Plain
         text is extracted by stripping HTML tags from `.html`.
      4. The `lang` parameter is kept on extract_text() for backward
         compatibility with callers (main.py's scan_and_read(lang=...)),
         but is no longer forwarded into the predictor call, since Surya 2
         does not accept it — it handles 90+ languages without hints.
    """
    _instance = None
    _lock     = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._manager    = None
                    obj._predictor  = None
                    obj._loaded     = False
                    obj._load_error = None
                    cls._instance   = obj
        return cls._instance

    # ── load ──────────────────────────────────────────────────────────────────
    def _load(self) -> bool:
        if self._loaded:
            return True
        if self._load_error:
            return False

        try:
            logger.info("Loading Surya OCR 2 — first load may take 30–60 s …")
            t0 = time.time()

            # ── CORRECTED IMPORT: SuryaInferenceManager is required ───────────
            from surya.inference import SuryaInferenceManager
            from surya.recognition import RecognitionPredictor

            self._manager = SuryaInferenceManager()
            self._predictor = RecognitionPredictor(self._manager)  # ← manager passed in
            # ──────────────────────────────────────────────────────────────────

            self._loaded = True
            logger.info(f"Surya OCR 2 ready  ({time.time()-t0:.1f} s)")
            return True

        except ImportError as e:
            self._load_error = (
                f"surya-ocr not installed: {e}. "
                "Run: pip install surya-ocr"
            )
            logger.error(self._load_error)
            return False

        except Exception as e:
            self._load_error = str(e)
            logger.error(f"Surya OCR 2 load failed: {e}")
            return False

    # ── html -> plain text helper ────────────────────────────────────────────
    @staticmethod
    def _html_to_text(fragment: str) -> str:
        """Strips HTML tags from a Surya OCR 2 block's .html field and
        unescapes entities, returning plain text suitable for TTS."""
        if not fragment:
            return ""
        text = re.sub(r"<[^>]+>", " ", fragment)
        text = html_lib.unescape(text)
        return re.sub(r"\s+", " ", text).strip()

    # ── extract ───────────────────────────────────────────────────────────────
    def extract_text(self, pil_image: Image.Image, lang: str = "eng") -> str:
        """
        Runs Surya OCR 2 on a PIL image.

        Args:
            pil_image : PIL.Image in RGB mode
            lang      : accepted for backward compatibility with callers,
                        but not forwarded to the predictor — Surya OCR 2
                        does not take a language argument.

        Returns:
            Extracted text string, or an error/status message.
        """
        if not self._load():
            return f"OCR unavailable: {self._load_error}"

        try:
            t0 = time.time()

            # ── CORRECTED CALL: single positional argument, no langs list ─────
            results = self._predictor([pil_image])
            # ──────────────────────────────────────────────────────────────────

            lines = []
            for page in results:
                # CORRECTED: Surya OCR 2 stores results in .blocks, each with
                # an .html fragment — not .text_lines/.text.
                for block in getattr(page, "blocks", []):
                    if getattr(block, "skipped", False):
                        continue  # non-text block (e.g. a picture) — nothing to read
                    txt = self._html_to_text(getattr(block, "html", ""))
                    if txt:
                        lines.append(txt)

            elapsed_ms = int((time.time() - t0) * 1000)
            logger.info(
                f"Surya OCR done in {elapsed_ms} ms  | blocks={len(lines)}"
            )

            if not lines:
                return "No text detected in the image."

            return " ".join(lines)

        except Exception as e:
            logger.error(f"Surya OCR inference error: {e}")
            return f"OCR error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Tesseract OCR Engine  (high-accuracy local fallback)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TesseractCandidate:
    text: str
    avg_confidence: float
    word_count: int
    variant: str
    psm: int
    line_count: int = 0
    vertical_coverage: float = 0.0
    low_confidence_ratio: float = 0.0

    @property
    def score(self) -> float:
        chars = len(self.text)
        signal = sum(1 for ch in self.text if ch.isalpha() or ch.isdigit())
        signal_ratio = signal / max(chars, 1)
        noise_penalty = max(0.0, 0.45 - signal_ratio) * 80.0
        low_confidence_penalty = self.low_confidence_ratio * 18.0
        return (
            self.avg_confidence
            # Logarithmic, non-saturating coverage bonuses distinguish a full
            # textbook page from a crisp eight-word heading without letting
            # pure noise win merely by being long.
            + math.log1p(self.word_count) * 5.0
            + math.log1p(max(chars, 0)) * 1.5
            + min(self.line_count, 24) * 0.55
            + self.vertical_coverage * 24.0
            - noise_penalty
            - low_confidence_penalty
        )


class TesseractOCREngine:
    """
    High-accuracy local OCR path for Raspberry Pi and laptop use.

    The old version ran one thresholded image through one page segmentation
    mode. This version tries a small set of document-aware variants and keeps
    the result with the strongest Tesseract confidence.
    """
    _instance = None
    _lock     = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._available  = None
                    obj._load_error = None
                    obj._languages = None
                    cls._instance   = obj
        return cls._instance

    def _check_available(self) -> bool:
        if self._available is not None:
            return self._available

        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._available = True
            return True
        except ImportError as e:
            self._load_error = f"pytesseract not installed: {e}. Run: pip install pytesseract"
        except Exception as e:
            self._load_error = (
                f"Tesseract binary unavailable: {e}. "
                "Install it with: sudo apt install tesseract-ocr"
            )

        self._available = False
        logger.error(self._load_error)
        return False

    @staticmethod
    def _parse_psm_modes() -> list[int]:
        raw = _config.get("ocr_tesseract_psm_modes", "3,6,4,11")
        parts = raw if isinstance(raw, (list, tuple)) else str(raw).split(",")
        modes = []
        for part in parts:
            try:
                mode = int(str(part).strip())
            except Exception:
                continue
            if mode not in modes and 3 <= mode <= 13:
                modes.append(mode)
        return modes or [3, 6, 4, 11]

    @staticmethod
    def _normalize_text(text: str) -> str:
        lines = []
        for line in str(text or "").splitlines():
            line = re.sub(r"[ \t]+", " ", line).strip()
            if line:
                lines.append(line)
        return "\n".join(lines).strip()

    @staticmethod
    def _order_points(points: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        pts = points.reshape(4, 2).astype("float32")
        sums = pts.sum(axis=1)
        diffs = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(sums)]
        rect[2] = pts[np.argmax(sums)]
        rect[1] = pts[np.argmin(diffs)]
        rect[3] = pts[np.argmax(diffs)]
        return rect

    @classmethod
    def _four_point_transform(cls, image: np.ndarray, points: np.ndarray) -> np.ndarray | None:
        rect = cls._order_points(points)
        tl, tr, br, bl = rect
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_width = int(max(width_a, width_b))
        max_height = int(max(height_a, height_b))
        if max_width < 300 or max_height < 300:
            return None

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ], dtype="float32")
        matrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, matrix, (max_width, max_height))

    @classmethod
    def _document_crop(cls, image: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 180)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        image_area = image.shape[0] * image.shape[1]
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
            area = cv2.contourArea(contour)
            if area < image_area * 0.18:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            warped = cls._four_point_transform(image, approx)
            if warped is not None:
                ratio = max(warped.shape[:2]) / max(min(warped.shape[:2]), 1)
                if ratio <= 3.5:
                    return warped
        return None

    @staticmethod
    def _deskew(gray: np.ndarray) -> np.ndarray:
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 120:
            return gray

        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) < 0.3 or abs(angle) > 15:
            return gray

        height, width = gray.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        return cv2.warpAffine(
            gray,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    @classmethod
    def _variant_images(cls, pil_image: Image.Image) -> list[tuple[str, Image.Image]]:
        image = np.array(pil_image.convert("RGB"))
        bases = [("full", image)]
        cropped = cls._document_crop(image)
        if cropped is not None:
            bases.insert(0, ("document", cropped))

        variants = []
        seen = set()
        for base_name, base in bases:
            gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape[:2]
            if width < 1800:
                scale = min(2.4, 1800 / max(width, 1))
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            elif width > 3200:
                scale = 3200 / width
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            gray = cls._deskew(gray)
            # Keep one unfiltered grayscale path. Fine, pale strokes can be
            # removed by denoising/thresholding, while Tesseract often does a
            # better job when allowed to binarize the source itself.
            raw = gray
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
            denoised = cv2.fastNlMeansDenoising(clahe, h=8)
            blur = cv2.GaussianBlur(denoised, (0, 0), 1.0)
            sharp = cv2.addWeighted(denoised, 1.55, blur, -0.55, 0)
            adaptive = cv2.adaptiveThreshold(
                sharp,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                35,
                11,
            )
            otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            for name, arr in (
                (f"{base_name}-raw-gray", raw),
                (f"{base_name}-enhanced", sharp),
                (f"{base_name}-adaptive", adaptive),
                (f"{base_name}-otsu", otsu),
            ):
                key = (arr.shape, int(arr.mean()), int(arr.std()))
                if key in seen:
                    continue
                seen.add(key)
                variants.append((name, Image.fromarray(arr)))

        return variants[:8]

    @staticmethod
    def _rotation_images(pil_image: Image.Image) -> list[tuple[str, Image.Image]]:
        """Return all right-angle orientations without changing the input."""
        source = pil_image.convert("RGB")
        return [
            ("rot0", source),
            ("rot90", source.rotate(90, expand=True)),
            ("rot180", source.rotate(180, expand=True)),
            ("rot270", source.rotate(270, expand=True)),
        ]

    @classmethod
    def _page_region(cls, pil_image: Image.Image) -> Image.Image:
        """Prefer a detected document while retaining a safe full-frame fallback."""
        rgb = np.array(pil_image.convert("RGB"))
        cropped = cls._document_crop(rgb)
        return Image.fromarray(cropped) if cropped is not None else pil_image.convert("RGB")

    @classmethod
    def _tile_images(cls, pil_image: Image.Image) -> list[tuple[str, Image.Image]]:
        """Build overlapping native-resolution tiles in stable reading order."""
        page = cls._page_region(pil_image)
        gray = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2GRAY)
        gray = cls._deskew(gray)
        height, width = gray.shape[:2]
        tile_size = _config_int("ocr_tile_size", 1800, 900, 2400)
        overlap = _config_int("ocr_tile_overlap", 180, 64, min(600, tile_size // 3))

        if width <= tile_size and height <= tile_size:
            return []

        def positions(length: int) -> list[int]:
            if length <= tile_size:
                return [0]
            step = tile_size - overlap
            starts = list(range(0, max(length - tile_size + 1, 1), step))
            final = length - tile_size
            if not starts or starts[-1] != final:
                starts.append(final)
            return starts

        tiles = []
        for row, y0 in enumerate(positions(height)):
            for col, x0 in enumerate(positions(width)):
                tile = gray[y0:min(y0 + tile_size, height), x0:min(x0 + tile_size, width)]
                tiles.append((f"tile-r{row}-c{col}", Image.fromarray(tile)))
        return tiles

    @staticmethod
    def _merge_overlapping_text(parts: list[str]) -> str:
        """Merge OCR tile text while removing exact token overlap at boundaries."""
        merged: list[str] = []
        for part in parts:
            tokens = str(part or "").split()
            if not tokens:
                continue
            max_overlap = min(40, len(merged), len(tokens))
            overlap = 0
            for count in range(max_overlap, 2, -1):
                if [t.lower() for t in merged[-count:]] == [t.lower() for t in tokens[:count]]:
                    overlap = count
                    break
            merged.extend(tokens[overlap:])
        return " ".join(merged).strip()

    def _resolve_lang(self, pytesseract, lang: str) -> str:
        configured = str(_config.get("ocr_tesseract_languages", "")).strip()
        requested = configured or TESSERACT_LANG_MAP.get(lang, "eng")
        requested_parts = [part.strip() for part in requested.split("+") if part.strip()]
        if not requested_parts:
            requested_parts = ["eng"]

        if self._languages is None:
            try:
                self._languages = set(pytesseract.get_languages(config=""))
            except Exception as e:
                logger.warning(f"Could not list Tesseract language packs: {e}")
                self._languages = set()

        if not self._languages:
            return "+".join(requested_parts)

        usable = [part for part in requested_parts if part in self._languages]
        if usable:
            return "+".join(usable)
        if "eng" in self._languages:
            logger.warning(
                f"Tesseract language {requested!r} is not installed; falling back to English."
            )
            return "eng"
        return "+".join(requested_parts)

    @classmethod
    def _candidate_from_data(
        cls,
        data: dict,
        variant: str,
        psm: int,
        image_size: tuple[int, int] | None = None,
    ) -> TesseractCandidate:
        words_by_line = {}
        confidences = []
        low_confidence_words = 0
        occupied_bands = set()
        image_height = image_size[1] if image_size else 0

        total = len(data.get("text", []))
        for i in range(total):
            raw = str(data.get("text", [""])[i] or "").strip()
            text = re.sub(r"\s+", " ", raw)
            if not text:
                continue

            try:
                confidence = float(data.get("conf", ["-1"])[i])
            except Exception:
                confidence = -1.0

            if confidence >= 0:
                confidences.append(confidence)
                if confidence < 40:
                    low_confidence_words += 1

            if image_height > 0:
                try:
                    top = int(data.get("top", [0])[i])
                    box_height = int(data.get("height", [0])[i])
                    first_band = max(0, min(19, int(20 * top / image_height)))
                    last_band = max(0, min(19, int(20 * (top + box_height) / image_height)))
                    occupied_bands.update(range(first_band, last_band + 1))
                except Exception:
                    pass

            block = data.get("block_num", [0])[i]
            par = data.get("par_num", [0])[i]
            line = data.get("line_num", [0])[i]
            words_by_line.setdefault((block, par, line), []).append(text)

        lines = [" ".join(words) for _, words in sorted(words_by_line.items())]
        normalized = cls._normalize_text("\n".join(lines))
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        word_count = sum(len(line.split()) for line in lines)
        low_ratio = low_confidence_words / max(len(confidences), 1)
        return TesseractCandidate(
            normalized,
            avg_conf,
            word_count,
            variant,
            psm,
            line_count=len(lines),
            vertical_coverage=len(occupied_bands) / 20.0,
            low_confidence_ratio=low_ratio,
        )

    @classmethod
    def _candidate_from_tiles(
        cls,
        candidates: list[TesseractCandidate],
        variant: str,
        psm: int,
    ) -> TesseractCandidate:
        usable = [candidate for candidate in candidates if candidate.text]
        if not usable:
            return TesseractCandidate("", 0.0, 0, variant, psm)
        text = cls._merge_overlapping_text([candidate.text for candidate in usable])
        weight = sum(max(candidate.word_count, 1) for candidate in usable)
        avg_conf = sum(
            candidate.avg_confidence * max(candidate.word_count, 1)
            for candidate in usable
        ) / max(weight, 1)
        low_ratio = sum(
            candidate.low_confidence_ratio * max(candidate.word_count, 1)
            for candidate in usable
        ) / max(weight, 1)
        return TesseractCandidate(
            text=text,
            avg_confidence=avg_conf,
            word_count=len(text.split()),
            variant=variant,
            psm=psm,
            line_count=sum(candidate.line_count for candidate in usable),
            vertical_coverage=len(usable) / max(len(candidates), 1),
            low_confidence_ratio=low_ratio,
        )

    def _best_candidate(self, pytesseract, pil_image: Image.Image, tess_lang: str) -> TesseractCandidate:
        best = TesseractCandidate("", 0.0, 0, "none", 6)

        # Probe all right-angle rotations on a bounded-size raw grayscale
        # image, then fully process the winner (plus a close runner-up). This
        # avoids multiplying every expensive enhancement/tile pass by four.
        orientation_probes = []
        for rotation_name, rotation_image in self._rotation_images(pil_image):
            probe = rotation_image.convert("L")
            max_dimension = max(probe.size)
            if max_dimension > 1600:
                scale = 1600 / max_dimension
                probe = probe.resize(
                    (max(1, int(probe.width * scale)), max(1, int(probe.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            data = pytesseract.image_to_data(
                probe,
                lang=tess_lang,
                config="--oem 1 --psm 11 --dpi 300",
                output_type=pytesseract.Output.DICT,
            )
            candidate = self._candidate_from_data(
                data,
                f"{rotation_name}-probe",
                11,
                image_size=probe.size,
            )
            # Prefer no rotation when evidence is otherwise indistinguishable.
            orientation_probes.append((
                candidate.score + (1.5 if rotation_name == "rot0" else 0.0),
                rotation_name,
                rotation_image,
                candidate,
            ))

        orientation_probes.sort(key=lambda item: item[0], reverse=True)
        selected_orientations = orientation_probes[:1]
        top_probe = orientation_probes[0][3]
        if (
            len(orientation_probes) > 1
            and top_probe.word_count < 8
            and top_probe.avg_confidence < 55.0
            and orientation_probes[0][0] - orientation_probes[1][0] < 8.0
        ):
            selected_orientations.append(orientation_probes[1])

        best_orientation_image = selected_orientations[0][2]
        best_orientation_score = -float("inf")
        psm_modes = self._parse_psm_modes()

        for _, rotation_name, rotation_image, _ in selected_orientations:
            orientation_best = TesseractCandidate("", 0.0, 0, "none", 6)
            for variant_name, variant_image in self._variant_images(rotation_image):
                for psm in psm_modes:
                    config = (
                        f"--oem 1 --psm {psm} --dpi 300 "
                        "-c preserve_interword_spaces=1 "
                        "-c textord_heavy_nr=1"
                    )
                    data = pytesseract.image_to_data(
                        variant_image,
                        lang=tess_lang,
                        config=config,
                        output_type=pytesseract.Output.DICT,
                    )
                    candidate = self._candidate_from_data(
                        data,
                        f"{rotation_name}-{variant_name}",
                        psm,
                        image_size=variant_image.size,
                    )
                    if candidate.score > orientation_best.score:
                        orientation_best = candidate
                    if candidate.score > best.score:
                        best = candidate
            if orientation_best.score > best_orientation_score:
                best_orientation_score = orientation_best.score
                best_orientation_image = rotation_image

        # Preserve native pixels for the page-region tile pass. One PSM keeps
        # latency bounded; full-image variants above still exercise every
        # configured segmentation mode.
        tiles = self._tile_images(best_orientation_image)
        if tiles:
            tile_psm = 6 if 6 in psm_modes else psm_modes[0]
            tile_candidates = []
            for tile_name, tile_image in tiles:
                config = (
                    f"--oem 1 --psm {tile_psm} --dpi 300 "
                    "-c preserve_interword_spaces=1 "
                    "-c textord_heavy_nr=1"
                )
                data = pytesseract.image_to_data(
                    tile_image,
                    lang=tess_lang,
                    config=config,
                    output_type=pytesseract.Output.DICT,
                )
                tile_candidates.append(self._candidate_from_data(
                    data,
                    tile_name,
                    tile_psm,
                    image_size=tile_image.size,
                ))
            tiled = self._candidate_from_tiles(
                tile_candidates,
                "native-overlapping-tiles",
                tile_psm,
            )
            if tiled.score > best.score:
                best = tiled

        return best

    def extract_text(self, pil_image: Image.Image, lang: str = "eng") -> str:
        if not self._check_available():
            return f"OCR unavailable: {self._load_error}"

        try:
            import pytesseract

            t0 = time.time()
            tess_lang = self._resolve_lang(pytesseract, lang)
            best = self._best_candidate(pytesseract, pil_image, tess_lang)
            min_conf = float(_config.get("ocr_tesseract_min_confidence", 45))
            elapsed_ms = int((time.time() - t0) * 1000)
            logger.info(
                "Tesseract OCR done in %s ms | chars=%s | confidence=%.1f | "
                "variant=%s | psm=%s",
                elapsed_ms,
                len(best.text),
                best.avg_confidence,
                best.variant,
                best.psm,
            )

            if not best.text or best.avg_confidence < min_conf:
                if best.text:
                    logger.warning(
                        "Rejecting low-confidence OCR candidate | confidence=%.1f | minimum=%.1f",
                        best.avg_confidence,
                        min_conf,
                    )
                return "No text detected in the image."
            return best.text

        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            return f"OCR error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Gemini OCR Engine  (fast/free cloud path — needs internet)
# ─────────────────────────────────────────────────────────────────────────────
class GeminiOCREngine:
    """
    Sends the captured frame to the Gemini API (image + prompt -> text).
    Runs on Google's servers, not the Pi, so it's fast (seconds, not
    minutes) even though the Pi has no GPU.

    Requires:
      - gemini_api_key set in settings.json (already present)
      - `pip install google-genai --break-system-packages`
      - working internet connection at call time

    On any failure (no internet, bad key, timeout, package missing),
    extract_text() raises — the caller (scan_and_read) is expected to
    catch that and fall back to local OCR.
    """

    _instance = None
    _lock     = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._client       = None
                    obj._model_name   = _config.get("gemini_model_name", "gemini-3.5-flash-lite")
                    obj._api_key      = _config.get("gemini_api_key", "")
                    obj._timeout_s    = _config.get("gemini_timeout_s", 8)
                    cls._instance     = obj
        return cls._instance

    def _get_client(self):
        api_key = _config.get("gemini_api_key", "") or self._api_key
        if api_key != self._api_key:
            self._api_key = api_key
            self._client = None
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise RuntimeError("gemini_api_key not set in settings.json")
        from google import genai  # local import: keep this optional at module load
        self._client = genai.Client(api_key=self._api_key)
        return self._client

    def extract_text(self, pil_image: Image.Image, lang: str = "eng") -> str:
        """
        Sends the image straight to Gemini and returns the text it reads.
        Raises on any failure — caller must catch and fall back.
        """
        from google.genai import types  # local import, same reason as above

        client = self._get_client()

        buf = io.BytesIO()
        image_format = str(_config.get("ocr_gemini_image_format", "JPEG")).upper()
        if image_format == "PNG":
            pil_image.save(buf, format="PNG")
            mime_type = "image/png"
        else:
            image_format = "JPEG"
            quality = _config_int("ocr_gemini_image_quality", 95, 85, 100)
            pil_image.convert("RGB").save(
                buf,
                format="JPEG",
                quality=quality,
                subsampling=0,
            )
            mime_type = "image/jpeg"
        image_bytes = buf.getvalue()

        t0 = time.time()
        try:
            timeout_s = float(_config.get(
                "ocr_gemini_timeout_s",
                _config.get("gemini_timeout_s", self._timeout_s),
            ))
        except (TypeError, ValueError):
            timeout_s = 20.0

        config_kwargs = {
            "http_options": types.HttpOptions(timeout=timeout_s * 1000),
        }
        if str(_config.get("ocr_gemini_media_resolution", "high")).lower() == "high":
            media_resolution = getattr(
                getattr(types, "MediaResolution", None),
                "MEDIA_RESOLUTION_HIGH",
                None,
            )
            if media_resolution is not None:
                config_kwargs["media_resolution"] = media_resolution

        response = client.models.generate_content(
            model=self._model_name,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                "Transcribe every visible word in this document, including "
                "small body text, headings, footnotes, labels, and page "
                "numbers. Preserve paragraphs and natural reading order. "
                "Do not summarize and do not guess unreadable words. Return "
                "only the transcription, with no markdown or commentary.",
            ],
            config=types.GenerateContentConfig(**config_kwargs),
        )
        elapsed_ms = int((time.time() - t0) * 1000)

        text = (response.text or "").strip()
        logger.info(f"Gemini OCR done in {elapsed_ms} ms")

        if not text:
            return "No text detected in the image."
        return text


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singletons
# ─────────────────────────────────────────────────────────────────────────────
_camera = CameraManager()
_tesseract_engine = TesseractOCREngine()  # fast local default
_engine = SuryaOCREngine()          # optional high-accuracy/heavy fallback
_gemini_engine = GeminiOCREngine()  # fast cloud path — used first when possible

# Optional Surya preload. Loading Surya can make a Pi feel frozen, so it only
# preloads when Surya is explicitly selected or enabled as a fallback.
_ocr_engine_name = str(_config.get("ocr_engine", "auto")).lower()
_surya_enabled = _ocr_engine_name == "surya" or bool(_config.get("ocr_surya_fallback", False))
if _surya_enabled and _config.get("ocr_preload", False):
    threading.Thread(target=_engine._load, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Public API  (called by main.py)
# ─────────────────────────────────────────────────────────────────────────────
def open_camera() -> bool:
    """Opens the best available camera.  Call once at mode startup."""
    return _camera.open()


def scan_and_read(lang: str = "eng", speak_fn=None, stop_check=None) -> str:
    """
    Captures one frame and returns the extracted text.

    Args:
        lang : language code — "eng", "hin", "guj"  (or ISO: "en", "hi", "gu")
        speak_fn: optional blocking prompt callback; enables guided positioning.
        stop_check: optional cancellation predicate. Existing callers without
            speak_fn retain immediate capture behavior.

    Returns:
        Extracted text string, or an error message starting with "OCR".
    """
    if not _camera.open():
        return "Camera not available. Please check the ribbon cable connection."

    if speak_fn is not None and _config.get("ocr_guidance_enabled", True):
        from modules.ocr_guidance import guided_capture
        frame, status = guided_capture(_camera, speak_fn, stop_check, _config)
        if frame is None:
            return status
    else:
        frame = _camera.capture()
    if stop_check and stop_check():
        return "OCR scan cancelled."
    if frame is None:
        return "Image capture failed. Please try again."

    # Convert BGR (OpenCV) → RGB (PIL)
    try:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception as e:
        logger.error(f"Frame conversion error: {e}")
        return "Image processing error."

    engine_name = str(_config.get("ocr_engine", "auto")).lower()

    # ── Helper: detect Gemini "I see no text" style answers ──────────
    _no_text_phrases = ("no text", "no visible text", "no readable text",
                        "i cannot", "i can't", "there is no")

    def _is_usable(t: str) -> bool:
        """True when `t` looks like real OCR content, not an error/empty."""
        if not t:
            return False
        low = t.lower()
        if low.startswith(("ocr unavailable", "ocr error")):
            return False
        if "no text detected" in low:
            return False
        if any(p in low for p in _no_text_phrases):
            return False
        return True

    tess_text = ""
    tess_ok = False
    gemini_first = bool(_config.get("ocr_gemini_first", True))

    def _try_gemini() -> str:
        try:
            gemini_text = _gemini_engine.extract_text(pil_image, lang=lang)
            if _is_usable(gemini_text):
                min_words = _config_int("ocr_gemini_min_words", 12, 1, 200)
                word_count = len(re.findall(r"\S+", gemini_text))
                min_pixels = _config_int(
                    "ocr_gemini_coverage_check_min_pixels",
                    2000000,
                    100000,
                    50000000,
                )
                is_full_page_capture = pil_image.width * pil_image.height >= min_pixels
                if not is_full_page_capture or word_count >= min_words:
                    return gemini_text
                logger.warning(
                    "Gemini OCR returned only %s word(s); running the local "
                    "coverage check instead.",
                    word_count,
                )
            else:
                logger.info(f"Gemini found no usable text: {gemini_text!r}")
        except Exception as e:
            logger.warning(f"Gemini OCR unavailable, falling back locally: {e}")
        return ""

    def _try_tesseract() -> str:
        nonlocal tess_text, tess_ok
        tess_text = _tesseract_engine.extract_text(pil_image, lang=lang)
        tess_ok = _is_usable(tess_text)
        return tess_text

    # ── HIGHEST-ACCURACY PATH: Gemini vision OCR first when configured ──
    # The Pi keeps working without internet/API access because every Gemini
    # failure falls back to the local Tesseract pipeline below.
    if engine_name in ("auto", "gemini") and _config.get("gemini_api_key") and gemini_first:
        gemini_text = _try_gemini()
        if gemini_text:
            return gemini_text

    # ── LOCAL PATH: improved multi-pass Tesseract fallback ─────────────
    if engine_name in ("auto", "tesseract", "gemini"):
        _try_tesseract()
        if engine_name == "tesseract" and tess_ok:
            return tess_text
        if engine_name == "auto" and tess_ok and not _config.get("gemini_api_key"):
            return tess_text

    # Optional cloud second look for users who prefer fast local-first scans.
    if engine_name in ("auto", "gemini") and _config.get("gemini_api_key") and not gemini_first:
        gemini_text = _try_gemini()
        if gemini_text:
            return gemini_text

    if engine_name in ("auto", "tesseract", "gemini"):
        if tess_ok:
            return tess_text
        if _config.get("ocr_surya_fallback", False):
            logger.warning(f"Tesseract OCR did not produce usable text, trying Surya: {tess_text}")
            return _engine.extract_text(pil_image, lang=lang)
        return tess_text

    return _engine.extract_text(pil_image, lang=lang)


def release_camera():
    """Releases the camera handle.  Call at mode exit."""
    _camera.release()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  BlindAssist — OCR Module Standalone Test")
    print("=" * 55)

    if not open_camera():
        print("ERROR: No camera found. Exiting.")
        raise SystemExit(1)

    # Capture and save a debug copy so we can inspect what the camera sees
    print("Camera open. Capturing frame …")
    _cam = CameraManager()
    frame = _cam.capture()
    if frame is not None:
        debug_path = "/tmp/ocr_debug.jpg"
        cv2.imwrite(debug_path, frame)
        print(f"DEBUG: Captured frame saved to {debug_path}")
    else:
        print("WARNING: capture() returned None")

    print("Running OCR …")
    result = scan_and_read(lang="eng")

    print("\nOCR Output:")
    print("-" * 55)
    print(result)
    print("-" * 55)

    release_camera()
    print("Done.")
