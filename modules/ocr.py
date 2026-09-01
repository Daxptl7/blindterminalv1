"""
ocr.py — BlindAssist Project
==============================
Surya OCR 2  ·  Pi Camera Module 3 (macro autofocus)  ·  USB camera fallback
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
        "ocr_capture_width": 2304,
        "ocr_capture_height": 1728,
        "ocr_gemini_first": True,
        "ocr_tesseract_min_confidence": 45,
        "ocr_tesseract_psm_modes": "6,4,11",
        "gemini_api_key":   "",
        "gemini_model_name": "gemini-3-flash-lite",
        "gemini_timeout_s": 8,
    }
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
    """Higher score means sharper text edges; used to pick the best capture."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Low-level rpicam-still capture
# ─────────────────────────────────────────────────────────────────────────────
def _rpicam_capture() -> "cv2 frame | None":
    """
    Captures one frame via rpicam-still with macro autofocus.
    Returns a BGR numpy array, or None on failure.
    """
    if not RPICAM_BIN or not os.path.exists(RPICAM_BIN):
        logger.error(f"rpicam-still not found at: {RPICAM_BIN}")
        return None

    tmp = os.path.join(tempfile.gettempdir(), f"aet_ocr_cap_{os.getpid()}_{time.time_ns()}.jpg")
    width = str(_config_int("ocr_capture_width", 2304, 640, 4608))
    height = str(_config_int("ocr_capture_height", 1728, 480, 3456))
    cmd = [
        RPICAM_BIN,
        "-o", tmp,
        "-n",                        # no preview window
        "-t", "2000",                # 2 s settle time for macro autofocus (ms)
        "--width",   width,
        "--height",  height,
        "--quality", "90",
    ]

    # Macro autofocus is only supported by the newer rpicam-still binary
    if "rpicam-still" in RPICAM_BIN:
        cmd += ["--autofocus-mode", "auto", "--autofocus-range", "macro"]

    try:
        subprocess.run(cmd, capture_output=True, timeout=10, check=True)
        frame = cv2.imread(tmp)
        if frame is None:
            logger.error("rpicam-still wrote nothing or file is unreadable.")
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
    def capture(self) -> "cv2 frame | None":
        """Captures one BGR frame from whichever camera is active."""
        capture_frames = _config_int("ocr_capture_frames", 3, 1, 5)

        if self._cam_type == "rpicam":
            best = None
            best_score = -1.0
            for _ in range(capture_frames):
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

    # ── release ───────────────────────────────────────────────────────────────
    def release(self):
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

    @property
    def score(self) -> float:
        chars = len(self.text)
        signal = sum(1 for ch in self.text if ch.isalpha() or ch.isdigit())
        signal_ratio = signal / max(chars, 1)
        noise_penalty = max(0.0, 0.45 - signal_ratio) * 80.0
        return (
            self.avg_confidence
            + min(self.word_count * 1.8, 25.0)
            + min(chars / 12.0, 20.0)
            - noise_penalty
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
        raw = _config.get("ocr_tesseract_psm_modes", "6,4,11")
        parts = raw if isinstance(raw, (list, tuple)) else str(raw).split(",")
        modes = []
        for part in parts:
            try:
                mode = int(str(part).strip())
            except Exception:
                continue
            if mode not in modes and 3 <= mode <= 13:
                modes.append(mode)
        return modes or [6, 4, 11]

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
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
            denoised = cv2.fastNlMeansDenoising(clahe, h=12)
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
                (f"{base_name}-enhanced", sharp),
                (f"{base_name}-adaptive", adaptive),
                (f"{base_name}-otsu", otsu),
            ):
                key = (arr.shape, int(arr.mean()), int(arr.std()))
                if key in seen:
                    continue
                seen.add(key)
                variants.append((name, Image.fromarray(arr)))

        return variants[:6]

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
    def _candidate_from_data(cls, data: dict, variant: str, psm: int) -> TesseractCandidate:
        words_by_line = {}
        confidences = []

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

            block = data.get("block_num", [0])[i]
            par = data.get("par_num", [0])[i]
            line = data.get("line_num", [0])[i]
            words_by_line.setdefault((block, par, line), []).append(text)

        lines = [" ".join(words) for _, words in sorted(words_by_line.items())]
        normalized = cls._normalize_text("\n".join(lines))
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        word_count = sum(len(line.split()) for line in lines)
        return TesseractCandidate(normalized, avg_conf, word_count, variant, psm)

    def _best_candidate(self, pytesseract, pil_image: Image.Image, tess_lang: str) -> TesseractCandidate:
        best = TesseractCandidate("", 0.0, 0, "none", 6)
        min_conf = float(_config.get("ocr_tesseract_min_confidence", 45))

        for variant_name, variant_image in self._variant_images(pil_image):
            for psm in self._parse_psm_modes():
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
                candidate = self._candidate_from_data(data, variant_name, psm)
                if candidate.score > best.score:
                    best = candidate
                if candidate.avg_confidence >= 82 and candidate.word_count >= 8:
                    return candidate

        if best.text and best.avg_confidence >= min_conf:
            return best
        return best

    def extract_text(self, pil_image: Image.Image, lang: str = "eng") -> str:
        if not self._check_available():
            return f"OCR unavailable: {self._load_error}"

        try:
            import pytesseract

            t0 = time.time()
            tess_lang = self._resolve_lang(pytesseract, lang)
            best = self._best_candidate(pytesseract, pil_image, tess_lang)
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

            if not best.text:
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
                    obj._model_name   = _config.get("gemini_model_name", "gemini-3-flash-lite")
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
        pil_image.save(buf, format="JPEG", quality=90)
        image_bytes = buf.getvalue()

        t0 = time.time()
        timeout_s = self._timeout_s
        response = client.models.generate_content(
            model=self._model_name,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                "Read all text visible in this image aloud-ready. "
                "Return ONLY the text you see, with no extra commentary, "
                "no markdown, and no labels like 'Text:'.",
            ],
            config=types.GenerateContentConfig(
                http_options=types.HttpOptions(timeout=timeout_s * 1000),
            ),
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


def scan_and_read(lang: str = "eng") -> str:
    """
    Captures one frame and returns the extracted text.

    Args:
        lang : language code — "eng", "hin", "guj"  (or ISO: "en", "hi", "gu")

    Returns:
        Extracted text string, or an error message starting with "OCR".
    """
    if not _camera.open():
        return "Camera not available. Please check the ribbon cable connection."

    frame = _camera.capture()
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
                return gemini_text
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
