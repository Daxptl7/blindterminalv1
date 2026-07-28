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

    tmp = os.path.join(tempfile.gettempdir(), "aet_ocr_cap.jpg")
    cmd = [
        RPICAM_BIN,
        "-o", tmp,
        "-n",                        # no preview window
        "-t", "1500",                # 1.5 s settle time (ms)
        "--width",   "1920",
        "--height",  "1080",
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
        if self._cam_type == "rpicam":
            return _rpicam_capture()

        if self._cam_type == "usb" and self._usb_cap:
            time.sleep(0.5)               # brief settle for exposure
            ret, frame = self._usb_cap.read()
            if ret:
                return frame
            logger.error("USB camera read() returned False.")
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
# Tesseract OCR Engine  (fast local default)
# ─────────────────────────────────────────────────────────────────────────────
class TesseractOCREngine:
    """
    Fast local OCR path for Raspberry Pi and laptop use.

    Surya OCR 2 can be accurate, but the app logs show it can spend more than
    a minute inside inference on the Pi. Tesseract is already a documented
    dependency for this project and is the right default for interactive scans.
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
    def _prepare_image(pil_image: Image.Image) -> Image.Image:
        image = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        height, width = gray.shape[:2]
        if width < 1400:
            scale = min(2.0, 1400 / max(width, 1))
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        processed = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        return Image.fromarray(processed)

    def extract_text(self, pil_image: Image.Image, lang: str = "eng") -> str:
        if not self._check_available():
            return f"OCR unavailable: {self._load_error}"

        try:
            import pytesseract

            t0 = time.time()
            tess_lang = TESSERACT_LANG_MAP.get(lang, "eng")
            processed = self._prepare_image(pil_image)
            text = pytesseract.image_to_string(
                processed,
                lang=tess_lang,
                config="--oem 3 --psm 6",
            )
            text = re.sub(r"\s+", " ", text).strip()
            elapsed_ms = int((time.time() - t0) * 1000)
            logger.info(f"Tesseract OCR done in {elapsed_ms} ms | chars={len(text)}")

            if not text:
                return "No text detected in the image."
            return text

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
    catch that and fall back to the local SuryaOCREngine.
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
        response = client.models.generate_content(
            model=self._model_name,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                "Read all text visible in this image aloud-ready. "
                "Return ONLY the text you see, with no extra commentary, "
                "no markdown, and no labels like 'Text:'.",
            ],
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

    # In auto mode, Gemini is still the first choice when configured, but any
    # failure drops to fast local OCR instead of the heavy Surya model.
    if engine_name in ("auto", "gemini") and _config.get("gemini_api_key"):
        try:
            text = _gemini_engine.extract_text(pil_image, lang=lang)
            if text and not text.lower().startswith(("ocr unavailable", "ocr error")):
                return text
        except Exception as e:
            logger.warning(f"Gemini OCR unavailable, falling back to local OCR: {e}")

    if engine_name in ("auto", "gemini", "tesseract"):
        text = _tesseract_engine.extract_text(pil_image, lang=lang)
        text_lower = text.lower() if text else ""
        if (
            text
            and not text_lower.startswith(("ocr unavailable", "ocr error"))
            and "no text detected" not in text_lower
        ):
            return text

        if not _config.get("ocr_surya_fallback", False):
            return text
        logger.warning(f"Tesseract OCR did not produce usable text, trying Surya: {text}")

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

    print("Camera open. Scanning …")
    result = scan_and_read(lang="eng")

    print("\nOCR Output:")
    print("-" * 55)
    print(result)
    print("-" * 55)

    release_camera()
    print("Done.")
