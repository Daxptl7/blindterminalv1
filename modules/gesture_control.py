"""
gesture_control.py — BlindAssist product gesture input
=======================================================
MediaPipe hand landmarks with rotation-independent joint-angle geometry,
camera framing guidance, explicit startup failures, and shuttered multi-frame
confirmation. The production vocabulary is intentionally small:

  MODE_SCAN      → open palm               → OCR scan
  MODE_VOICE     → index + middle fingers  → voice question
  OBJECT_DETECT  → index finger            → object detection
  GPS_CHECK      → index + middle + ring    → GPS mode
  STOP           → closed fist             → leave gesture mode

The physical shutter remains mandatory in Mode 5. A pose never launches a
feature merely because a hand passed through the camera view.
"""

import sys
import signal
import logging
import json
import os
import time
import cv2
import numpy as np
import platform
from collections import Counter, deque

from pathlib import Path
from typing import Optional, Callable

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "gesture.log"
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("GestureModule")

_settings = {}
try:
    with open(CONFIG_PATH, 'r') as f:
        _settings = json.load(f)
except Exception:
    pass

GESTURE_CONFIDENCE          = float(_settings.get("gesture_confidence", 0.60))
GESTURE_DISPLAY             = bool(_settings.get("gesture_display", False))
GESTURE_CAMERA_INDEX        = int(_settings.get("gesture_camera_index", 0))
GESTURE_CAMERA_BACKEND      = str(
    _settings.get("gesture_camera_backend", "auto")
).strip().lower()
GESTURE_BACKEND             = str(_settings.get("gesture_backend", "auto")).lower()
GESTURE_ALLOW_UNSAFE_TASKS  = bool(_settings.get("gesture_allow_unsafe_tasks", False))
GESTURE_ENABLE_SWIPES       = bool(_settings.get("gesture_enable_swipes", False))
DEFAULT_HAND_MODEL_PATH     = BASE_DIR / "models_local" / "mediapipe" / "hand_landmarker.task"

# ── MEDIAPIPE VIDEO MODE SETUP ──────────────────────────────────────────────
mp              = None
vision          = None
BaseOptions     = None
ClassicHands    = None
MEDIAPIPE_AVAILABLE         = False
TASKS_AVAILABLE             = False
CLASSIC_HANDS_AVAILABLE     = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    logger.error(f"mediapipe not available: {e}")

if MEDIAPIPE_AVAILABLE:
    try:
        from mediapipe.tasks.python import BaseOptions, vision
        TASKS_AVAILABLE = True
    except Exception as e:
        logger.warning(
            f"MediaPipe Tasks API unavailable, will use classic Hands fallback: {e}"
        )

    try:
        ClassicHands = mp.solutions.hands.Hands
        CLASSIC_HANDS_AVAILABLE = True
    except Exception:
        try:
            from mediapipe.python.solutions.hands import Hands as ClassicHands
            CLASSIC_HANDS_AVAILABLE = True
        except Exception:
            CLASSIC_HANDS_AVAILABLE = False

# ── LANDMARK INDICES (MediaPipe 21-point hand model) ────────────────────────
WRIST                       = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

_FINGER_CHAINS = (
    (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
)
_STRAIGHT_PIP_DEG = float(_settings.get("gesture_straight_pip_degrees", 150.0))
_STRAIGHT_DIP_DEG = float(_settings.get("gesture_straight_dip_degrees", 145.0))

# ── GEOMETRY HELPERS ─────────────────────────────────────────────────────────

def _dist(a, b) -> float:
    """3D Euclidean distance between two normalised landmarks."""
    return float(np.linalg.norm(_point(a) - _point(b)))


def _point(landmark) -> np.ndarray:
    return np.asarray(
        [float(landmark.x), float(landmark.y), float(getattr(landmark, "z", 0.0))],
        dtype=np.float64,
    )


def _joint_angle(a, joint, c) -> float:
    """Angle ABC in degrees; zero means the landmarks are degenerate."""
    first = _point(a) - _point(joint)
    second = _point(c) - _point(joint)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-9:
        return 0.0
    cosine = float(np.dot(first, second) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _finger_extended(lm, chain) -> bool:
    """Return True for a nearly straight finger, independent of screen Y."""
    mcp, pip, dip, tip = chain
    pip_angle = _joint_angle(lm[mcp], lm[pip], lm[dip])
    dip_angle = _joint_angle(lm[pip], lm[dip], lm[tip])
    reaches_out = _dist(lm[WRIST], lm[tip]) > _dist(lm[WRIST], lm[pip]) * 1.03
    return (
        pip_angle >= _STRAIGHT_PIP_DEG
        and dip_angle >= _STRAIGHT_DIP_DEG
        and reaches_out
    )


def _thumb_extended(lm) -> bool:
    mcp_angle = _joint_angle(lm[THUMB_CMC], lm[THUMB_MCP], lm[THUMB_IP])
    ip_angle = _joint_angle(lm[THUMB_MCP], lm[THUMB_IP], lm[THUMB_TIP])
    reaches_out = (
        _dist(lm[WRIST], lm[THUMB_TIP])
        > _dist(lm[WRIST], lm[THUMB_IP]) * 1.02
    )
    return mcp_angle >= 135.0 and ip_angle >= 145.0 and reaches_out


def _hand_size(lm) -> float:
    """
    Approximate hand size as the wrist-to-middle-MCP distance.
    Used to normalise distance thresholds so they work at any
    camera-to-hand distance.
    Returns at least 0.01 to avoid division-by-zero.
    """
    return max(_dist(lm[WRIST], lm[MIDDLE_MCP]), 0.01)


def framing_guidance(landmarks) -> str:
    """Return blind-friendly hand-placement feedback for the preview button."""
    if not landmarks or len(landmarks) < 21:
        return "NO_HAND"
    xs = [float(point.x) for point in landmarks]
    ys = [float(point.y) for point in landmarks]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    width, height = right - left, bottom - top
    if left < 0.02 or right > 0.98 or top < 0.02 or bottom > 0.98:
        return "HAND_CROPPED"
    if max(width, height) < 0.22:
        return "MOVE_CLOSER"
    center_x = (left + right) / 2
    if center_x < 0.35:
        return "MOVE_RIGHT"
    if center_x > 0.65:
        return "MOVE_LEFT"
    return "READY"


# ── GESTURE CLASSIFICATION ───────────────────────────────────────────────────

def classify_gesture(landmarks) -> Optional[str]:
    """
    Map 21 MediaPipe hand landmarks to BlindAssist gesture commands.

    Only poses connected to real Mode 5 actions are emitted. Finger extension
    is based on joint angles rather than whether a fingertip points upward in
    the image, so rotating the hand does not change its meaning.
    """
    if not landmarks or len(landmarks) < 21:
        return None

    lm = landmarks
    index, middle, ring, pinky = (
        _finger_extended(lm, chain) for chain in _FINGER_CHAINS
    )
    thumb = _thumb_extended(lm)

    # Open palm: the four long fingers matter; thumb pose varies substantially
    # between users and is not needed to distinguish this from other commands.
    if index and middle and ring and pinky:
        return "MODE_SCAN"

    if index and middle and ring and not pinky:
        return "GPS_CHECK"

    if index and middle and not ring and not pinky:
        return "MODE_VOICE"

    if index and not middle and not ring and not pinky:
        return "OBJECT_DETECT"

    # A thumb-only pose is not a fist. This guard prevents a thumbs-up from
    # accidentally leaving gesture mode.
    if not any((index, middle, ring, pinky)) and not thumb:
        return "STOP"

    return None


# ── SWIPE DETECTOR (cross-frame velocity tracking) ───────────────────────────
# These are used only inside detect_gesture() but defined here so they are
# importable and testable in isolation.

_SWIPE_WINDOW    = 8     # number of recent wrist-X positions to track
_SWIPE_THRESHOLD = 0.18  # minimum normalised X displacement to count as swipe
_SWIPE_MIN_SPEED = 0.02  # minimum displacement per frame to reject slow drift

def _make_swipe_state() -> dict:
    """Create a fresh swipe tracking state. Called once at loop startup."""
    return {
        "wrist_x_history": deque(maxlen=_SWIPE_WINDOW),
        "last_swipe": None,
        "last_swipe_at": 0.0,
    }


def _update_swipe_detector(state: dict, lm, now_ms: float) -> Optional[str]:
    """
    Push the current wrist X into the rolling window and check for swipe.
    Returns 'SWIPE_RIGHT', 'SWIPE_LEFT', or None.
    A swipe cooldown of 1200ms prevents double-triggers.
    """
    if lm is None:
        state["wrist_x_history"].clear()
        return None

    state["wrist_x_history"].append(lm[WRIST].x)

    if len(state["wrist_x_history"]) < _SWIPE_WINDOW:
        return None                              # not enough history yet

    xs       = list(state["wrist_x_history"])
    delta    = xs[-1] - xs[0]                   # total displacement
    per_frame = abs(delta) / _SWIPE_WINDOW       # average per-frame speed

    if abs(delta) < _SWIPE_THRESHOLD or per_frame < _SWIPE_MIN_SPEED:
        return None                              # too slow or too short

    if now_ms - state["last_swipe_at"] < 1200:
        return None                              # still in cooldown

    gesture = "SWIPE_RIGHT" if delta > 0 else "SWIPE_LEFT"
    state["wrist_x_history"].clear()            # reset after emit
    state["last_swipe"]    = gesture
    state["last_swipe_at"] = now_ms
    return gesture


class GestureVoteWindow:
    """Time-bounded majority vote that counts unrecognized frames as misses."""

    def __init__(self, window_ms: int = 1000, fresh_ms: int = 300,
                 min_frames: int = 5, min_support: float = 0.60):
        self.window_ms = max(100, int(window_ms))
        self.fresh_ms = max(50, int(fresh_ms))
        self.min_frames = max(1, int(min_frames))
        self.min_support = min(max(float(min_support), 0.50), 1.0)
        self.samples = deque(maxlen=120)

    def add(self, timestamp_ms: float, gesture: Optional[str]):
        self.samples.append((float(timestamp_ms), gesture))

    def clear(self):
        self.samples.clear()

    def vote(self, now_ms: float) -> Optional[str]:
        window = [
            (timestamp, gesture)
            for timestamp, gesture in self.samples
            if now_ms - timestamp <= self.window_ms
        ]
        if len(window) < self.min_frames:
            return None
        seen = [gesture for _, gesture in window if gesture]
        if not seen:
            return None
        winner, count = Counter(seen).most_common(1)[0]
        if not any(
            gesture == winner and now_ms - timestamp <= self.fresh_ms
            for timestamp, gesture in window
        ):
            return None
        if count < self.min_frames or count / len(window) < self.min_support:
            return None
        return winner


# ── CAMERA HELPERS ────────────────────────────────────────────────────────────

def _callback_allows_continue(callback_fn: Optional[Callable], gesture: str) -> bool:
    if callback_fn is None:
        print(f"[GESTURE] {gesture}")
        return True
    result = callback_fn(gesture)
    return result is not False


# OpenCV probing deliberately excludes Pi ISP/metadata nodes: several open
# successfully and then deliver nothing. In auto mode, a real USB stream is
# preferred and Camera Module 3 is the command-line rpicam fallback.
_NON_CAMERA_NODE_HINTS = ("unicam", "bcm2835-isp", "rpivid", "pispbe",
                          "codec", "hevc", "isp", "stat", "meta")


def _v4l2_nodes() -> list:
    """(is_usb, index) for every /dev/video* node that could be a camera.

    A Raspberry Pi publishes a dozen video nodes — ISP stages, codecs and
    metadata streams alongside real cameras — so the index that happens to be
    free is not the index that has a lens on it. Reading the driver name and
    the bus each node sits on is what separates the wired webcam from the rest.
    """
    nodes = []
    base = "/sys/class/video4linux"
    if not os.path.isdir(base):
        return nodes                       # not Linux: caller scans blindly

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

    # USB first, then by index, so the wired camera wins even when a CSI node
    # sorts ahead of it.
    nodes.sort(key=lambda item: (not item[0], item[1]))
    return nodes


def _probe_v4l2(index: int):
    """Open a V4L2 index and confirm it actually yields a frame, else None.

    isOpened() alone is not proof. Several Pi video nodes open cleanly and then
    never produce an image, and accepting one of those is how gesture mode ends
    up watching a camera that shows nothing.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    for _ in range(5):                     # warmup doubles as proof of life
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
    cap.release()
    return None


def _open_usb_camera(camera_index: Optional[int]):
    """Open a wired USB camera. The configured index is only a hint.

    gesture_camera_index was 8 on the device — a node that does not exist — and
    the mode reported "Camera failed: index 8" and gave up while a working USB
    camera was plugged in. The index is tried first if it is real, and
    otherwise the USB cameras found on the bus are tried in order.
    """
    nodes = _v4l2_nodes()
    usb_indices = [index for is_usb, index, _ in nodes if is_usb]
    other_indices = [index for is_usb, index, _ in nodes if not is_usb]

    if nodes:
        logger.info("Video nodes: " + ", ".join(
            f"video{index}{'(usb)' if is_usb else ''}"
            f"{' ' + name if name else ''}" for is_usb, index, name in nodes))

    order = []
    if camera_index is not None and camera_index >= 0:
        order.append(camera_index)
    order += [i for i in usb_indices if i not in order]
    order += [i for i in other_indices if i not in order]
    sysfs_present = sys.platform.startswith("linux") and os.path.isdir(
        "/sys/class/video4linux"
    )
    if not nodes and not sysfs_present:   # no sysfs (macOS): scan blindly
        order += [i for i in range(0, 11) if i not in order]

    # Probing absent indices makes OpenCV shout on stderr; quiet it for the
    # scan so the successful result is not buried in warnings.
    previous_level = None
    try:
        previous_level = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        previous_level = None

    try:
        for index in order:
            cap = _probe_v4l2(index)
            if cap is None:
                continue
            if index != camera_index:
                logger.warning(
                    f"gesture_camera_index={camera_index} is not a working "
                    f"camera; using index {index}. Set gesture_camera_index "
                    f"to {index} in settings.json to skip this search.")
            else:
                logger.info(f"Gesture camera ready on index {index}.")
            return cap
    finally:
        if previous_level is not None:
            try:
                cv2.utils.logging.setLogLevel(previous_level)
            except Exception:
                pass

    logger.warning("No working USB camera found for gesture control.")
    return None


def _open_rpicam_camera():
    """Reuse the tested Camera Module 3 MJPEG adapter from object detection."""
    try:
        from modules.object_detection import (
            _RpicamMjpegCapture,
            _find_rpicam_binary,
        )
    except Exception as exc:
        logger.warning(f"Raspberry Pi camera adapter unavailable: {exc}")
        return None
    binary = _find_rpicam_binary()
    if not binary:
        return None
    try:
        camera = int(_settings.get("gesture_rpicam_camera", 0))
        fps = int(_settings.get("gesture_camera_fps", 15))
        cap = _RpicamMjpegCapture(binary, fps=fps, camera=camera)
        for _ in range(3):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                logger.info(f"Gesture camera ready through {Path(binary).name}.")
                return cap
        cap.release()
    except Exception as exc:
        logger.warning(f"Raspberry Pi gesture camera failed: {exc}")
    return None


def _open_camera(camera_index: Optional[int]):
    selected = GESTURE_CAMERA_BACKEND
    if selected not in {"auto", "usb", "opencv", "rpicam"}:
        logger.warning(f"Unknown gesture_camera_backend={selected!r}; using auto.")
        selected = "auto"

    if selected in {"auto", "usb", "opencv"}:
        cap = _open_usb_camera(camera_index)
        if cap is not None:
            return cap
        if selected != "auto":
            return None
    if selected in {"auto", "rpicam"}:
        cap = _open_rpicam_camera()
        if cap is not None:
            return cap

    logger.error(
        "No gesture camera returned frames. Check the USB camera or rpicam-apps."
    )
    return None


def _create_task_detector(model_path: Path):
    base_options = BaseOptions(
        model_asset_path=str(model_path),
        delegate=BaseOptions.Delegate.CPU,
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=GESTURE_CONFIDENCE,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def _resolve_hand_model_path() -> Path:
    """Return the configured Tasks model path, including the old location.

    The model is intentionally kept under models_local so it is not committed
    to git. Existing installations that placed it in config/ continue to work.
    """
    configured = str(_settings.get("gesture_hand_model_path", "")).strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = BASE_DIR / path
        return path.resolve()

    if DEFAULT_HAND_MODEL_PATH.exists():
        return DEFAULT_HAND_MODEL_PATH
    legacy = CONFIG_PATH.parent / "hand_landmarker.task"
    if legacy.exists():
        return legacy
    return DEFAULT_HAND_MODEL_PATH


def _create_classic_detector():
    return ClassicHands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=GESTURE_CONFIDENCE,
        min_tracking_confidence=0.5,
    )


def _tasks_backend_allowed(model_path: Path) -> bool:
    if not TASKS_AVAILABLE or not model_path.exists():
        return False
    if platform.system() == "Darwin" and not GESTURE_ALLOW_UNSAFE_TASKS:
        logger.warning(
            "Skipping MediaPipe Tasks on macOS because this build can abort in "
            "the native Metal graph. Set gesture_allow_unsafe_tasks=true only "
            "after validating this machine."
        )
        return False
    return True


def diagnostics(probe_camera: bool = False, load_backend: bool = False) -> dict:
    """Return actionable gesture readiness information without starting Mode 5."""
    model_path = _resolve_hand_model_path()
    info = {
        "mediapipe_ready": MEDIAPIPE_AVAILABLE,
        "classic_backend_ready": CLASSIC_HANDS_AVAILABLE,
        "tasks_backend_ready": TASKS_AVAILABLE,
        "model_path": str(model_path),
        "model_ready": model_path.is_file(),
        "backend": None,
        "backend_ready": False,
        "backend_error": None,
        "camera_backend": GESTURE_CAMERA_BACKEND,
        "camera_ready": None,
        "camera_error": None,
    }
    if not MEDIAPIPE_AVAILABLE:
        info["backend_error"] = "mediapipe is not installed"
    elif CLASSIC_HANDS_AVAILABLE and GESTURE_BACKEND in {"auto", "classic"}:
        info["backend"] = "classic"
        info["backend_ready"] = True
    elif TASKS_AVAILABLE and GESTURE_BACKEND in {"auto", "tasks"}:
        info["backend"] = "tasks"
        if not model_path.is_file():
            info["backend_error"] = (
                "hand landmarker model is missing; run "
                "python3 scripts/prepare_gesture_control.py"
            )
        elif not _tasks_backend_allowed(model_path):
            info["backend_error"] = (
                "MediaPipe Tasks is disabled on this macOS installation"
            )
        else:
            info["backend_ready"] = True
    else:
        info["backend_error"] = "no compatible MediaPipe hand backend"

    if load_backend and info["backend_ready"]:
        detector = None
        try:
            detector = (
                _create_classic_detector()
                if info["backend"] == "classic"
                else _create_task_detector(model_path)
            )
        except Exception as exc:
            info["backend_ready"] = False
            info["backend_error"] = str(exc)
        finally:
            if detector is not None:
                detector.close()

    if probe_camera:
        cap = _open_camera(GESTURE_CAMERA_INDEX)
        if cap is None:
            info["camera_ready"] = False
            info["camera_error"] = (
                "no camera returned frames; check camera permission, USB, or rpicam-apps"
            )
        else:
            try:
                ok, frame = cap.read()
                info["camera_ready"] = bool(ok and frame is not None)
                if not info["camera_ready"]:
                    info["camera_error"] = "camera opened but did not return an image"
            finally:
                cap.release()
    return info


# ── MAIN DETECTION LOOP ───────────────────────────────────────────────────────

def detect_gesture(callback_fn: Optional[Callable] = None,
                   camera_index: Optional[int] = None,
                   display: Optional[bool] = None,
                   stop_event=None,
                   max_frames: Optional[int] = None,
                   capture_check: Optional[Callable] = None):
    """
    High-performance gesture detection using VIDEO mode.
    Processes at 30 FPS on Raspberry Pi 5.

    The callback receives one gesture string at a time (debounced).
    Return False from the callback to stop the loop.
    Return anything else (True, None) to continue.

    Gesture strings emitted:
        MODE_SCAN, MODE_VOICE, OBJECT_DETECT, GPS_CHECK, STOP
    """
    if not MEDIAPIPE_AVAILABLE:
        logger.error("MediaPipe unavailable")
        if callback_fn is not None:
            _callback_allows_continue(callback_fn, "MEDIAPIPE_UNAVAILABLE")
        return

    if camera_index is None:
        camera_index = GESTURE_CAMERA_INDEX
    if display is None:
        display = GESTURE_DISPLAY

    requested_backend = GESTURE_BACKEND
    if requested_backend not in {"auto", "classic", "tasks"}:
        logger.warning(f"Unknown gesture_backend '{requested_backend}', using auto")
        requested_backend = "auto"

    model_path = _resolve_hand_model_path()
    detector   = None
    backend    = None

    if requested_backend in {"auto", "classic"} and CLASSIC_HANDS_AVAILABLE:
        try:
            detector = _create_classic_detector()
            backend  = "classic"
        except Exception as e:
            logger.warning(f"MediaPipe classic Hands setup failed: {e}")

    if (detector is None
            and requested_backend in {"auto", "tasks"}
            and _tasks_backend_allowed(model_path)):
        try:
            detector = _create_task_detector(model_path)
            backend  = "tasks"
        except Exception as e:
            logger.warning(f"MediaPipe Tasks setup failed, using classic Hands fallback: {e}")

    if detector is None:
        model_missing = (
            requested_backend in {"auto", "tasks"}
            and TASKS_AVAILABLE
            and not model_path.is_file()
            and not (
                requested_backend in {"auto", "classic"}
                and CLASSIC_HANDS_AVAILABLE
            )
        )
        logger.error(
            "MediaPipe hand backend unavailable: "
            + (
                f"model missing at {model_path}; run "
                "python3 scripts/prepare_gesture_control.py."
                if model_missing else
                "install a compatible MediaPipe build or validate the Tasks backend."
            )
        )
        if callback_fn is not None:
            _callback_allows_continue(
                callback_fn,
                "HAND_MODEL_MISSING" if model_missing else "BACKEND_UNAVAILABLE",
            )
        return

    cap = _open_camera(camera_index)
    if cap is None:
        # _open_camera has already logged what it tried and what to check.
        # Tell the caller, so the mode can say it out loud instead of dropping
        # the user back at the menu with no explanation.
        if callback_fn is not None:
            try:
                callback_fn("CAMERA_UNAVAILABLE")
            except Exception as e:
                logger.debug(f"Camera-failure callback raised: {e}")
        detector.close()
        return

    # Warmup — let AGC / AWB settle
    for _ in range(5):
        cap.read()

    logger.info(f"Gesture detection active ({backend} backend)")

    # ── Per-gesture debounce / cooldown state ─────────────────────────────
    stable_gesture = None
    stable_since   = 0.0
    last_emitted   = None
    last_emit_at   = 0.0
    DEBOUNCE_MS    = int(_settings.get("gesture_debounce_ms", 600))
    COOLDOWN_MS    = int(_settings.get("gesture_cooldown_ms", 1500))

    # ── Shutter mode ──────────────────────────────────────────────────────
    # When capture_check is supplied nothing is ever emitted on its own: the
    # loop watches, and reports only when the caller asks. Recognising a hand
    # shape reliably enough to act on it unprompted turned out to be the weak
    # link — a button press is not ambiguous, so the user holds the pose and
    # presses, and the pose under the shutter is the one that counts.
    #
    # The reading is a majority vote over the last CAPTURE_WINDOW_MS rather
    # than the single frame the press landed on. Classification flickers
    # frame to frame, and pressing a button nudges the hand slightly, so the
    # one frame at the instant of the press is the least trustworthy sample
    # available.
    shutter_mode   = capture_check is not None
    CAPTURE_WINDOW_MS = int(_settings.get("gesture_capture_window_ms", 1000))
    CAPTURE_FRESH_MS = int(_settings.get("gesture_capture_fresh_ms", 300))
    CAPTURE_MIN_FRAMES = int(_settings.get("gesture_capture_min_frames", 5))
    CAPTURE_MIN_SUPPORT = float(
        _settings.get("gesture_capture_min_support", 0.60)
    )
    CAPTURE_MIN_SUPPORT = min(max(CAPTURE_MIN_SUPPORT, 0.50), 1.0)
    vote_window = GestureVoteWindow(
        window_ms=CAPTURE_WINDOW_MS,
        fresh_ms=CAPTURE_FRESH_MS,
        min_frames=CAPTURE_MIN_FRAMES,
        min_support=CAPTURE_MIN_SUPPORT,
    )
    latest_guidance = "NO_HAND"

    # ── Swipe detector state ───────────────────────────────────────────────
    swipe_state = _make_swipe_state()

    # ── FPS counter ───────────────────────────────────────────────────────
    frame_times      = deque(maxlen=30)
    last_timestamp_ms = 0
    frame_count      = 0
    failed_reads     = 0

    try:
        while not (stop_event and stop_event.is_set()):
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera frame read failed")
                failed_reads += 1
                if failed_reads >= 5:
                    if callback_fn is not None:
                        _callback_allows_continue(callback_fn, "FRAME_READ_FAILED")
                    break
                continue
            failed_reads = 0

            # Mirror so hand movement feels natural (right swipe = right on screen)
            frame = cv2.flip(frame, 1)

            # ── Convert and run detection ──────────────────────────────────
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands = []
            hand_lm_raw = None  # raw landmark list for swipe detector

            if backend == "tasks":
                timestamp_ms = int(time.monotonic() * 1000)
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results  = detector.detect_for_video(mp_image, timestamp_ms)
                hands    = results.hand_landmarks or []
            else:
                rgb.flags.writeable = False
                results = detector.process(rgb)
                if results.multi_hand_landmarks:
                    hands = [hand.landmark for hand in results.multi_hand_landmarks]

            current = None
            latest_guidance = "NO_HAND"

            for hand in hands:
                hand_lm_raw = hand
                latest_guidance = framing_guidance(hand)
                # A cropped or tiny hand can produce plausible landmarks but a
                # dangerously unreliable pose. Keep it out of the vote and let
                # Preview explain how to correct the framing.
                if latest_guidance == "READY":
                    current = classify_gesture(hand)

                if display:
                    h, w = frame.shape[:2]
                    for lm_pt in hand:
                        cv2.circle(
                            frame,
                            (int(lm_pt.x * w), int(lm_pt.y * h)),
                            3, (0, 255, 0), -1
                        )
                break  # Only process first detected hand

            # ── Swipe detection (cross-frame, runs every frame) ───────────
            now_ms  = time.time() * 1000
            swipe = None
            if GESTURE_ENABLE_SWIPES and not shutter_mode:
                swipe = _update_swipe_detector(swipe_state, hand_lm_raw, now_ms)

            # Swipe overrides classify_gesture if one was detected this frame
            if swipe is not None:
                current = swipe

            # ── Shutter mode: report only when the caller asks ─────────────
            if shutter_mode:
                vote_window.add(now_ms, current)
                request = None
                try:
                    request = capture_check()
                except Exception as e:
                    logger.debug(f"capture_check raised: {e}")

                if request == "preview":
                    captured = vote_window.vote(now_ms)
                    report = (f"PREVIEW:{captured}" if captured
                              else f"PREVIEW_GUIDANCE:{latest_guidance}")
                    if not _callback_allows_continue(callback_fn, report):
                        break
                elif request == "capture":
                    captured = vote_window.vote(now_ms)
                    logger.info(f"Shutter pressed — captured gesture: {captured}")
                    report = (captured if captured
                              else f"NO_GESTURE:{latest_guidance}")
                    if not _callback_allows_continue(callback_fn, report):
                        break
                    vote_window.clear()       # do not reuse a spent pose

                if display:
                    label = current or "None"
                    cv2.putText(frame, label, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow("Gesture", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_times.append(time.time() - loop_start)
                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    break
                continue

            # ── Debounce / cooldown logic (unchanged from original) ────────
            if current != stable_gesture:
                stable_gesture = current
                stable_since   = now_ms
            elif current and now_ms - stable_since >= DEBOUNCE_MS:
                can_emit = (
                    current != last_emitted
                    or now_ms - last_emit_at >= COOLDOWN_MS
                )
                if can_emit:
                    logger.info(f"Gesture: {current}")
                    if not _callback_allows_continue(callback_fn, current):
                        break
                    last_emitted = current
                    last_emit_at = now_ms
            else:
                if current is None:
                    last_emitted = None

            # ── Optional display overlay ───────────────────────────────────
            if display:
                label = current or "None"
                cv2.putText(
                    frame, label,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                total = sum(frame_times)
                fps   = len(frame_times) / total if total > 0 else 0
                cv2.putText(
                    frame, f"{fps:.1f} FPS",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
                cv2.imshow("Gesture", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_times.append(time.time() - loop_start)
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break

    finally:
        detector.close()
        cap.release()
        if display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        logger.info("Gesture detection stopped")


# ── STANDALONE TEST ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    print("BlindAssist Gesture Control — Enhanced Edition")
    print("=" * 54)
    print("GESTURE REFERENCE:")
    print("  Open Palm                  → MODE_SCAN      (OCR)")
    print("  Two Fingers (V sign)       → MODE_VOICE")
    print("  Index Finger               → OBJECT_DETECT")
    print("  Three Fingers              → GPS_CHECK")
    print("  Closed Fist                → STOP")
    print("=" * 54)
    print("Press Q in the display window to quit.")
    detect_gesture(display=True)
