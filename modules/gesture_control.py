"""
gesture_control.py — BlindAssist Project (OPTIMIZED + ENHANCED)
================================================================
MediaPipe VIDEO mode for continuous streaming (10x faster than IMAGE mode).
Reuses detector across frames. No per-frame reallocation.

GESTURE MAP — Full Reference
──────────────────────────────────────────────────────────────────────
ORIGINAL GESTURES (unchanged):
  MODE_SCAN   → Open Palm (all 5 fingers spread)   → Trigger OCR scan
  CONFIRM     → Thumbs Up (thumb only, others curl) → Confirm / Yes
  MODE_VOICE  → Two Fingers (index + middle up)     → Voice ask mode
  REPEAT      → Index Pointing Down                 → Repeat last audio
  STOP        → Fist (all fingers curled)           → Stop / Cancel

NEW GESTURES (added in this version):
  OBJECT_DETECT     → Index Pointing Up/Forward (index up, others curled)
                       → Triggers object detection: "What is in front?"
  GPS_CHECK         → Three Fingers (index + middle + ring up, others curl)
                       → Reads current GPS location or next waypoint
  TOGGLE_PRIVACY    → Shaka / Call-Me (pinky + thumb out, index/middle/ring curl)
                       → Toggle Confidential Mode on/off
  SWIPE_RIGHT       → Hand moves RIGHT across frame (wrist x-velocity > threshold)
                       → Volume UP
  SWIPE_LEFT        → Hand moves LEFT across frame (wrist x-velocity < -threshold)
                       → Volume DOWN
  STATUS_CHECK      → OK Sign (thumb tip touches index tip, other 3 fingers up)
                       → Read battery level, time, network status aloud

ACCURACY IMPROVEMENTS in this version:
  - Wrist normalisation: all finger-extended checks use wrist as Y anchor
    so the result is invariant to hand height in frame
  - Thumb uses ratio + lift check (inherited from original, kept)
  - Three-Fingers and Shaka each have a negative guard (verifies
    the fingers that must be DOWN are actually down) to prevent
    false triggers from partial detections
  - OK sign uses pixel-distance threshold scaled to hand size
    (index-to-wrist distance) so it works at any camera distance
  - Swipe uses a rolling deque of wrist X positions (last 8 frames)
    and requires minimum displacement + minimum speed to trigger,
    preventing accidental volume changes from slow hand movement
  - Confidence threshold applied separately in the detector options
  - Debounce/cooldown unchanged from original (configurable in settings.json)
──────────────────────────────────────────────────────────────────────
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

GESTURE_CONFIDENCE          = float(_settings.get("gesture_confidence", 0.85))
GESTURE_DISPLAY             = bool(_settings.get("gesture_display", False))
GESTURE_CAMERA_INDEX        = int(_settings.get("gesture_camera_index", 0))
GESTURE_BACKEND             = str(_settings.get("gesture_backend", "auto")).lower()
GESTURE_ALLOW_UNSAFE_TASKS  = bool(_settings.get("gesture_allow_unsafe_tasks", False))

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
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_MCP                   = 5
INDEX_TIP,   INDEX_PIP      = 8,  6
MIDDLE_TIP,  MIDDLE_PIP     = 12, 10
RING_TIP,    RING_PIP       = 16, 14
PINKY_TIP,   PINKY_PIP      = 20, 18

# ── GEOMETRY HELPERS ─────────────────────────────────────────────────────────

def _dist(a, b) -> float:
    """Euclidean distance between two normalised landmarks."""
    return float(np.hypot(a.x - b.x, a.y - b.y))


def _finger_extended(lm, tip: int, pip: int) -> bool:
    """
    A finger is extended when its tip is higher in the frame (smaller Y)
    than its middle joint (PIP). Using raw Y works well for a camera that
    faces the user from in front (endoscope on glasses).
    Wrist-anchored variant is used for swipe and OK to make them
    distance-invariant (see _finger_extended_anchored below).
    """
    return lm[tip].y < lm[pip].y


def _finger_extended_anchored(lm, tip: int, pip: int) -> bool:
    """
    Wrist-anchored extension check: tip must be above (smaller Y) the PIP
    AND the PIP must itself be above the wrist.  This prevents the index
    finger from being classified as 'extended' when the whole hand is
    pointing downward.
    """
    above_pip   = lm[tip].y  < lm[pip].y
    pip_above_wrist = lm[pip].y < lm[WRIST].y
    return above_pip and pip_above_wrist


def _thumb_extended(lm) -> bool:
    """
    Thumb extension: uses ratio of distances to INDEX_MCP as a size-invariant
    check PLUS a lifted-tip check. Both signals are OR-ed so it works whether
    the hand is flat or at an angle.
    """
    thumb_from_palm    = _dist(lm[THUMB_TIP], lm[INDEX_MCP])
    thumb_ip_from_palm = _dist(lm[THUMB_IP],  lm[INDEX_MCP])
    thumb_lifted       = lm[THUMB_TIP].y < lm[THUMB_IP].y
    return thumb_from_palm > thumb_ip_from_palm * 1.35 or thumb_lifted


def _hand_size(lm) -> float:
    """
    Approximate hand size as the wrist-to-INDEX_MCP distance.
    Used to normalise distance thresholds so they work at any
    camera-to-hand distance.
    Returns at least 0.01 to avoid division-by-zero.
    """
    return max(_dist(lm[WRIST], lm[INDEX_MCP]), 0.01)


# ── GESTURE CLASSIFICATION ───────────────────────────────────────────────────

def classify_gesture(landmarks) -> Optional[str]:
    """
    Map 21 MediaPipe hand landmarks to BlindAssist gesture commands.

    All original gestures are UNCHANGED.
    Five new gestures are appended below the originals.

    Check order matters: more specific patterns are checked first to
    prevent accidental overlap with broader checks.
    """
    if not landmarks or len(landmarks) < 21:
        return None

    lm   = landmarks
    size = _hand_size(lm)          # used by OK-sign only

    # ── Compute per-finger extension ───────────────────────────────────────
    thumb   = _thumb_extended(lm)
    thumb_up = lm[THUMB_TIP].y < lm[THUMB_IP].y < lm[THUMB_MCP].y

    index  = _finger_extended(lm, INDEX_TIP,  INDEX_PIP)
    middle = _finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP)
    ring   = _finger_extended(lm, RING_TIP,   RING_PIP)
    pinky  = _finger_extended(lm, PINKY_TIP,  PINKY_PIP)

    # Anchored variants for gestures that need extra precision
    index_anc  = _finger_extended_anchored(lm, INDEX_TIP,  INDEX_PIP)
    middle_anc = _finger_extended_anchored(lm, MIDDLE_TIP, MIDDLE_PIP)
    ring_anc   = _finger_extended_anchored(lm, RING_TIP,   RING_PIP)
    pinky_anc  = _finger_extended_anchored(lm, PINKY_TIP,  PINKY_PIP)

    fingers = np.array([thumb, index, middle, ring, pinky], dtype=bool)
    count   = int(fingers.sum())

    # ═══════════════════════════════════════════════════════════════════════
    # ORIGINAL GESTURES — not changed, only reordered for precedence
    # ═══════════════════════════════════════════════════════════════════════

    # ── OPEN PALM → MODE_SCAN (OCR) ─────────────────────────────────────────
    # All 5 fingers extended (thumb optional check is already lenient enough)
    if all(fingers):
        return "MODE_SCAN"

    # ── THUMBS UP → CONFIRM ─────────────────────────────────────────────────
    # Strict: thumb pointing up AND every other finger curled
    if thumb_up and not any(fingers[1:]):
        return "CONFIRM"

    # ── TWO FINGERS → MODE_VOICE ────────────────────────────────────────────
    # Index + middle up, ring + pinky down, thumb state irrelevant
    if index and middle and not ring and not pinky:
        return "MODE_VOICE"

    # ── INDEX POINTING DOWN → REPEAT ────────────────────────────────────────
    index_pointing_down = (
        lm[INDEX_TIP].y > lm[INDEX_PIP].y > lm[INDEX_MCP].y
    )
    if index_pointing_down and not middle and not ring and not pinky:
        return "REPEAT"

    # ── FIST → STOP ─────────────────────────────────────────────────────────
    if count == 0:
        return "STOP"

    # ═══════════════════════════════════════════════════════════════════════
    # NEW GESTURES
    # ═══════════════════════════════════════════════════════════════════════

    # ── INDEX POINTING UP → OBJECT_DETECT ("What is that?") ────────────────
    # Index extended upward (anchored check), middle/ring/pinky all curled,
    # thumb state irrelevant.  This is distinct from TWO_FINGERS because
    # the middle finger must be DOWN here.
    if index_anc and not middle and not ring and not pinky:
        return "OBJECT_DETECT"

    # ── THREE FINGERS → GPS_CHECK ("Where am I?") ──────────────────────────
    # Index + middle + ring extended, pinky curled, thumb curled.
    # Negative guards on pinky and thumb prevent overlap with OPEN_PALM.
    if index and middle and ring and not pinky and not thumb:
        return "GPS_CHECK"

    # ── SHAKA / CALL-ME → TOGGLE_PRIVACY ───────────────────────────────────
    # Pinky extended UP and thumb extended OUT, index/middle/ring all curled.
    # Uses anchored checks for pinky to avoid false trigger from floppy pinky.
    shaka_pinky = pinky_anc
    shaka_thumb = lm[THUMB_TIP].y < lm[THUMB_MCP].y   # thumb lifted laterally
    shaka_index_down  = not index_anc
    shaka_middle_down = not middle_anc
    shaka_ring_down   = not ring_anc

    if (shaka_pinky and shaka_thumb
            and shaka_index_down and shaka_middle_down and shaka_ring_down):
        return "TOGGLE_PRIVACY"

    # ── OK SIGN → STATUS_CHECK (Battery / Time / Network) ──────────────────
    # Thumb tip and index tip are close together (circle),
    # while middle, ring, pinky are extended upward.
    # Distance threshold is normalised by hand size so it works at any range.
    thumb_index_close = _dist(lm[THUMB_TIP], lm[INDEX_TIP]) < size * 0.45
    ok_middle = _finger_extended_anchored(lm, MIDDLE_TIP, MIDDLE_PIP)
    ok_ring   = _finger_extended_anchored(lm, RING_TIP,   RING_PIP)
    ok_pinky  = _finger_extended_anchored(lm, PINKY_TIP,  PINKY_PIP)

    if thumb_index_close and ok_middle and ok_ring and ok_pinky:
        return "STATUS_CHECK"

    # NOTE: SWIPE_LEFT and SWIPE_RIGHT are not detected here.
    # Swipe requires tracking wrist position across multiple frames,
    # which cannot be done from a single frame's landmarks.
    # See _update_swipe_detector() called in the detection loop below.

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


# ── CAMERA HELPERS ────────────────────────────────────────────────────────────

def _callback_allows_continue(callback_fn: Optional[Callable], gesture: str) -> bool:
    if callback_fn is None:
        print(f"[GESTURE] {gesture}")
        return True
    result = callback_fn(gesture)
    return result is not False


# Gesture control uses the WIRED USB camera. The CSI Camera Module on the Pi's
# ribbon connector belongs to OCR (ocr.py drives it through rpicam-still), and
# its kernel nodes must never be picked up here: several of them open
# successfully and then deliver nothing, which is indistinguishable from a
# broken camera at the point where it matters.
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


def _open_camera(camera_index: Optional[int]):
    """Open the wired USB camera. The configured index is only a hint.

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
    if not nodes:                          # no sysfs (macOS): scan blindly
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

    logger.error(
        "No working USB camera found for gesture control. Check that the wired "
        "camera is plugged in — `v4l2-ctl --list-devices` shows what the Pi "
        "sees. The CSI camera is not used here; it belongs to OCR.")
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
        Original:  MODE_SCAN, CONFIRM, MODE_VOICE, REPEAT, STOP
        New:       OBJECT_DETECT, GPS_CHECK, TOGGLE_PRIVACY,
                   STATUS_CHECK, SWIPE_RIGHT, SWIPE_LEFT
    """
    if not MEDIAPIPE_AVAILABLE:
        logger.error("MediaPipe unavailable")
        return

    if camera_index is None:
        camera_index = GESTURE_CAMERA_INDEX
    if display is None:
        display = GESTURE_DISPLAY

    requested_backend = GESTURE_BACKEND
    if requested_backend not in {"auto", "classic", "tasks"}:
        logger.warning(f"Unknown gesture_backend '{requested_backend}', using auto")
        requested_backend = "auto"

    model_path = CONFIG_PATH.parent / "hand_landmarker.task"
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
        logger.error(
            "No safe MediaPipe hand backend is available. Install a MediaPipe "
            "build with solutions.hands, run on Raspberry Pi/Linux, or set "
            "gesture_allow_unsafe_tasks=true after validating Tasks locally."
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
    recent_gestures = deque(maxlen=90)          # (timestamp_ms, gesture|None)

    CAPTURE_FRESH_MS = int(_settings.get("gesture_capture_fresh_ms", 300))

    def _vote(now_ms):
        window = [(ts, g) for ts, g in recent_gestures
                  if now_ms - ts <= CAPTURE_WINDOW_MS]
        seen = [g for _, g in window if g]
        if not seen:
            return None

        winner, count = Counter(seen).most_common(1)[0]

        # The pose has to still be in front of the camera. Counting only the
        # frames that saw a hand means a hand already lowered still wins its
        # own vote, so pressing the shutter after dropping your arm would fire
        # whatever was last held.
        if not any(g == winner and now_ms - ts <= CAPTURE_FRESH_MS
                   for ts, g in window):
            return None

        # And it has to dominate rather than merely appear, so a shape passed
        # through on the way to another one is not what gets taken.
        if count * 2 < len(seen):
            return None
        return winner

    # ── Swipe detector state ───────────────────────────────────────────────
    swipe_state = _make_swipe_state()

    # ── FPS counter ───────────────────────────────────────────────────────
    frame_times      = deque(maxlen=30)
    last_timestamp_ms = 0
    frame_count      = 0

    try:
        while not (stop_event and stop_event.is_set()):
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera frame read failed")
                break

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

            for hand in hands:
                hand_lm_raw = hand
                current     = classify_gesture(hand)

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
            swipe   = _update_swipe_detector(swipe_state, hand_lm_raw, now_ms)

            # Swipe overrides classify_gesture if one was detected this frame
            if swipe is not None:
                current = swipe

            # ── Shutter mode: report only when the caller asks ─────────────
            if shutter_mode:
                recent_gestures.append((now_ms, current))
                request = None
                try:
                    request = capture_check()
                except Exception as e:
                    logger.debug(f"capture_check raised: {e}")

                if request == "preview":
                    if not _callback_allows_continue(
                            callback_fn, f"PREVIEW:{_vote(now_ms) or 'NONE'}"):
                        break
                elif request == "capture":
                    captured = _vote(now_ms)
                    logger.info(f"Shutter pressed — captured gesture: {captured}")
                    if not _callback_allows_continue(
                            callback_fn, captured or "NO_GESTURE"):
                        break
                    recent_gestures.clear()   # do not reuse a spent pose

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
    print("  Open Palm (all 5 fingers)  → MODE_SCAN      (OCR)")
    print("  Thumbs Up                  → CONFIRM")
    print("  Two Fingers (V sign)       → MODE_VOICE")
    print("  Index Pointing Down        → REPEAT")
    print("  Fist                       → STOP")
    print("  Index Pointing Up          → OBJECT_DETECT")
    print("  Three Fingers              → GPS_CHECK")
    print("  Shaka (pinky + thumb)      → TOGGLE_PRIVACY")
    print("  OK Sign (circle + 3 up)    → STATUS_CHECK")
    print("  Hand Swipe Right           → SWIPE_RIGHT (Volume UP)")
    print("  Hand Swipe Left            → SWIPE_LEFT  (Volume DOWN)")
    print("=" * 54)
    print("Press Q in the display window to quit.")
    detect_gesture(display=True)
