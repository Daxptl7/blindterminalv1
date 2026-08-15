"""
voice.py — BlindAssist Project (v6, single-mode)
================================================
One recording mode, nothing else.

    Enter the mode  ->  the microphone starts recording immediately.
    Button 3 pressed three times  ->  recording stops.
    The WAV is written to the USB pen drive.
    The speaker asks whether the clip is confidential.
    Button 1 -> play it in the earphone.  Button 2 (or 10s silence) -> speaker.
    Then it is transcribed.

WHY THIS VERSION EXISTS
-----------------------
v4 tried three capture backends (PyAudio+webrtcvad, arecord, SpeechRecognition)
and chose between them at runtime. On this Pi the PyAudio path cannot work at
all: the USB PnP capture card (card 4) refuses every rate PortAudio asks for,
so every attempt ended in

    Expression 'paInvalidSampleRate' failed in 'src/hostapi/alsa/pa_linux_alsa.c'

behind ~60 lines of ALSA/JACK config warnings. The arecord path on the same
card works perfectly. So PyAudio, webrtcvad and SpeechRecognition-microphone
capture are gone: one capture backend (arecord), one mode (open-ended, the user
ends it). That also retires the "which identical dongle is the real mic"
guessing — the device comes from settings.json and nothing overrides it.

WHAT v6 ADDS OVER v5
--------------------
1. Button 3 now works when voice.py is run on its own.
   v5 exposed register_press() and waited for something else to call it. Run
   standalone, nothing did, so only ENTER could stop a recording. There is now
   a watcher inside this module that reads the Pico W directly and counts
   presses using the same rules as main.py's _button3_stop_signal(): RAW:3
   messages inside the press window, and CONFIRM treated as proof of two
   presses (the firmware stops scanning the pins while it decides whether a
   gesture was a double-press, so one RAW:3 of the pair can go missing).

   It is only started when the caller passes no stop_check. main.py always
   passes one when a Pico is attached, so the two never open the port at the
   same time and main.py's single reader keeps its queue to itself. If you
   would rather share one connection, call set_morse_serial(serial) once at
   startup and this module will use that object instead of opening its own.

2. Playback asks before it plays.
   v5 played every recording straight into the earphone. The question is asked
   on the *speaker* — a user who cannot see the screen has to hear the prompt
   without wearing the earphone to find out it was asked. Button 1 routes
   playback to the earphone, Button 2 to the speaker, and 10 seconds of silence
   falls through to the speaker so an unattended device is never left waiting.

HOW A RECORDING IS STOPPED
--------------------------
Any of these ends it:
  * three presses of Button 3 within STOP_WINDOW_S
  * a stop_check() passed in by the caller returning True
  * ENTER, when a terminal is attached (for testing over SSH)
  * voice.stop() from another thread
  * MANUAL_MAX_S as a backstop, so a stuck button cannot record forever

There is no silence detection anywhere: pausing mid-sentence ("what is the
capital of … Australia?") must not end the recording, which is what the old VAD
path did.

WHERE THE AUDIO GOES
--------------------
Every capture is written to the pen drive as recordings/voice_YYYYmmdd_HHMMSS.wav.
_resolve_recording_dir() looks for a real removable mount first (/media, /mnt or
/run/media backed by /dev/sd*), then the configured path, then the SD card, then
/tmp — so an unplugged stick degrades instead of losing the recording.

The file is always rewritten from the decoded frames rather than left as
arecord finished it. Terminating arecord mid-write leaves a RIFF header
claiming zero frames, which is why aplay refused such files with
"audio open error: Unknown error 524".

WIRING INTO main.py
-------------------
No change is required — _listen_until_stopped() already passes the stop_check
built by _button3_stop_signal(), and this module honours it exactly as before.
Optionally, to reuse one serial connection:

    voice.set_morse_serial(_morse_serial_singleton)

Public API
----------
    listen(lang='en-IN', speak_fn=None, max_seconds=None,
           stop_check=None, manual_stop=True)            -> str | None
    record(stop_check=None, max_seconds=None,
           speak_fn=None)                                -> str | None  (WAV path)
    play(path=None, device=None)                         -> bool
    play_with_privacy_prompt(path=None, speak_fn=None,
                             timeout=10.0)               -> bool
    transcribe_file(path, lang='en-IN')                  -> str | None
    set_morse_serial(serial) / register_press() / reset_presses()
    get_last_error() / last_recording_path() / stop() / diagnostics()
"""

import json
import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("VoiceModule")

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "voice.log"
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_settings() -> dict:
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read {CONFIG_PATH} ({e}); using defaults.")
        return {}


_settings = _load_settings()

# ── CONFIGURATION ───────────────────────────────────────────
SAMPLE_RATE = 16000                    # every STT engine here wants 16 kHz mono

# Capture card. On this device the microphone is the USB PnP dongle on card 4;
# cards 2 and 3 are output-only and their capture inputs float, which is what
# produced the mains hum that older versions accepted as speech.
MIC_DEVICE = _settings.get("mic_device") or "plughw:4,0"

# Two outputs, deliberately distinct. The prompt is always asked on the
# speaker; only the answer decides where the recording is played back.
SPEAKER_DEVICE = _settings.get("speaker_device") or "plughw:3,0"
# `bone_device` is the old name for earphone_device — there is no bone
# conduction unit on this device — and is still read so an existing
# settings.json keeps working.
EARPHONE_DEVICE = (_settings.get("earphone_device")
                   or _settings.get("bone_device")
                   or "plughw:2,0")

MANUAL_MAX_S = float(_settings.get("voice_manual_max_seconds", 180.0))

# Button 3: this many presses inside this window stop the recording. Three
# rather than one so a stray knock cannot cut a question short.
STOP_PRESS_COUNT = int(_settings.get("voice_stop_press_count", 3))
STOP_WINDOW_S = float(_settings.get("voice_stop_press_window_seconds", 4.0))

# How long the confidentiality question waits before defaulting to the speaker.
_privacy_settings = _settings.get("privacy")
PRIVACY_PROMPT_TIMEOUT_S = float(
    _privacy_settings.get("confidential_prompt_timeout_seconds", 10.0)
    if isinstance(_privacy_settings, dict) else 10.0)

# The first fraction of a second after a USB capture stream opens is a DC step,
# not audio. Trimming it keeps a click out of the file and out of the level.
OPEN_TRANSIENT_MS = int(_settings.get("voice_open_transient_ms", 250))

RECORDING_DIR_SETTING = _settings.get("voice_recording_dir", "data/recordings")
KEEP_RECORDINGS = int(_settings.get("voice_keep_recordings", 20))  # <=0 = keep all
BEEP_ENABLED = bool(_settings.get("voice_beep", True))
UNMUTE_MIC = bool(_settings.get("voice_unmute_mic", True))

# -20 dBFS is roughly what the recognisers are tuned for; a USB electret on a
# Pi lands nearer -35, and every engine degrades sharply on input that quiet.
NORMALIZE_TARGET_DBFS = float(_settings.get("voice_normalize_target_dbfs", -20.0))
NORMALIZE_MAX_GAIN = float(_settings.get("voice_normalize_max_gain", 20.0))
SILENCE_DBFS = float(_settings.get("voice_silence_dbfs", -60.0))

STT_ENGINES: List[str] = list(_settings.get("stt_engines", ["vosk", "google", "sphinx"]))

_abort = threading.Event()
_last_error: Optional[str] = None
_last_recording: Optional[str] = None
_error_lock = threading.Lock()


def _set_error(message: Optional[str]):
    global _last_error
    with _error_lock:
        _last_error = message


def get_last_error() -> Optional[str]:
    """Why the last capture returned nothing, phrased so it can be read aloud."""
    with _error_lock:
        return _last_error


def last_recording_path() -> Optional[str]:
    """Path of the most recent saved WAV on the pen drive."""
    return _last_recording


def stop():
    """Abort an in-progress recording from another thread."""
    _abort.set()


# ── PICO W BUTTONS ──────────────────────────────────────────
# main.py owns a single MorseSerial reader; a second reader on the same port
# would steal messages out of its queue. So: if main.py hands us its object via
# set_morse_serial(), use that. Otherwise open our own connection, but only
# when the caller gave us no stop_check — which is exactly the standalone case,
# where main.py is not running and the port is free.

_shared_serial = None
_own_serial = None
_serial_lock = threading.Lock()


def set_morse_serial(serial):
    """Share main.py's MorseSerial singleton with this module (optional)."""
    global _shared_serial
    _shared_serial = serial
    logger.info("Using the caller's Morse serial connection.")


def _morse_serial_class():
    """Find the MorseSerial class without depending on its name.

    The module is known to expose get_message()/close(); hunting for the
    capability rather than a hard-coded class name means a rename in
    morse_serial.py cannot silently disable the stop button.
    """
    module = None
    for import_path in ("modules.morse_serial", "morse_serial"):
        try:
            module = __import__(import_path, fromlist=["*"])
            break
        except Exception:
            continue
    if module is None:
        return None
    for value in vars(module).values():
        if (isinstance(value, type)
                and hasattr(value, "get_message")
                and hasattr(value, "close")):
            return value
    return None


def _get_serial(open_if_needed: bool = True):
    """The Pico connection to listen on, or None when no Pico is attached."""
    global _own_serial
    if _shared_serial is not None:
        return _shared_serial
    if _own_serial is not None or not open_if_needed:
        return _own_serial

    with _serial_lock:
        if _own_serial is not None:
            return _own_serial
        cls = _morse_serial_class()
        if cls is None:
            logger.info("morse_serial not importable — buttons unavailable here.")
            return None
        try:
            _own_serial = cls()
            logger.info("Pico W buttons connected (own serial connection).")
        except Exception as e:
            # Busy port (main.py already holds it) or no Pico plugged in.
            logger.info(f"Pico W buttons unavailable ({e}); using ENTER/backstop.")
            _own_serial = None
    return _own_serial


def _close_own_serial():
    global _own_serial
    with _serial_lock:
        if _own_serial is not None:
            try:
                _own_serial.close()
            except Exception:
                pass
            _own_serial = None


# Software press counter, kept from v5 so a caller with its own button handler
# can still drive the stop condition by calling register_press().
_press_times: List[float] = []
_press_lock = threading.Lock()


def register_press():
    """Call once per Button-3 press from an external button handler."""
    now = time.time()
    with _press_lock:
        _press_times.append(now)
        del _press_times[:-STOP_PRESS_COUNT]


def reset_presses():
    with _press_lock:
        _press_times.clear()


def _presses_say_stop() -> bool:
    now = time.time()
    with _press_lock:
        recent = [t for t in _press_times if now - t <= STOP_WINDOW_S]
        _press_times[:] = recent
        return len(recent) >= STOP_PRESS_COUNT


def _button3_stop_watcher() -> Tuple[Optional[Callable[[], bool]], Callable[[], None]]:
    """Watch Button 3 in the background; return (stop_check, cancel).

    Deliberately the same rules as main.py's _button3_stop_signal(): only
    RAW:3 counts, and CONFIRM is taken as proof that two presses happened. The
    firmware stops scanning the pins while it decides whether a gesture was a
    double-press, so one RAW:3 of the pair can never arrive — without the
    CONFIRM clause a user tapping quickly could press five times and never
    reach three.
    """
    serial = _get_serial()
    if serial is None:
        return None, (lambda: None)

    stopped = threading.Event()
    finished = threading.Event()

    def _watch():
        presses = 0
        last_press = 0.0
        while not finished.is_set():
            try:
                msg = serial.get_message(timeout=0.3)
            except Exception as e:
                logger.debug(f"Button read error while recording: {e}")
                time.sleep(0.3)
                continue
            if msg is None:
                continue

            now = time.time()
            text = str(msg).strip()
            if text.startswith("RAW:3"):
                presses = presses + 1 if (now - last_press) <= STOP_WINDOW_S else 1
            elif text == "CONFIRM":
                presses = max(presses, 2)
            else:
                continue

            last_press = now
            logger.info(f"Stop button press {presses} of {STOP_PRESS_COUNT}")
            if presses >= STOP_PRESS_COUNT:
                stopped.set()
                return

    threading.Thread(target=_watch, daemon=True).start()
    return stopped.is_set, finished.set


def _wait_for_button(valid: Tuple[str, ...], timeout: float) -> Optional[str]:
    """Wait for one of `valid` button numbers. Returns '1'/'2'/... or None.

    Uses wait_for_raw_button() when morse_serial provides it, since that is the
    call main.py already uses for menu choices, and falls back to reading RAW:n
    messages off the queue directly.
    """
    serial = _get_serial()
    if serial is None:
        return None

    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = max(0.1, deadline - time.time())
        try:
            if hasattr(serial, "wait_for_raw_button"):
                btn = serial.wait_for_raw_button(timeout=remaining)
                if btn is None:
                    continue
                digits = "".join(ch for ch in str(btn) if ch.isdigit())
                if digits and digits[-1] in valid:
                    return digits[-1]
                continue

            msg = serial.get_message(timeout=remaining)
            if msg is None:
                continue
            text = str(msg).strip()
            if text.startswith("RAW:"):
                digit = text.split("RAW:", 1)[1].strip()[:1]
                if digit in valid:
                    return digit
        except Exception as e:
            logger.debug(f"Button read error: {e}")
            time.sleep(0.2)
    return None


# ── STT ENGINE PROBING (once, at import) ────────────────────
SR_AVAILABLE = False
try:
    import speech_recognition as sr

    SR_AVAILABLE = True
except Exception as e:
    logger.info(f"SpeechRecognition unavailable ({e}); Google/Sphinx STT disabled.")

VOSK_AVAILABLE = False
_vosk_model = None
_vosk_model_dir: Optional[Path] = None


def _looks_like_vosk_model(path: Path) -> bool:
    return path.is_dir() and ((path / "am").is_dir() or (path / "conf").is_dir())


def _find_vosk_model() -> Optional[Path]:
    """Prefer the on-board copy: speech must work with the USB stick unplugged."""
    roots: List[Path] = []
    configured = _settings.get("vosk_model_path")
    if configured:
        p = Path(configured)
        roots.append(p if p.is_absolute() else BASE_DIR / p)
    roots.append(BASE_DIR / "models_local" / "vosk")
    roots.append(BASE_DIR / "models" / "vosk")

    for root in roots:
        try:
            if _looks_like_vosk_model(root):
                return root
            if root.is_dir():
                for child in sorted(root.iterdir()):
                    if _looks_like_vosk_model(child):
                        return child
        except OSError:
            continue
    return None


try:
    from vosk import KaldiRecognizer, Model as VoskModel, SetLogLevel as _vosk_set_log

    try:
        _vosk_set_log(-1)                 # keep Kaldi's chatter off the console
    except Exception:
        pass

    _vosk_model_dir = _find_vosk_model()
    if _vosk_model_dir is not None:
        _vosk_model = VoskModel(str(_vosk_model_dir))
        VOSK_AVAILABLE = True
    else:
        logger.info(
            "Vosk installed but no model found. Download vosk-model-small-en-in-0.4 "
            "from https://alphacephei.com/vosk/models and extract it to "
            f"{BASE_DIR / 'models_local' / 'vosk'}/")
except ImportError:
    logger.info("Vosk not installed (pip install vosk) — offline STT unavailable.")
except Exception as e:
    logger.warning(f"Vosk model failed to load: {e}")

SPHINX_AVAILABLE = False
if SR_AVAILABLE:
    try:
        import pocketsphinx  # noqa: F401

        SPHINX_AVAILABLE = True
    except Exception:
        logger.info("PocketSphinx not installed — last-resort offline STT unavailable.")

_ENGINE_AVAILABLE = {
    "vosk": lambda: VOSK_AVAILABLE,
    "google": lambda: SR_AVAILABLE,
    "sphinx": lambda: SR_AVAILABLE and SPHINX_AVAILABLE,
}


def _active_engines() -> List[str]:
    return [e for e in STT_ENGINES if _ENGINE_AVAILABLE.get(e, lambda: False)()]


_ACTIVE = _active_engines()
if not _ACTIVE:
    logger.error("NO SPEECH RECOGNITION ENGINE AVAILABLE — recordings will still be "
                 "saved to the pen drive, but nothing will be transcribed.")
elif not (VOSK_AVAILABLE or SPHINX_AVAILABLE):
    logger.warning(f"STT engines active: {', '.join(_ACTIVE)} — all need internet.")
else:
    logger.info(f"STT engines active: {' -> '.join(_ACTIVE)}")


# ── WHERE RECORDINGS ARE SAVED (USB PEN DRIVE FIRST) ────────
def _removable_mounts() -> List[Path]:
    """Writable mounts that look like a plugged-in USB stick.

    Reads /proc/mounts rather than guessing at /media/pi/<label>: the label
    changes with the stick, and udisks does not always mount under /media.
    """
    found: List[Path] = []
    try:
        with open("/proc/mounts", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                device, raw_point = parts[0], parts[1]
                if not device.startswith("/dev/sd"):
                    continue              # the SD card is /dev/mmcblk*, skip it
                point = Path(raw_point.replace("\\040", " "))
                if str(point).startswith(("/media", "/mnt", "/run/media")):
                    found.append(point)
    except Exception as e:
        logger.debug(f"Could not read /proc/mounts: {e}")
    return found


def _resolve_recording_dir(configured: Optional[str]) -> Optional[str]:
    """First writable candidate wins; the pen drive is tried before the SD card."""
    candidates: List[Path] = [m / "recordings" for m in _removable_mounts()]

    if configured:
        p = Path(configured)
        candidates.append(p if p.is_absolute() else BASE_DIR / p)
    candidates.append(BASE_DIR / "data" / "recordings")
    candidates.append(Path(tempfile.gettempdir()) / "blindassist-recordings")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_bytes(b"")
            probe.unlink()
            return str(candidate)
        except Exception:
            continue

    logger.error("No writable recordings directory — captures cannot be saved.")
    return None


RECORDING_DIR = _resolve_recording_dir(RECORDING_DIR_SETTING)
if RECORDING_DIR:
    logger.info(f"Recordings will be saved to {RECORDING_DIR}")


def _prune_recordings(keep: int = KEEP_RECORDINGS):
    """Bound storage. keep <= 0 means keep everything."""
    if not RECORDING_DIR or keep <= 0:
        return
    try:
        files = sorted(Path(RECORDING_DIR).glob("voice_*.wav"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in files[keep:]:
            try:
                stale.unlink()
            except OSError:
                pass
    except Exception as e:
        logger.debug(f"Recording cleanup skipped: {e}")


# ── AUDIO HELPERS ───────────────────────────────────────────
def _to_array(pcm: bytes) -> np.ndarray:
    usable = (len(pcm) // 2) * 2
    return np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32)


def _to_pcm(samples: np.ndarray) -> bytes:
    return np.clip(samples, -32768, 32767).astype("<i2").tobytes()


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _dbfs(samples: np.ndarray) -> float:
    rms = _rms(samples)
    return -120.0 if rms <= 0 else 20.0 * math.log10(rms / 32768.0)


def _trim_transient(samples: np.ndarray, rate: int) -> np.ndarray:
    skip = int(rate * OPEN_TRANSIENT_MS / 1000)
    if skip <= 0 or samples.size <= skip * 2:
        return samples
    return samples[skip:]


def _resample_to_16k(samples: np.ndarray, rate: int) -> np.ndarray:
    if rate == SAMPLE_RATE or samples.size == 0:
        return samples
    if rate % SAMPLE_RATE == 0:                  # box-filtered decimation
        factor = rate // SAMPLE_RATE
        trimmed = samples[: (samples.size // factor) * factor]
        return trimmed.reshape(-1, factor).mean(axis=1)
    out_len = int(samples.size * SAMPLE_RATE / rate)
    if out_len <= 1:
        return np.zeros(0, dtype=np.float32)
    src_idx = np.linspace(0, samples.size - 1, out_len)
    return np.interp(src_idx, np.arange(samples.size), samples).astype(np.float32)


def _normalize(samples: np.ndarray) -> np.ndarray:
    """Remove DC offset and lift the level, peak-limited rather than clipped."""
    if samples.size == 0:
        return samples
    centred = samples - float(np.mean(samples))
    rms = _rms(centred)
    if rms < 1.0:                                # essentially digital silence
        return centred

    target_rms = (10.0 ** (NORMALIZE_TARGET_DBFS / 20.0)) * 32768.0
    gain = min(target_rms / rms, NORMALIZE_MAX_GAIN)
    if gain <= 1.0:
        return centred                           # already loud enough
    boosted = centred * gain
    peak = float(np.max(np.abs(boosted)))
    if peak > 32000.0:
        boosted *= 32000.0 / peak
    return boosted


def _bandpass(samples: np.ndarray, rate: int = SAMPLE_RATE,
              low_freq: int = 300, high_freq: int = 3400) -> np.ndarray:
    """Single-pole IIR bandpass over the telephony speech band.

    Skipped when scipy is missing: a pure-Python version costs seconds of a
    blind user's time, and unfiltered-but-normalized audio transcribes better
    than filtered audio that arrives four seconds late.
    """
    if samples.size < 2:
        return samples
    try:
        from scipy.signal import lfilter
    except ImportError:
        return samples

    dt = 1.0 / rate
    rc_high = 1.0 / (2.0 * math.pi * low_freq)
    rc_low = 1.0 / (2.0 * math.pi * high_freq)
    alpha_hp = rc_high / (rc_high + dt)
    alpha_lp = dt / (rc_low + dt)
    hp = lfilter([alpha_hp, -alpha_hp], [1.0, -alpha_hp], samples)
    lp = lfilter([alpha_lp], [1.0, -(1.0 - alpha_lp)], hp)
    return lp.astype(np.float32)


def _write_wav(path: str, pcm: bytes, rate: int = SAMPLE_RATE) -> bool:
    try:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(pcm)
        return True
    except Exception as e:
        logger.error(f"Could not write {path}: {e}")
        return False


def _mic_card_number() -> Optional[str]:
    """'plughw:4,0' -> '4', for amixer."""
    if ":" not in MIC_DEVICE:
        return None
    return MIC_DEVICE.split(":", 1)[1].split(",")[0].strip() or None


def _unmute_microphone():
    """Raise and unmute the capture control. Best-effort, every call optional.

    A muted capture mixer records digital silence on a perfectly good card, and
    the resulting "I didn't catch that" sends the user hunting for a loose plug.
    """
    if not UNMUTE_MIC:
        return
    card = _mic_card_number()
    if not card or not shutil.which("amixer"):
        return
    for args in (["sset", "Mic", "100%"], ["sset", "Mic", "cap"],
                 ["sset", "Capture", "100%"], ["sset", "Capture", "cap"]):
        try:
            subprocess.run(["amixer", "-c", card] + args, timeout=3,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


_unmute_microphone()


def _drain_stdin():
    """Discard anything already typed but not yet read.

    Keystrokes buffer in the terminal indefinitely while nothing is reading
    them. A stray ENTER — pressed at an earlier prompt that had already timed
    out, or out of impatience while the AI was thinking — therefore survives in
    the buffer until the *next* reader consumes it, which is the recording
    loop's stop-on-ENTER check. The recording then ends within a millisecond of
    starting and reports "no audio captured", looking exactly like a dead
    microphone.

    Input typed before recording began cannot have been meant to stop a
    recording that did not exist yet, so it is dropped here.
    """
    if sys.stdin is None or not sys.stdin.isatty():
        return
    try:
        import termios

        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        return
    except Exception as e:
        logger.debug(f"tcflush unavailable ({e}); draining with select.")

    try:
        import select

        while select.select([sys.stdin], [], [], 0)[0]:
            if not os.read(sys.stdin.fileno(), 4096):
                break
    except Exception as e:
        logger.debug(f"Could not drain stdin: {e}")


def _beep(device: Optional[str] = None, freq: float = 880.0, duration: float = 0.12):
    """Earcon so a blind user knows the microphone went live.

    Played before arecord opens the capture stream, so it is not recorded back.
    """
    if not BEEP_ENABLED:
        return
    aplay = shutil.which("aplay")
    if not aplay:
        return
    try:
        rate = 16000
        t = np.arange(int(rate * duration), dtype=np.float32) / rate
        tone = np.sin(2 * math.pi * freq * t) * 8000.0
        fade = min(200, tone.size // 4)
        if fade:
            tone[:fade] *= np.linspace(0, 1, fade)
            tone[-fade:] *= np.linspace(1, 0, fade)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        _write_wav(path, _to_pcm(tone), rate)
        subprocess.run([aplay, "-q", "-D", device or SPEAKER_DEVICE, path], timeout=4,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.unlink(path)
    except Exception as e:
        logger.debug(f"Earcon skipped: {e}")


def _announce(text: str, device: str, speak_fn: Optional[Callable] = None) -> bool:
    """Say something on a *specific* card.

    The project's tts module speaks wherever it is configured to; the privacy
    question has to come out of the speaker specifically, so espeak-to-WAV then
    aplay -D is tried first because it is the only path that controls the card.
    speak_fn is the fallback, and a double beep is the last resort so the
    prompt is never silent.
    """
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    aplay = shutil.which("aplay")
    if espeak and aplay:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                path = tmp.name
            subprocess.run([espeak, "-s", "130", "-w", path, text], timeout=15,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.getsize(path) > 44:
                subprocess.run([aplay, "-q", "-D", device, path], timeout=30,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.unlink(path)
                return True
            os.unlink(path)
        except Exception as e:
            logger.debug(f"espeak announcement failed: {e}")

    if speak_fn:
        try:
            speak_fn(text)
            return True
        except Exception as e:
            logger.debug(f"speak_fn failed: {e}")

    _beep(device, freq=660.0)
    _beep(device, freq=660.0)
    return False


# ── THE ONE CAPTURE PATH: arecord ───────────────────────────
def record(stop_check: Optional[Callable[[], bool]] = None,
           max_seconds: Optional[float] = None,
           speak_fn: Optional[Callable] = None) -> Optional[str]:
    """Record until the user stops it. Returns the saved WAV path, or None.

    Recording starts as soon as this is called — there is no "press to start".
    When the caller passes no stop_check, a Button-3 watcher is started here so
    the buttons work even when this module is run on its own.
    """
    global _last_recording

    _abort.clear()
    _set_error(None)
    reset_presses()

    if not shutil.which("arecord"):
        _set_error("The recording program is missing on this device.")
        logger.error("arecord is not installed (apt install alsa-utils).")
        return None

    own_watcher = None
    cancel_watcher: Callable[[], None] = lambda: None
    if stop_check is None:
        own_watcher, cancel_watcher = _button3_stop_watcher()

    backstop = float(max_seconds or MANUAL_MAX_S)
    duration = max(1, int(round(backstop)))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_path = tmp.name

    cmd = ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-r", str(SAMPLE_RATE),
           "-c", "1", "-d", str(duration), "-t", "wav", temp_path]

    buttons_live = stop_check is not None or own_watcher is not None
    if speak_fn:
        if buttons_live:
            speak_fn(f"Recording. Press button 3 {STOP_PRESS_COUNT} times "
                     "when you are finished.")
        else:
            speak_fn("Recording. Press Enter when you are finished.")
    _beep()

    logger.info(f"Recording from {MIC_DEVICE} (backstop {duration}s, "
                f"buttons {'live' if buttons_live else 'unavailable'})")
    tty = sys.stdin is not None and sys.stdin.isatty()
    if tty:
        how = ("three presses of Button 3, or ENTER here"
               if buttons_live else "ENTER here")
        print(f"\n🔴 Recording from {MIC_DEVICE} — {how} — to stop…")
        # Only ENTER pressed from here on may stop this recording.
        _drain_stdin()

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, text=True)

        deadline = time.time() + duration + 2.0
        stopped_by = "backstop"

        while process.poll() is None and time.time() < deadline:
            if _abort.is_set():
                stopped_by = "abort"
                process.terminate()
                break

            if own_watcher is not None and own_watcher():
                stopped_by = "button 3"
                process.terminate()
                break

            if _presses_say_stop():
                stopped_by = "button 3 (external handler)"
                process.terminate()
                break

            if stop_check is not None:
                try:
                    if stop_check():
                        stopped_by = "button 3 (caller)"
                        process.terminate()
                        break
                except Exception as e:
                    logger.debug(f"stop_check raised: {e}")

            if tty:
                import select

                if select.select([sys.stdin], [], [], 0.15)[0]:
                    sys.stdin.readline()
                    stopped_by = "ENTER"
                    process.terminate()
                    break
            else:
                time.sleep(0.1)

        try:
            _, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr = process.communicate(timeout=5)

        logger.info(f"Recording ended by {stopped_by}.")

        size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
        if size <= 44:
            detail = (stderr or "").strip().splitlines()
            logger.error(f"arecord failed on {MIC_DEVICE}: "
                         f"{detail[-1] if detail else 'no audio captured'}")
            _set_error("I could not use the microphone. "
                       "Please check that it is plugged in.")
            return None

        # arecord was killed mid-write, so its RIFF header can still claim zero
        # frames though megabytes of samples were written. Read the payload and
        # write a correct header ourselves — a file saved with the truncated
        # header is why aplay reported "Unknown error 524".
        with wave.open(temp_path, "rb") as wf:
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        if not frames:
            with open(temp_path, "rb") as raw:
                frames = raw.read()[44:]
            rate = SAMPLE_RATE
            logger.debug("WAV header was truncated; read the payload directly.")

        if not frames:
            _set_error("Nothing was recorded. Please try again.")
            return None

        samples = _trim_transient(_to_array(frames), rate)
        if rate != SAMPLE_RATE:
            samples = _resample_to_16k(samples, rate)

        level = _dbfs(samples)
        seconds = samples.size / SAMPLE_RATE
        logger.info(f"Captured {seconds:.1f}s at {level:.0f} dBFS")

        if level <= SILENCE_DBFS:
            _set_error("The microphone is not picking up any sound. "
                       "Please check that it is switched on and turned up.")
            return None

        if not RECORDING_DIR:
            _set_error("There is nowhere to save the recording. "
                       "Please check the USB drive.")
            return None

        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(RECORDING_DIR, f"voice_{stamp}.wav")
        if not _write_wav(out_path, _to_pcm(samples)):
            _set_error("The recording could not be saved to the USB drive.")
            return None

        _prune_recordings()
        _last_recording = out_path
        logger.info(f"Saved {out_path} ({seconds:.1f}s)")
        if tty:
            print(f"💾 Saved: {out_path}  ({seconds:.1f}s, {level:.0f} dBFS, "
                  f"stopped by {stopped_by})")
        return out_path

    except Exception as e:
        logger.error(f"Recording failed: {e}")
        _set_error("Something went wrong with the microphone. Please try again.")
        return None
    finally:
        cancel_watcher()
        try:
            os.unlink(temp_path)
        except OSError:
            pass


# ── PLAYBACK ────────────────────────────────────────────────
def play(path: Optional[str] = None, device: Optional[str] = None) -> bool:
    """Play a recording on a given card. Defaults to the latest, on the speaker."""
    target = path or _last_recording
    if not target or not os.path.exists(target):
        logger.warning("Nothing to play back.")
        return False
    aplay = shutil.which("aplay")
    if not aplay:
        return False
    try:
        subprocess.run([aplay, "-q", "-D", device or SPEAKER_DEVICE, target],
                       timeout=600, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        logger.warning(f"Playback failed: {e}")
        return False


def play_with_privacy_prompt(path: Optional[str] = None,
                             speak_fn: Optional[Callable] = None,
                             timeout: Optional[float] = None) -> bool:
    """Ask on the speaker where to play the clip, then play it there.

    Button 1 -> earphone (confidential). Button 2 -> speaker. No answer within
    `timeout` -> speaker, because a device left waiting for an answer that is
    not coming is worse than a default, and the user can always stop playback.

    The question is deliberately asked on the speaker even though the answer
    may route audio to the earphone: someone not already wearing the earphone
    would otherwise never hear that they had been asked.
    """
    target = path or _last_recording
    if not target or not os.path.exists(target):
        logger.warning("Nothing to play back.")
        return False

    wait_s = float(timeout if timeout is not None else PRIVACY_PROMPT_TIMEOUT_S)
    question = ("Is this confidential? Press button 1 to listen in the earphone, "
                "or button 2 to play it on the speaker.")
    _announce(question, SPEAKER_DEVICE, speak_fn)

    tty = sys.stdin is not None and sys.stdin.isatty()
    if tty:
        print(f"\n🔒 {question}")
        print(f"   (or type 1 / 2 here — {wait_s:.0f}s, then the speaker)")
        # A key pressed before the question was asked is not an answer to it,
        # and a stale "2" would put confidential audio on the open speaker.
        _drain_stdin()

    choice: Optional[str] = None
    deadline = time.time() + wait_s
    serial_live = _get_serial() is not None

    while time.time() < deadline and choice is None:
        if serial_live:
            choice = _wait_for_button(("1", "2"), timeout=min(0.5, wait_s))
            if choice:
                break
        if tty:
            import select

            if select.select([sys.stdin], [], [], 0.2)[0]:
                typed = sys.stdin.readline().strip()
                if typed in ("1", "2"):
                    choice = typed
                    break
        elif not serial_live:
            time.sleep(0.2)

    if choice == "1":
        device, where = EARPHONE_DEVICE, "earphone"
    else:
        device, where = SPEAKER_DEVICE, "speaker"
        if choice is None:
            logger.info(f"No answer in {wait_s:.0f}s — defaulting to the speaker.")

    logger.info(f"Playing {os.path.basename(target)} on the {where} ({device}).")
    if tty:
        print(f"▶  Playing on the {where} ({device})…")
    return play(target, device)


# ── TRANSCRIPTION ───────────────────────────────────────────
def _transcribe_vosk(pcm: bytes) -> Optional[str]:
    """Offline, free, no API key. ~85-92% on clean short commands."""
    if not VOSK_AVAILABLE or _vosk_model is None:
        return None
    try:
        recognizer = KaldiRecognizer(_vosk_model, SAMPLE_RATE)
        recognizer.SetWords(True)
        for i in range(0, len(pcm), 4000):
            recognizer.AcceptWaveform(pcm[i:i + 4000])
        text = json.loads(recognizer.FinalResult()).get("text", "").strip()
        if text:
            logger.info(f"Vosk transcribed: '{text}'")
            return text
        return None
    except Exception as e:
        logger.error(f"Vosk transcription error: {e}")
        return None


def _transcribe_google(pcm: bytes, lang: str) -> Optional[str]:
    """Online, free tier. ~92-97% on clean speech."""
    if not SR_AVAILABLE:
        return None
    try:
        recognizer = sr.Recognizer()
        text = recognizer.recognize_google(sr.AudioData(pcm, SAMPLE_RATE, 2),
                                           language=lang).strip()
        if text:
            logger.info(f"Google transcribed: '{text}'")
            return text
        return None
    except sr.UnknownValueError:
        logger.info("Google could not understand the audio.")
        return None
    except sr.RequestError as e:
        # A network/quota problem, not a diction problem — the user should be
        # told that rather than asked to repeat themselves.
        _set_error("I could not reach the speech service. "
                   "Please check the internet connection.")
        logger.error(f"Google STT unreachable: {e}")
        return None
    except Exception as e:
        logger.error(f"Google transcription error: {e}")
        return None


def _transcribe_sphinx(pcm: bytes) -> Optional[str]:
    """Offline, no model download, ~70-80%. Last resort."""
    if not (SR_AVAILABLE and SPHINX_AVAILABLE):
        return None
    try:
        recognizer = sr.Recognizer()
        text = recognizer.recognize_sphinx(sr.AudioData(pcm, SAMPLE_RATE, 2)).strip()
        if text:
            logger.info(f"Sphinx transcribed: '{text}'")
            return text
        return None
    except sr.UnknownValueError:
        logger.info("Sphinx could not understand the audio.")
        return None
    except Exception as e:
        logger.error(f"Sphinx transcription error: {e}")
        return None


def _run_engines(pcm: bytes, lang: str) -> Optional[str]:
    for name in STT_ENGINES:
        if not _ENGINE_AVAILABLE.get(name, lambda: False)():
            continue                       # skip, rather than pretend to try
        if name == "vosk":
            result = _transcribe_vosk(pcm)
        elif name == "google":
            result = _transcribe_google(pcm, lang)
        elif name == "sphinx":
            result = _transcribe_sphinx(pcm)
        else:
            logger.warning(f"Unknown STT engine configured: {name}")
            continue
        if result:
            return result
        logger.debug(f"{name} returned nothing; trying next engine.")
    return None


def _transcribe(pcm: bytes, lang: str) -> Optional[str]:
    """Normalize, then try the engine chain on the best-sounding variant.

    Two variants: normalized+bandpassed (wins in a noisy room) and normalized
    only (wins when the filter has eaten a quiet or unusual voice).
    """
    raw = _to_array(pcm)
    if raw.size == 0:
        return None

    normalized = _normalize(raw)
    logger.info(f"Level {_dbfs(raw):.0f} dBFS -> {_dbfs(normalized):.0f} dBFS after gain")

    for label, samples in (("filtered", _bandpass(normalized)),
                           ("unfiltered", normalized)):
        if samples.size == 0 or _dbfs(samples) <= SILENCE_DBFS:
            continue
        result = _run_engines(_to_pcm(samples), lang)
        if result:
            if label != "filtered":
                logger.info("Bandpass hurt this clip; unfiltered audio won.")
            return result
    return None


def transcribe_file(path: str, lang: str = "en-IN") -> Optional[str]:
    """Transcribe an existing mono 16-bit WAV."""
    try:
        with wave.open(path, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                logger.warning(f"{path} is not mono 16-bit; cannot transcribe.")
                return None
            frames = wf.readframes(wf.getnframes())
            rate = wf.getframerate()
    except Exception as e:
        logger.error(f"Could not read {path}: {e}")
        return None

    samples = _to_array(frames)
    if rate != SAMPLE_RATE:
        samples = _resample_to_16k(samples, rate)
    return _transcribe(_to_pcm(samples), lang)


# ── PUBLIC ENTRY POINT ──────────────────────────────────────
def listen(lang: str = "en-IN", speak_fn: Optional[Callable] = None,
           max_seconds: Optional[float] = None,
           stop_check: Optional[Callable[[], bool]] = None,
           manual_stop: bool = True) -> Optional[str]:
    """Record until the user stops it, save to the pen drive, transcribe.

    The signature matches v4 so main.py keeps working unchanged, but there is
    only one behaviour now: max_seconds is a backstop and manual_stop is
    ignored. Returns the recognised text, or None — and when None,
    get_last_error() explains why in a sentence fit to be read aloud.

    Playback is deliberately NOT part of this call: a question on its way to
    the AI does not need playing back, and the privacy prompt would sit between
    the user and their answer. Call play_with_privacy_prompt() when review is
    what is wanted.
    """
    path = record(stop_check=stop_check, max_seconds=max_seconds, speak_fn=speak_fn)
    if not path:
        return None

    if not _active_engines():
        _set_error("Speech recognition is not installed, "
                   "but your recording has been saved.")
        return None

    text = transcribe_file(path, lang)
    if text:
        logger.info(f"Final transcription: '{text}'")
        _set_error(None)
        return text

    # The audio is on the pen drive either way, so this is recoverable — the
    # clip can be re-run later with transcribe_file().
    if get_last_error() is None:
        _set_error("I didn't catch that. Please try again.")
    return None


def listen_with_vad(lang: str = "en-IN",
                    speak_fn: Optional[Callable] = None) -> Optional[str]:
    """Kept so older callers do not break; there is one mode now."""
    return listen(lang, speak_fn)


def diagnostics() -> dict:
    """Machine-readable state for the startup self-test."""
    mounts = _removable_mounts()
    return {
        "mic_device": MIC_DEVICE,
        "speaker_device": SPEAKER_DEVICE,
        "earphone_device": EARPHONE_DEVICE,
        "arecord": bool(shutil.which("arecord")),
        "aplay": bool(shutil.which("aplay")),
        "espeak": bool(shutil.which("espeak-ng") or shutil.which("espeak")),
        "recording_dir": RECORDING_DIR,
        "recording_dir_is_usb": bool(RECORDING_DIR) and any(
            str(RECORDING_DIR).startswith(str(m)) for m in mounts),
        "usb_mounts": [str(m) for m in mounts],
        "pico_buttons": (_get_serial(open_if_needed=False) is not None
                         or _morse_serial_class() is not None),
        "vosk": VOSK_AVAILABLE,
        "vosk_model": str(_vosk_model_dir) if _vosk_model_dir else None,
        "speech_recognition": SR_AVAILABLE,
        "sphinx": SPHINX_AVAILABLE,
        "engines_configured": list(STT_ENGINES),
        "engines_active": _active_engines(),
        "stop_press_count": STOP_PRESS_COUNT,
        "stop_window_seconds": STOP_WINDOW_S,
        "privacy_prompt_timeout_s": PRIVACY_PROMPT_TIMEOUT_S,
        "manual_max_seconds": MANUAL_MAX_S,
    }


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: (stop(), sys.exit(0)))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("Voice Module v6 — single mode")
    info = diagnostics()
    for key, value in info.items():
        print(f"  {key:24} {value}")

    if not info["recording_dir_is_usb"]:
        print("\n⚠  The USB pen drive is not mounted — recordings will go to "
              f"{RECORDING_DIR} instead.")
    if not info["pico_buttons"]:
        print("⚠  No Pico W found — use ENTER to stop the recording.")

    started = time.time()
    try:
        wav = record(speak_fn=print)
        if wav:
            play_with_privacy_prompt(wav, speak_fn=print)
            print("📝 Transcribing…")
            print(f"Result: {transcribe_file(wav)}   "
                  f"({time.time() - started:.1f}s total)")
        print(f"Reason: {get_last_error()}")
    finally:
        _close_own_serial()
