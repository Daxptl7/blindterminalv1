"""
voice.py — BlindAssist Project (v5, single-mode)
================================================
One recording mode, nothing else.

    Enter the mode  ->  the microphone starts recording immediately.
    Button 3 pressed three times  ->  recording stops.
    The WAV is written to the USB pen drive, then transcribed.

WHY THIS VERSION EXISTS
-----------------------
v4 tried three capture backends (PyAudio+webrtcvad, arecord, SpeechRecognition)
and picked between them at runtime. On this Pi the PyAudio path cannot work at
all: the USB PnP capture card (card 4) refuses every rate PortAudio asks for,
so every attempt ended in

    Expression 'paInvalidSampleRate' failed in 'src/hostapi/alsa/pa_linux_alsa.c'

preceded by ~60 lines of ALSA/JACK config warnings. The arecord path on the
same card works perfectly:

    arecord -D plughw:4,0 -f S16_LE -r 16000 -c 1 -d 8 -t wav test.wav

So PyAudio, webrtcvad and SpeechRecognition-microphone capture are gone. There
is one capture backend (arecord) and one mode (open-ended, user stops it).
That also removes the entire "which of the identical dongles is the real mic"
guessing game — the device comes from settings.json and nothing overrides it.

HOW THE RECORDING IS STOPPED
----------------------------
Any of these ends it:
  * three presses of Button 3 within STOP_WINDOW_S  (main.py calls
    voice.register_press() once per press — see "WIRING INTO main.py" below)
  * a stop_check() callable passed in by the caller returning True
  * ENTER, when a terminal is attached (for testing over SSH)
  * voice.stop() from another thread
  * MANUAL_MAX_S as a backstop, so a stuck button cannot record forever

WHERE THE AUDIO GOES
--------------------
Every capture is written to the USB pen drive as
recordings/voice_YYYYmmdd_HHMMSS.wav. _resolve_recording_dir() looks for a real
removable mount first (/media/..., /mnt/... backed by /dev/sd*), then the
configured path, then the SD card, then /tmp — so an unplugged stick degrades
instead of losing the recording.

The file is always written from the decoded frames rather than by letting
arecord finish its own header. Terminating arecord mid-write leaves a RIFF
header claiming zero frames, which is exactly why `aplay` on such a file
failed; rewriting the header ourselves makes every saved file playable.

WIRING INTO main.py
-------------------
    import modules.voice as voice

    # once per Button-3 press, from your existing button handler:
    voice.register_press()

    # the mode itself:
    text = voice.listen(speak_fn=speak)      # blocks until the user stops it
    if text is None:
        speak(voice.get_last_error())

listen() already uses the press counter as its default stop condition, so no
stop_check argument is needed. Passing one still works and is ORed with it.

Public API
----------
    listen(lang='en-IN', speak_fn=None, max_seconds=None,
           stop_check=None, manual_stop=True)          -> str | None
    record(stop_check=None, max_seconds=None,
           speak_fn=None)                              -> str | None   (WAV path)
    play(path=None)                                    -> bool
    transcribe_file(path, lang='en-IN')                -> str | None
    register_press() / reset_presses()                 -> None
    get_last_error()                                   -> str | None
    last_recording_path()                              -> str | None
    stop()                                             -> None
    diagnostics()                                      -> dict
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
from typing import Callable, List, Optional

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
SAMPLE_RATE = 16000                       # every STT engine here wants 16 kHz mono

# Capture card. On this device the microphone is the USB PnP dongle on card 4;
# cards 2 and 3 are output-only (earphone and speaker) and their capture inputs
# float. `python3 mic_check.py` re-detects this if the hardware moves.
MIC_DEVICE = _settings.get("mic_device") or "plughw:4,0"

# Playback card for the earcon and for play(). The earphone dongle is card 2.
PLAYBACK_DEVICE = (_settings.get("playback_device")
                   or _settings.get("bone_device")
                   or "plughw:2,0")

# Backstop only — the user, not the clock, ends a recording.
MANUAL_MAX_S = float(_settings.get("voice_manual_max_seconds", 180.0))

# Button 3: three presses inside this window stop the recording.
STOP_PRESS_COUNT = int(_settings.get("voice_stop_press_count", 3))
STOP_WINDOW_S = float(_settings.get("voice_stop_press_window_seconds", 4.0))

# The first fraction of a second after a USB capture stream opens is a DC step,
# not audio. Trimming it keeps a click out of the saved file and out of the
# level measurement.
OPEN_TRANSIENT_MS = int(_settings.get("voice_open_transient_ms", 250))

RECORDING_DIR_SETTING = _settings.get("voice_recording_dir", "data/recordings")
KEEP_RECORDINGS = int(_settings.get("voice_keep_recordings", 20))   # <=0 = keep all
BEEP_ENABLED = bool(_settings.get("voice_beep", True))
UNMUTE_MIC = bool(_settings.get("voice_unmute_mic", True))

# -20 dBFS is roughly what the recognisers are tuned for; a USB electret on a
# Pi lands nearer -35, and every engine degrades sharply on input that quiet.
NORMALIZE_TARGET_DBFS = float(_settings.get("voice_normalize_target_dbfs", -20.0))
NORMALIZE_MAX_GAIN = float(_settings.get("voice_normalize_max_gain", 20.0))

# Anything at or below this is not worth sending to a recogniser.
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


# ── BUTTON 3: THREE PRESSES STOP THE RECORDING ──────────────
_press_times: List[float] = []
_press_lock = threading.Lock()


def register_press():
    """Call once per Button-3 press, from main.py's button handler.

    Presses older than STOP_WINDOW_S are discarded, so three *deliberate*
    presses stop the recording while three presses spread over a minute do not.
    """
    now = time.time()
    with _press_lock:
        _press_times.append(now)
        del _press_times[:-STOP_PRESS_COUNT]
        logger.debug(f"Button 3 press {len(_press_times)}/{STOP_PRESS_COUNT}")


def reset_presses():
    """Clear the press history (called automatically when a recording starts)."""
    with _press_lock:
        _press_times.clear()


def _presses_say_stop() -> bool:
    now = time.time()
    with _press_lock:
        recent = [t for t in _press_times if now - t <= STOP_WINDOW_S]
        _press_times[:] = recent
        return len(recent) >= STOP_PRESS_COUNT


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
        _vosk_set_log(-1)                    # keep Kaldi's chatter off the console
    except Exception:
        pass

    _vosk_model_dir = _find_vosk_model()
    if _vosk_model_dir is not None:
        _vosk_model = VoskModel(str(_vosk_model_dir))
        VOSK_AVAILABLE = True
    else:
        logger.info(
            "Vosk installed but no model found. Download vosk-model-small-en-in-0.4 "
            f"from https://alphacephei.com/vosk/models and extract it to "
            f"{BASE_DIR / 'models_local' / 'vosk'}/"
        )
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
    """Writable mounts that look like a plugged-in USB stick, newest first.

    Reads /proc/mounts rather than guessing at /media/pi/<label>, because the
    label changes with the stick and udisks does not always mount under /media.
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
                    continue                      # SD card is /dev/mmcblk*, skip it
                point = Path(raw_point.replace("\\040", " "))
                if str(point).startswith(("/media", "/mnt", "/run/media")):
                    found.append(point)
    except Exception as e:
        logger.debug(f"Could not read /proc/mounts: {e}")
    return found


def _resolve_recording_dir(configured: Optional[str]) -> Optional[str]:
    """First writable candidate wins; the pen drive is tried before the SD card."""
    candidates: List[Path] = []

    for mount in _removable_mounts():
        candidates.append(mount / "recordings")

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
    """Drop the DC step / click a USB capture device emits when it opens."""
    skip = int(rate * OPEN_TRANSIENT_MS / 1000)
    if skip <= 0 or samples.size <= skip * 2:
        return samples
    return samples[skip:]


def _resample_to_16k(samples: np.ndarray, rate: int) -> np.ndarray:
    if rate == SAMPLE_RATE or samples.size == 0:
        return samples
    if rate % SAMPLE_RATE == 0:                     # box-filtered decimation
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
    if rms < 1.0:                                   # essentially digital silence
        return centred

    target_rms = (10.0 ** (NORMALIZE_TARGET_DBFS / 20.0)) * 32768.0
    gain = min(target_rms / rms, NORMALIZE_MAX_GAIN)
    if gain <= 1.0:
        return centred                              # already loud enough
    boosted = centred * gain
    peak = float(np.max(np.abs(boosted)))
    if peak > 32000.0:
        boosted *= 32000.0 / peak
    return boosted


def _bandpass(samples: np.ndarray, rate: int = SAMPLE_RATE,
              low_freq: int = 300, high_freq: int = 3400) -> np.ndarray:
    """Single-pole IIR bandpass over the telephony speech band.

    Skipped entirely when scipy is missing: a pure-Python version costs seconds
    of a blind user's time, and unfiltered-but-normalized audio transcribes
    better than filtered audio that arrives four seconds late.
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
    card = MIC_DEVICE.split(":", 1)[1].split(",")[0].strip()
    return card or None


def _unmute_microphone():
    """Raise and uncmute the capture control. Both calls are best-effort.

    A muted capture mixer records digital silence on a perfectly good card, and
    the resulting "I didn't catch that" sends the user hunting for a loose plug.
    Cheaper to just set it every time the module loads.
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


def _beep(freq: float = 880.0, duration: float = 0.12):
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
        subprocess.run([aplay, "-q", "-D", PLAYBACK_DEVICE, path], timeout=4,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.unlink(path)
    except Exception as e:
        logger.debug(f"Earcon skipped: {e}")


# ── THE ONE CAPTURE PATH: arecord ───────────────────────────
def record(stop_check: Optional[Callable[[], bool]] = None,
           max_seconds: Optional[float] = None,
           speak_fn: Optional[Callable] = None) -> Optional[str]:
    """Record until the user stops it. Returns the saved WAV path, or None.

    Recording starts as soon as this is called — there is no "press to start".
    It ends on three Button-3 presses, on stop_check(), on ENTER at a terminal,
    on stop(), or at the max_seconds backstop.

    No silence detection anywhere: pausing mid-sentence ("what is the capital
    of … Australia?") must not end the recording, which is what the old
    VAD path did.
    """
    global _last_recording

    _abort.clear()
    _set_error(None)
    reset_presses()

    if not shutil.which("arecord"):
        _set_error("The recording program is missing on this device.")
        logger.error("arecord is not installed (apt install alsa-utils).")
        return None

    backstop = float(max_seconds or MANUAL_MAX_S)
    duration = max(1, int(round(backstop)))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_path = tmp.name

    cmd = ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-r", str(SAMPLE_RATE),
           "-c", "1", "-d", str(duration), "-t", "wav", temp_path]

    if speak_fn:
        speak_fn("Recording. Press button three, three times, when you are done.")
    _beep()

    logger.info(f"Recording from {MIC_DEVICE} (backstop {duration}s)")
    tty = sys.stdin is not None and sys.stdin.isatty()
    if tty:
        print(f"\n🔴 Recording from {MIC_DEVICE} — three presses of Button 3, "
              f"or ENTER here, to stop…")

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

            if _presses_say_stop():
                stopped_by = "button 3"
                logger.info("Three presses of Button 3 — ending recording.")
                process.terminate()
                break

            if stop_check is not None:
                try:
                    if stop_check():
                        stopped_by = "stop_check"
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
            reason = detail[-1] if detail else "no audio was captured"
            logger.error(f"arecord failed on {MIC_DEVICE}: {reason}")
            _set_error("I could not use the microphone. "
                       "Please check that it is plugged in.")
            return None

        # arecord was killed mid-write, so its RIFF header can still claim zero
        # frames even though megabytes of samples were written. Read the
        # payload directly and write a correct header ourselves — a file saved
        # with the truncated header is the reason `aplay` refused to play it.
        with wave.open(temp_path, "rb") as wf:
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        if not frames:
            with open(temp_path, "rb") as raw:
                frames = raw.read()[44:]
            rate = SAMPLE_RATE
            logger.debug("WAV header was truncated; read payload directly.")

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
            print(f"💾 Saved: {out_path}  ({seconds:.1f}s, {level:.0f} dBFS)")
        return out_path

    except Exception as e:
        logger.error(f"Recording failed: {e}")
        _set_error("Something went wrong with the microphone. Please try again.")
        return None
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def play(path: Optional[str] = None) -> bool:
    """Play a recording back through the earphone. Defaults to the latest one."""
    target = path or _last_recording
    if not target or not os.path.exists(target):
        logger.warning("Nothing to play back.")
        return False
    aplay = shutil.which("aplay")
    if not aplay:
        return False
    try:
        subprocess.run([aplay, "-q", "-D", PLAYBACK_DEVICE, target], timeout=300,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        logger.warning(f"Playback failed: {e}")
        return False


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
            continue                        # skip, rather than pretend to try
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

    The signature matches v4 so existing callers keep working, but there is
    only one behaviour now: max_seconds is a backstop and manual_stop is
    ignored. Returns the recognised text, or None — and when None,
    get_last_error() explains why in a sentence fit to be read aloud.
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

    # The audio is on the pen drive either way, so this is recoverable: the
    # clip can be re-run later with transcribe_file().
    if get_last_error() is None:
        _set_error("I didn't catch that. Please try again.")
    return None


# Kept so older callers do not break; both are the one mode now.
def listen_with_vad(lang: str = "en-IN",
                    speak_fn: Optional[Callable] = None) -> Optional[str]:
    return listen(lang, speak_fn)


def diagnostics() -> dict:
    """Machine-readable state for the startup self-test."""
    return {
        "mic_device": MIC_DEVICE,
        "playback_device": PLAYBACK_DEVICE,
        "arecord": bool(shutil.which("arecord")),
        "aplay": bool(shutil.which("aplay")),
        "recording_dir": RECORDING_DIR,
        "recording_dir_is_usb": bool(RECORDING_DIR) and any(
            str(RECORDING_DIR).startswith(str(m)) for m in _removable_mounts()),
        "usb_mounts": [str(m) for m in _removable_mounts()],
        "vosk": VOSK_AVAILABLE,
        "vosk_model": str(_vosk_model_dir) if _vosk_model_dir else None,
        "speech_recognition": SR_AVAILABLE,
        "sphinx": SPHINX_AVAILABLE,
        "engines_configured": list(STT_ENGINES),
        "engines_active": _active_engines(),
        "stop_press_count": STOP_PRESS_COUNT,
        "stop_window_seconds": STOP_WINDOW_S,
        "manual_max_seconds": MANUAL_MAX_S,
    }


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: (stop(), sys.exit(0)))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("Voice Module v5 — single mode")
    for key, value in diagnostics().items():
        print(f"  {key:22} {value}")

    if not diagnostics()["recording_dir_is_usb"]:
        print("\n⚠  The USB pen drive is not mounted — recordings will go to "
              f"{RECORDING_DIR} instead.")

    started = time.time()
    wav = record(speak_fn=print)
    if wav:
        print(f"\n▶  Playing it back on {PLAYBACK_DEVICE}…")
        play(wav)
        print("📝 Transcribing…")
        print(f"Result: {transcribe_file(wav)}   ({time.time() - started:.1f}s total)")
    print(f"Reason: {get_last_error()}")
