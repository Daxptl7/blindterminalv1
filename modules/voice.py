"""
voice.py — BlindAssist Project (OPTIMIZED v2)
============================================
Fast voice capture with WebRTC VAD for precise speech detection.

Optimizations over v1:
- Multi-engine STT: Vosk (offline, free) → Google (online, free tier) → PocketSphinx (offline fallback)
- Audio preprocessing: Bandpass filter (300Hz–3400Hz) removes ambient noise before transcription
- Fixed duplicate raw_data block in listen_with_vad
- Improved VAD parameters for better speech boundary detection
- Reduced ambient noise adjustment from 1s to 0.3s
- Uses energy-based pre-detection to skip silence
"""

import sys
import signal
import json
import logging
import os
import shutil
import time
import io
import wave
import struct
import collections

from pathlib import Path
from typing import Optional, Callable

import numpy as np

logger = logging.getLogger("VoiceModule")

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "voice.log"
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_settings() -> dict:
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


_settings = _load_settings()

# ── CONFIGURATION ───────────────────────────────────────────
SAMPLE_RATE = 16000  # WebRTC VAD requires 8k, 16k, 32k, or 48k
FRAME_DURATION = 30  # ms (10, 20, or 30)
VAD_AGGRESSIVENESS = int(_settings.get("vad_aggressiveness", 2))  # 0-3
TIMEOUT_SECONDS = float(_settings.get("voice_timeout_seconds", 8))
PHRASE_SECONDS = 10
PAUSE_SECONDS = 1.5  # Stop listening after this much silence
PRE_BUFFER_SECONDS = 0.3  # Keep this much audio before speech starts

# Hardware/paths are configuration, not constants. These used to be hardcoded
# to one developer's setup ("plughw:2,0", input_device_index=2,
# "/mnt/aet_usb/data"), so the module only worked on that exact machine with
# that exact USB stick mounted.
MIC_DEVICE = _settings.get("mic_device", "default")
MIC_INDEX = _settings.get("mic_input_index")           # None = system default
RECORD_MAX_SECONDS = float(_settings.get("voice_record_max_seconds", 8))
RECORDING_DIR = _settings.get("voice_recording_dir", str(BASE_DIR / "data" / "recordings"))
# Recordings used to accumulate forever (two WAVs per utterance, never
# deleted), which eventually fills the SD card on a device meant to run for
# months unattended.
KEEP_RECORDINGS = int(_settings.get("voice_keep_recordings", 20))


def _pyaudio_available() -> bool:
    try:
        import pyaudio  # noqa: F401
        return True
    except Exception:
        return False


def _prune_recordings(directory: str, keep: int = KEEP_RECORDINGS):
    """Keep only the newest `keep` recordings so storage cannot grow unbounded."""
    if keep <= 0:
        return
    try:
        files = sorted(
            (p for p in Path(directory).glob("voice_*.wav")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in files[keep:]:
            try:
                stale.unlink()
            except OSError:
                pass
    except Exception as e:
        logger.debug(f"Recording cleanup skipped: {e}")


def _listen_via_speechrecognition(lang: str, max_seconds: float) -> Optional[str]:
    """Last-resort capture path when neither VAD nor arecord is usable."""
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.Microphone(sample_rate=SAMPLE_RATE) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio = recognizer.listen(source, timeout=max_seconds, phrase_time_limit=max_seconds)
        return _multi_engine_transcribe(audio.get_raw_data(convert_rate=SAMPLE_RATE, convert_width=2),
                                        lang, SAMPLE_RATE)
    except Exception as e:
        logger.error(f"speech_recognition capture failed: {e}")
        return None

# ── STT ENGINE PRIORITY ────────────────────────────────────
# The system tries each engine in order. First successful result wins.
# Vosk (offline, free, accurate) → Google (online, free tier) → PocketSphinx (offline, basic)
STT_ENGINES = ["vosk", "google", "sphinx"]

# ── VAD SETUP (conditional) ─────────────────────────────────
VAD_AVAILABLE = False
vad = None
try:
    import webrtcvad
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    VAD_AVAILABLE = True
    logger.info("WebRTC VAD initialized.")
except ImportError:
    logger.info("webrtcvad not installed. Using standard speech_recognition.")
except Exception as e:
    logger.warning(f"WebRTC VAD unavailable: {e}. Using energy-based detection.")

# ── VOSK SETUP (conditional) ────────────────────────────────
VOSK_AVAILABLE = False
_vosk_model = None
try:
    from vosk import Model as VoskModel, KaldiRecognizer
    import json as _json

    # Vosk model path: expects a downloaded model folder (e.g., vosk-model-small-en-in-0.4)
    _vosk_model_path = BASE_DIR / "models" / "vosk"
    if _vosk_model_path.exists() and _vosk_model_path.is_dir():
        _vosk_model = VoskModel(str(_vosk_model_path))
        VOSK_AVAILABLE = True
        logger.info(f"Vosk STT model loaded from: {_vosk_model_path}")
    else:
        logger.info(f"Vosk model directory not found at {_vosk_model_path}. Vosk STT disabled.")
        logger.info("Download a free model from: https://alphacephei.com/vosk/models")
        logger.info("  Recommended: vosk-model-small-en-in-0.4 (~40MB, Indian English)")
        logger.info("  Extract to: models/vosk/")
except ImportError:
    logger.info("Vosk not installed. Install with: pip install vosk")
except Exception as e:
    logger.warning(f"Vosk init failed: {e}")


# ── AUDIO PREPROCESSING ────────────────────────────────────

def _bandpass_filter(audio_bytes: bytes, sample_rate: int = 16000,
                     low_freq: int = 300, high_freq: int = 3400) -> bytes:
    """
    Simple single-pole IIR bandpass filter for voice frequency isolation (300Hz–3400Hz).
    Removes low-frequency hum (fans, AC, traffic) and high-frequency hiss (electronics).
    This significantly improves STT accuracy in noisy environments.

    Uses a lightweight integer-math approach that runs efficiently on Raspberry Pi
    without requiring scipy or numpy signal processing libraries.
    """
    n_samples = len(audio_bytes) // 2
    if n_samples < 2:
        return audio_bytes

    # IIR filter coefficients (derived from bilinear transform approximation)
    # These are tuned for 16kHz sample rate and 300-3400Hz passband
    import math
    dt = 1.0 / sample_rate
    rc_high = 1.0 / (2.0 * math.pi * low_freq)  # High-pass RC (removes below 300Hz)
    rc_low = 1.0 / (2.0 * math.pi * high_freq)   # Low-pass RC (removes above 3400Hz)
    alpha_hp = rc_high / (rc_high + dt)
    alpha_lp = dt / (rc_low + dt)

    samples = np.frombuffer(audio_bytes[:n_samples * 2], dtype='<i2').astype(np.float64)

    # PERFORMANCE: these two recursions used to be per-sample Python loops
    # (plus a per-sample list comprehension) — ~400k interpreted iterations for
    # an 8-second clip, adding seconds of dead air on a Pi between the user
    # speaking and the device answering. scipy.signal.lfilter runs the exact
    # same difference equations in C.
    #   high-pass:  y[n] = a*(y[n-1] + x[n] - x[n-1])
    #   low-pass:   y[n] = (1-b)*y[n-1] + b*x[n]
    try:
        from scipy.signal import lfilter
        hp_out = lfilter([alpha_hp, -alpha_hp], [1.0, -alpha_hp], samples)
        lp_out = lfilter([alpha_lp], [1.0, -(1.0 - alpha_lp)], hp_out)
    except ImportError:
        # Pure-Python fallback (correct, just slower) when scipy is absent.
        hp_out = np.empty(n_samples, dtype=np.float64)
        hp_out[0] = samples[0]
        for i in range(1, n_samples):
            hp_out[i] = alpha_hp * (hp_out[i - 1] + samples[i] - samples[i - 1])
        lp_out = np.empty(n_samples, dtype=np.float64)
        lp_out[0] = hp_out[0]
        for i in range(1, n_samples):
            lp_out[i] = lp_out[i - 1] + alpha_lp * (hp_out[i] - lp_out[i - 1])

    # Clamp to int16 range and pack back to bytes
    return np.clip(lp_out, -32768, 32767).astype('<i2').tobytes()


def _read_audio_chunk(stream, chunk_size: int) -> bytes:
    """Read raw audio bytes from PyAudio stream."""
    return stream.read(chunk_size, exception_on_overflow=False)


def _energy_detect(audio_bytes: bytes, threshold: int = 500) -> bool:
    """Simple energy-based voice detection fallback."""
    # Convert bytes to 16-bit samples and compute RMS
    n_samples = len(audio_bytes) // 2
    if n_samples == 0:
        return False
    samples = struct.unpack(f'<{n_samples}h', audio_bytes[:n_samples * 2])
    rms = (sum(s * s for s in samples) / n_samples) ** 0.5
    return rms > threshold


# ── STT ENGINE FUNCTIONS ───────────────────────────────────

def _transcribe_vosk(audio_bytes: bytes, sample_rate: int = 16000) -> Optional[str]:
    """
    Transcribe audio using Vosk (100% offline, free, no API key needed).
    Vosk uses Kaldi's neural network models and supports 20+ languages.
    Accuracy: ~85-92% for clean speech (comparable to Google for short commands).
    """
    if not VOSK_AVAILABLE or not _vosk_model:
        return None

    try:
        recognizer = KaldiRecognizer(_vosk_model, sample_rate)
        recognizer.SetWords(True)

        # Feed audio in chunks for streaming-style recognition
        chunk_size = 4000  # 2000 samples × 2 bytes each
        for i in range(0, len(audio_bytes), chunk_size):
            recognizer.AcceptWaveform(audio_bytes[i:i + chunk_size])

        result = _json.loads(recognizer.FinalResult())
        text = result.get("text", "").strip()

        if text:
            logger.info(f"Vosk transcribed: '{text}'")
            return text
        return None

    except Exception as e:
        logger.error(f"Vosk transcription error: {e}")
        return None


def _transcribe_google(audio_bytes: bytes, lang: str = 'en-IN',
                       sample_rate: int = 16000) -> Optional[str]:
    """
    Transcribe audio using Google Speech Recognition (free tier, online).
    Free quota: ~50 requests/day without an API key.
    Accuracy: ~92-97% for clean speech.
    """
    import speech_recognition as sr

    try:
        recognizer = sr.Recognizer()
        audio_data = sr.AudioData(audio_bytes, sample_rate, 2)  # 2 bytes per sample (16-bit)
        text = recognizer.recognize_google(audio_data, language=lang)
        text = text.strip()
        if text:
            logger.info(f"Google transcribed: '{text}'")
            return text
        return None

    except sr.UnknownValueError:
        logger.warning("Google could not understand audio.")
        return None
    except sr.RequestError as e:
        logger.error(f"Google API error (offline?): {e}")
        return None
    except Exception as e:
        logger.error(f"Google transcription error: {e}")
        return None


def _transcribe_sphinx(audio_bytes: bytes, sample_rate: int = 16000) -> Optional[str]:
    """
    Transcribe audio using CMU PocketSphinx (100% offline, free, lightweight).
    Accuracy: ~70-80% (lower than Vosk/Google but works without any model download).
    Best for: simple command recognition on resource-constrained devices.
    """
    import speech_recognition as sr

    try:
        recognizer = sr.Recognizer()
        audio_data = sr.AudioData(audio_bytes, sample_rate, 2)
        text = recognizer.recognize_sphinx(audio_data)
        text = text.strip()
        if text:
            logger.info(f"Sphinx transcribed: '{text}'")
            return text
        return None

    except sr.UnknownValueError:
        logger.warning("Sphinx could not understand audio.")
        return None
    except sr.RequestError as e:
        logger.error(f"Sphinx error: {e}")
        return None
    except Exception as e:
        logger.error(f"Sphinx transcription error: {e}")
        return None


def _multi_engine_transcribe(audio_bytes: bytes, lang: str = 'en-IN',
                             sample_rate: int = 16000) -> Optional[str]:
    """
    Multi-engine STT with automatic fallback chain.
    Tries each engine in priority order until one succeeds.
    Order: Vosk (offline, fast) → Google (online, accurate) → Sphinx (offline, basic)
    """
    engine_map = {
        "vosk": lambda: _transcribe_vosk(audio_bytes, sample_rate),
        "google": lambda: _transcribe_google(audio_bytes, lang, sample_rate),
        "sphinx": lambda: _transcribe_sphinx(audio_bytes, sample_rate),
    }

    for engine_name in STT_ENGINES:
        fn = engine_map.get(engine_name)
        if fn:
            result = fn()
            if result:
                return result
            logger.info(f"{engine_name} returned no result, trying next engine...")

    logger.warning("All STT engines failed to transcribe.")
    return None


# ── MAIN LISTEN FUNCTIONS ──────────────────────────────────

def listen(lang: str = 'en-IN', speak_fn: Optional[Callable] = None,
           max_seconds: Optional[float] = None) -> Optional[str]:
    """
    Capture one spoken utterance and transcribe it. Fully hands-free.

    HANDS-FREE FIX
    --------------
    This function used to start `arecord` and then call
    `input("press ENTER to stop")` — it could not complete without a sighted
    user physically pressing a key on a keyboard the product does not have.
    On the headless Pi that blocked forever, taking every voice-driven mode
    (3, 4, 7, 9 and the Confidential Mode prompt) down with it.

    Recording now ends on its own:
      1. WebRTC VAD stops as soon as the user stops speaking (preferred), else
      2. a fixed, bounded recording window via arecord.
    A keypress still stops it early when a real terminal is attached, but is
    never required.
    """
    import os
    import subprocess

    if max_seconds is None:
        max_seconds = RECORD_MAX_SECONDS

    if speak_fn:
        speak_fn("Speak now")
    logger.info("Listening for one utterance…")

    # Preferred path: VAD detects the end of speech by itself.
    if VAD_AVAILABLE and _pyaudio_available():
        result = listen_with_vad(lang, speak_fn=None)
        if result:
            return result
        logger.info("VAD capture produced nothing; falling back to timed recording.")

    if not shutil.which("arecord"):
        logger.warning("arecord not found; trying speech_recognition microphone.")
        return _listen_via_speechrecognition(lang, max_seconds)

    # 1. RECORDING PATH (configurable — no longer hardcoded to a USB mount)
    timestamp = int(time.time())
    os.makedirs(RECORDING_DIR, exist_ok=True)
    wav_path = os.path.join(RECORDING_DIR, f"voice_{timestamp}.wav")

    logger.info(f"Recording up to {max_seconds:.0f}s to {wav_path} (device={MIC_DEVICE})")

    # 2. RECORD WITH A HARD DURATION LIMIT so it always terminates on its own
    cmd = [
        "arecord",
        "-D", MIC_DEVICE,
        "-f", "S16_LE",
        "-r", str(SAMPLE_RATE),
        "-c", "1",
        "-d", str(int(max_seconds)),   # self-terminating
        wav_path,
    ]

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Optional early stop — only when a real terminal is attached.
        if sys.stdin is not None and sys.stdin.isatty():
            import select
            print(f"\n🔴 Recording (auto-stops in {int(max_seconds)}s, or press ENTER)…")
            deadline = time.time() + max_seconds
            while process.poll() is None and time.time() < deadline:
                if select.select([sys.stdin], [], [], 0.2)[0]:
                    sys.stdin.readline()
                    break
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    except Exception as e:
        logger.error(f"arecord failed: {e}")
        return None

    # 3. LOAD AND PREPROCESS THE AUDIO
    logger.info("Recording finished, preprocessing and transcribing…")

    try:
        # Extract raw PCM from WAV container
        with wave.open(wav_path, 'rb') as wf:
            raw_audio = wf.readframes(wf.getnframes())

        # Apply bandpass filter to remove ambient noise (300Hz–3400Hz voice band)
        filtered_audio = _bandpass_filter(raw_audio, SAMPLE_RATE)

        _prune_recordings(RECORDING_DIR)

        # 4. MULTI-ENGINE TRANSCRIPTION (Vosk → Google → Sphinx)
        text = _multi_engine_transcribe(filtered_audio, lang, SAMPLE_RATE)
        if text:
            logger.info(f"Final transcription: '{text}'")
            return text

        # If filtered audio failed, retry with raw audio (filter might have been too aggressive)
        logger.info("Filtered audio produced no result, retrying with raw audio...")
        text = _multi_engine_transcribe(raw_audio, lang, SAMPLE_RATE)
        if text:
            logger.info(f"Raw audio transcription: '{text}'")
            return text

        logger.warning("All transcription attempts failed.")
        return None

    except OSError as e:
        logger.error(f"Audio file could not be opened: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None


def listen_with_vad(lang: str = 'en-IN', speak_fn: Optional[Callable] = None) -> Optional[str]:
    """
    Advanced listening with WebRTC VAD for precise speech boundaries.
    Cuts off immediately when user stops speaking.
    Now includes audio preprocessing and multi-engine STT.
    """
    if not VAD_AVAILABLE:
        return listen(lang, speak_fn)

    import pyaudio
    import os

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    CHUNK_DURATION_MS = 30
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

    pa = pyaudio.PyAudio()
    # input_device_index was hardcoded to 2, which is whichever device happened
    # to be enumerated second on one particular machine. None = system default,
    # overridable via mic_input_index in settings.json.
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=MIC_INDEX,
        frames_per_buffer=CHUNK_SIZE
    )

    # Pre-buffer to capture audio slightly before speech onset
    ring_buffer = collections.deque(maxlen=int(PRE_BUFFER_SECONDS * 1000 / CHUNK_DURATION_MS))
    triggered = False
    voiced_frames = []
    num_voiced = 0
    num_unvoiced = 0

    # Improved VAD trigger/untrigger thresholds for better boundary detection
    TRIGGER_THRESHOLD = 3    # consecutive voiced frames to start capturing (90ms of speech)
    UNTRIGGER_THRESHOLD = 15  # consecutive unvoiced frames to stop (450ms of silence — more patient)

    logger.info("VAD listening started...")
    start_time = time.time()

    try:
        while True:
            if time.time() - start_time > TIMEOUT_SECONDS:
                logger.warning("VAD timeout.")
                break

            chunk = _read_audio_chunk(stream, CHUNK_SIZE)
            is_speech = vad.is_speech(chunk, SAMPLE_RATE)

            if not triggered:
                ring_buffer.append(chunk)
                num_voiced = num_voiced + 1 if is_speech else 0
                if num_voiced >= TRIGGER_THRESHOLD:
                    triggered = True
                    voiced_frames.extend(ring_buffer)
                    ring_buffer.clear()
                    logger.info("Speech detected.")
            else:
                voiced_frames.append(chunk)
                num_unvoiced = num_unvoiced + 1 if not is_speech else 0
                if num_unvoiced >= UNTRIGGER_THRESHOLD:
                    logger.info("Speech ended.")
                    break

    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    if not voiced_frames:
        return None

    # Combine captured audio frames into a single byte string
    raw_data = b''.join(voiced_frames)

    # 1. SAVE THE CAPTURE (configurable location, pruned to a bounded count)
    timestamp = int(time.time())
    os.makedirs(RECORDING_DIR, exist_ok=True)
    wav_path = os.path.join(RECORDING_DIR, f"voice_{timestamp}.wav")

    try:
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(raw_data)
        _prune_recordings(RECORDING_DIR)
        logger.info(f"Saved VAD audio: {wav_path}")
    except OSError as e:
        # A full or unwritable disk must not cost the user their question.
        logger.warning(f"Could not save recording ({e}); continuing with in-memory audio.")

    # 3. PREPROCESS: Apply bandpass filter to remove ambient noise
    filtered_data = _bandpass_filter(raw_data, SAMPLE_RATE)

    # 4. MULTI-ENGINE TRANSCRIPTION (Vosk → Google → Sphinx)
    text = _multi_engine_transcribe(filtered_data, lang, SAMPLE_RATE)
    if text:
        return text

    # Retry with raw audio if filtered version failed
    logger.info("Filtered VAD audio produced no result, retrying with raw...")
    text = _multi_engine_transcribe(raw_data, lang, SAMPLE_RATE)
    if text:
        return text

    logger.warning("VAD: All transcription attempts failed.")
    return None


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    print("Voice Module Optimized v2 Test")
    print(f"  Vosk available: {VOSK_AVAILABLE}")
    print(f"  VAD available:  {VAD_AVAILABLE}")
    print(f"  STT priority:   {' → '.join(STT_ENGINES)}")
    print("1: Standard listen  2: VAD listen  Q: Quit")

    while True:
        choice = input("\nSelect: ").strip().lower()
        if choice == 'q':
            break
        if choice == '1':
            result = listen('en-IN')
            print(f"Result: {result}")
        elif choice == '2':
            if VAD_AVAILABLE:
                result = listen_with_vad('en-IN')
                print(f"VAD Result: {result}")
            else:
                print("VAD not available. Install: pip install webrtcvad-wheels")

