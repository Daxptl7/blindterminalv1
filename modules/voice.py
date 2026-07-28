"""
voice.py — BlindAssist Project (OPTIMIZED)
============================================
Fast voice capture with WebRTC VAD for precise speech detection.
Reduces ambient noise adjustment from 1s to 0.3s.
Uses energy-based pre-detection to skip silence.
"""

import sys
import signal
import logging
import time
import io
import wave
import collections

from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger("VoiceModule")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "voice.log"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── CONFIGURATION ───────────────────────────────────────────
SAMPLE_RATE = 16000  # WebRTC VAD requires 8k, 16k, 32k, or 48k
FRAME_DURATION = 30  # ms (10, 20, or 30)
VAD_AGGRESSIVENESS = 2  # 0-3 (3 = most aggressive, filters more noise)
TIMEOUT_SECONDS = 8
PHRASE_SECONDS = 10
PAUSE_SECONDS = 1.5  # Stop listening after this much silence
PRE_BUFFER_SECONDS = 0.3  # Keep this much audio before speech starts

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


def _read_audio_chunk(stream, chunk_size: int) -> bytes:
    """Read raw audio bytes from PyAudio stream."""
    return stream.read(chunk_size, exception_on_overflow=False)


def _energy_detect(audio_bytes: bytes, threshold: int = 500) -> bool:
    """Simple energy-based voice detection fallback."""
    import struct
    # Convert bytes to 16-bit samples and compute RMS
    n_samples = len(audio_bytes) // 2
    if n_samples == 0:
        return False
    samples = struct.unpack(f'<{n_samples}h', audio_bytes[:n_samples * 2])
    rms = (sum(s * s for s in samples) / n_samples) ** 0.5
    return rms > threshold


def listen(lang: str = 'en-IN', speak_fn: Optional[Callable] = None) -> Optional[str]:
    """
    Optimized voice recognition locked to Card 4 using arecord.
    Saves every recording directly to the USB drive.
    """
    import speech_recognition as sr
    import time
    import os
    import subprocess

    prompt = "Speak now"
    logger.info(f"Prompt: {prompt}")

    if speak_fn:
        speak_fn(prompt)

    # 1. GENERATE TIMESTAMP AND USB PATH
    timestamp = int(time.time())
    usb_path = "/mnt/aet_usb/data"
    os.makedirs(usb_path, exist_ok=True)
    wav_path = f"{usb_path}/voice_{timestamp}.wav"

    logger.info(f"Recording to {wav_path} (Locked to Card 4)...")

# 2. RECORD AUDIO DIRECTLY TO PENDRIVE (MANUAL CONTROL)
    cmd = [
        "arecord",
        "-D", "plughw:2,0",  # Locks onto your specific USB mic
        "-f", "S16_LE",      # High quality 16-bit audio
        "-r", "16000",       # 16kHz for Google Speech API
        # Notice we removed "-d", "5" so it records infinitely until you stop it!
        wav_path
    ]
    
    try:
        # Popen starts the recording in the background without freezing the script
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # This halts the Python script and waits for your command to stop
        input("\n🔴 Recording... Speak your command, then press ENTER to stop and process: ")
        
        # When you press Enter, it kills the recording process cleanly
        process.terminate()
        process.wait() 
        
    except Exception as e:
        logger.error(f"arecord failed: {e}")
        return None
    # 3. TRANSCRIBE THE SAVED FILE
    logger.info("Audio saved to USB, transcribing...")
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
            
        text = recognizer.recognize_google(audio, language=lang)
        text = text.strip()
        logger.info(f"Transcribed: '{text}'")
        return text
            
    except OSError as e:
        logger.error(f"Microphone could not be opened: {e}")
        return None
    except sr.WaitTimeoutError:
        logger.warning("Timeout — no speech detected.")
        return None
    except sr.UnknownValueError:
        logger.warning("Could not understand audio.")
        return None
    except sr.RequestError as e:
        logger.error(f"Google API error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None


def listen_with_vad(lang: str = 'en-IN', speak_fn: Optional[Callable] = None) -> Optional[str]:
    """
    Advanced listening with WebRTC VAD for precise speech boundaries.
    Cuts off immediately when user stops speaking.
    """
    if not VAD_AVAILABLE:
        return listen(lang, speak_fn)
    
    import pyaudio
    import speech_recognition as sr
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    CHUNK_DURATION_MS = 30
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
    
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=2,
        frames_per_buffer=CHUNK_SIZE
    )
    
    # Pre-buffer
    ring_buffer = collections.deque(maxlen=int(PRE_BUFFER_SECONDS * 1000 / CHUNK_DURATION_MS))
    triggered = False
    voiced_frames = []
    num_voiced = 0
    num_unvoiced = 0
    
    # VAD parameters
    RING_BUFFER_MAX = int(PRE_BUFFER_SECONDS * 1000 / CHUNK_DURATION_MS)
    TRIGGER_THRESHOLD = 3  # consecutive voiced frames to trigger
    UNTRIGGER_THRESHOLD = 10  # consecutive unvoiced to stop
    
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
    
    # Convert to AudioData for recognition
    raw_data = b''.join(voiced_frames)
    
    
    
# Convert to AudioData for recognition
    raw_data = b''.join(voiced_frames)
    
    # 1. GENERATE TIMESTAMP AND USB PATH FOR VAD
    import os
    import time
    timestamp = int(time.time())
    usb_path = "/mnt/aet_usb/data"
    os.makedirs(usb_path, exist_ok=True)
    wav_path = f"{usb_path}/voice_vad_{timestamp}.wav"
    
    # 2. SAVE DIRECTLY TO PEN DRIVE
    with wave.open(wav_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_data)
    
    logger.info(f"Saved VAD audio to USB: {wav_path}")
    
    # 3. TRANSCRIBE FROM THE SAVED FILE
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)    
    try:
        text = recognizer.recognize_google(audio, language=lang)
        logger.info(f"VAD transcribed: '{text}'")
        return text.strip()
    except sr.UnknownValueError:
        logger.warning("VAD could not understand.")
        return None
    except sr.RequestError as e:
        logger.error(f"VAD API error: {e}")
        return None


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    print("Voice Module Optimized Test")
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
