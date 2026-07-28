"""
tts.py — BlindAssist Project (UPGRADED for gTTS)
==========================================
Zero-disk streaming TTS using pyttsx3 and gTTS in-memory buffers.
Supports native Hindi, Gujarati, and Indian English via Google,
falling back to offline pyttsx3 if Wi-Fi drops.
"""

import logging
import signal
import sys
import threading
import json
import tempfile
import os
import io
from pathlib import Path
from queue import Queue

logger = logging.getLogger("TTSModule")

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "tts.log"
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_settings_static():
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


_static_settings = _load_settings_static()
SPEAKER_DEVICE = _static_settings.get("speaker_device", "plughw:3,0")
BONE_DEVICE = _static_settings.get("bone_device", "hw:3,0")

# ── PYGAME INIT (conditional) ───────────────────────────────
PYGAME_AVAILABLE = False
_current_device = None
try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
    PYGAME_AVAILABLE = True
    _current_device = "default"
    logger.info("pygame.mixer initialized on default device.")
except ImportError:
    logger.info("pygame not installed — using fallback audio playback.")
except Exception as e:
    logger.warning(f"pygame mixer init failed: {e}")


def switch_output_device(device_name: str) -> bool:
    global _current_device
    if not PYGAME_AVAILABLE:
        return False
    if device_name == _current_device:
        return True
    try:
        pygame.mixer.quit()
        pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512,
                           devicename=device_name)
        _current_device = device_name
        logger.info(f"Audio output switched to {device_name}")
        return True
    except TypeError:
        logger.warning("This pygame build does not support devicename= routing.")
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
        except Exception:
            pass
        return False
    except Exception as e:
        logger.error(f"Failed to switch audio device to {device_name}: {e}")
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
        except Exception:
            pass
        return False

def use_speaker_output():
    return switch_output_device(SPEAKER_DEVICE)

def use_bone_conduction_output():
    return switch_output_device(BONE_DEVICE)


class TTSManager:
    def __init__(self):
        self.settings = self._load_settings()
        self.queue = Queue()
        self.running = True
        self._current_rate = self.settings.get("tts_rate", 150)
        self._speaking = threading.Event()

        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', self._current_rate)
            self._engine.setProperty('volume', self.settings.get("tts_volume", 1.0))
            self._engine_available = True
        except Exception as e:
            logger.warning(f"pyttsx3 init failed: {e}. TTS will print to console.")
            self._engine = None
            self._engine_available = False

        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        logger.info("TTS Manager ready.")

    def _load_settings(self):
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            return {"tts_rate": 150, "tts_volume": 1.0, "tts_engine": "gtts"}

    def _speak_to_bytes(self, text: str, lang: str) -> bytes:
        engine_choice = self.settings.get("tts_engine", "pyttsx3")

        # ATTEMPT GTTS FIRST IF SELECTED
        if engine_choice == "gtts":
            try:
                from gtts import gTTS
                # Map languages: eng -> en, hin -> hi, guj -> gu
                lang_map = {'eng': 'en', 'hin': 'hi', 'guj': 'gu'}
                g_lang = lang_map.get(lang.lower(), 'en')
                
                # Force Indian accent for English
                tld = 'co.in' if g_lang == 'en' else 'com'

                tts = gTTS(text=text, lang=g_lang, tld=tld)
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                return fp.read()
            except Exception as e:
                logger.error(f"gTTS failed (Check Wi-Fi): {e}. Falling back to offline engine.")

        # OFFLINE PYTTSX3 FALLBACK
        if not self._engine_available:
            return b''

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self._engine.save_to_file(text, tmp_path)
            self._engine.runAndWait()
            with open(tmp_path, 'rb') as f:
                return f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _play_wav_bytes(self, wav_data: bytes):
        if not wav_data or not PYGAME_AVAILABLE:
            return

        try:
            sound = pygame.mixer.Sound(io.BytesIO(wav_data))
            self._speaking.set()
            sound.play()
            while pygame.mixer.get_busy() and self.running:
                pygame.time.wait(10)
            self._speaking.clear()
        except Exception as e:
            logger.error(f"Playback error: {e}")
            self._speaking.clear()

    def _process_queue(self):
        while self.running:
            text, lang = self.queue.get()
            if text is None:
                break

            try:
                if self._engine_available:
                    self._engine.setProperty('rate', self._current_rate)
                logger.info(f"Speaking: \"{text[:60]}...\"")

                # Pass language to our new function
                wav_bytes = self._speak_to_bytes(text, lang)
                if wav_bytes:
                    self._play_wav_bytes(wav_bytes)
                else:
                    print(f"[TTS] {text}")

            except Exception as e:
                logger.error(f"TTS error: {e}")
                print(f"[TTS FALLBACK] {text}")
            finally:
                self.queue.task_done()

    def speak(self, text: str, lang: str = 'eng', block: bool = False):
        if not text:
            return
        self.queue.put((text, lang))
        if block:
            self.queue.join()

    def set_rate(self, rate: int):
        self._current_rate = max(80, min(300, rate))

    def get_rate(self):
        return self._current_rate

    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def stop(self):
        if PYGAME_AVAILABLE:
            pygame.mixer.stop()
        self._speaking.clear()

    def shutdown(self):
        self.running = False
        self.queue.put((None, None))
        self.worker_thread.join(timeout=3)
        if PYGAME_AVAILABLE:
            pygame.mixer.quit()


_tts_manager = None
_tts_lock = threading.Lock()


def _get_manager():
    global _tts_manager
    if _tts_manager is None:
        with _tts_lock:
            if _tts_manager is None:
                _tts_manager = TTSManager()
    return _tts_manager


def speak(text: str, lang: str = 'eng', block: bool = False):
    _get_manager().speak(text, lang, block)


def set_rate(rate: int):
    _get_manager().set_rate(rate)


def get_rate() -> int:
    return _get_manager().get_rate()


def is_speaking() -> bool:
    return _get_manager().is_speaking()


def stop():
    _get_manager().stop()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda s, f: (_get_manager().shutdown(), sys.exit(0)))
    print("TTS Test — type text, QUIT to exit")
    while True:
        try:
            inp = input("Text: ").strip()
            if inp.upper() == "QUIT":
                break
            if inp:
                speak(inp, block=True)
        except (EOFError, KeyboardInterrupt):
            break
    _get_manager().shutdown()
