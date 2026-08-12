"""
tts.py — BlindAssist Project
============================
Centralised, non-blocking text-to-speech with a *guaranteed* output path.

Speech is this device's only output channel. The previous version generated
audio and then dropped it on the floor whenever pygame was unavailable —
producing total silence with no error and no console fallback. That failure
mode is now impossible: every utterance walks a playback chain and, if every
audio backend fails, is still printed to the console and logged.

Synthesis:  gTTS (online, native Hindi/Gujarati/Indian English) → pyttsx3 (offline)
Playback:   pygame → system player (afplay/aplay/mpg123) → pyttsx3 direct → console

Public API (unchanged, plus shutdown/flush):
    speak(text, lang='eng', block=False)
    set_rate(rate) / get_rate()
    is_speaking() / stop() / flush() / shutdown()
    use_speaker_output() / use_bone_conduction_output() / switch_output_device(name)
"""

import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from queue import Queue, Empty

logger = logging.getLogger("TTSModule")

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "tts.log"
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

IS_MACOS = platform.system() == "Darwin"


def _load_settings_static() -> dict:
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


_static_settings = _load_settings_static()


def _alsa_playback_devices() -> list:
    """ALSA playback devices that actually exist, as plughw:CARD,DEV strings.

    Confidential Mode routes private speech to one sound card and public
    speech to another. Pointing either at a card that is not present makes the
    private/public split silently collapse — which for this product means a
    blind user's private document can be read out of the wrong speaker. The
    configured names are therefore checked against the hardware at startup.
    """
    devices = []
    if IS_MACOS:
        return devices
    try:
        out = subprocess.run(["aplay", "-l"], capture_output=True, text=True,
                             timeout=5).stdout
        for line in out.splitlines():
            if not line.startswith("card "):
                continue
            try:
                card = line.split("card ")[1].split(":")[0].strip()
                dev = line.split("device ")[1].split(":")[0].strip()
                name = f"plughw:{card},{dev}"
                if name not in devices:
                    devices.append(name)
            except (IndexError, ValueError):
                continue
    except Exception as e:
        logger.debug(f"Could not enumerate ALSA playback devices: {e}")
    return devices


_AVAILABLE_OUTPUTS = _alsa_playback_devices()


def _validate_device(configured: str, role: str, fallback_index: int) -> str:
    """Return `configured` if the hardware has it, else the best substitute."""
    if IS_MACOS or not configured or configured == "default":
        return configured
    if not _AVAILABLE_OUTPUTS:
        return configured                 # cannot verify; trust the config
    if configured in _AVAILABLE_OUTPUTS:
        return configured

    substitute = _AVAILABLE_OUTPUTS[min(fallback_index, len(_AVAILABLE_OUTPUTS) - 1)]
    logger.error(
        f"{role} device {configured!r} from settings.json does not exist on this "
        f"machine (available: {', '.join(_AVAILABLE_OUTPUTS)}). Falling back to "
        f"{substitute!r} — fix {role.lower()}_device in settings.json."
    )
    return substitute


SPEAKER_DEVICE = _validate_device(
    _static_settings.get("speaker_device", "plughw:3,0"), "Speaker", 0)
BONE_DEVICE = _validate_device(
    _static_settings.get("bone_device", "plughw:2,0"), "Bone", 1)

# ── OPTIONAL BACKENDS ────────────────────────────────────────
PYGAME_AVAILABLE = False
_current_device = None
try:
    import pygame

    # gTTS returns 24 kHz MP3; matching the mixer rate avoids pitch-shifted
    # ("chipmunk") playback that the old hardcoded 22050 Hz produced.
    pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=1024)
    PYGAME_AVAILABLE = True
    _current_device = "default"
    logger.info("pygame.mixer initialised on default device.")
except ImportError:
    logger.info("pygame not installed — falling back to system audio players.")
except Exception as e:
    logger.warning(f"pygame mixer init failed: {e}")


def _find_player(fmt: str):
    """Return (binary, arg_builder) for the best available CLI audio player.

    `dev` is honoured only where the player supports it. When routing actually
    matters, _play_routed() is used instead — it guarantees the requested card
    is used or reports failure, rather than quietly falling back to default.
    """
    if IS_MACOS:
        afplay = shutil.which("afplay")
        if afplay:
            return afplay, lambda path, dev: [afplay, path]
        return None, None

    if fmt == "mp3":
        for name in ("mpg123", "mpg321"):
            binary = shutil.which(name)
            if binary:
                # mpg123/mpg321 accept an ALSA device via -a, enabling the
                # speaker/earphone routing Confidential Mode depends on.
                return binary, lambda path, dev, b=binary: (
                    [b, "-q", "-a", dev, path] if dev else [b, "-q", path])

    aplay = shutil.which("aplay")
    if aplay and fmt == "wav":
        return aplay, lambda path, dev: ([aplay, "-q", "-D", dev, path] if dev else [aplay, "-q", path])

    # ffplay plays anything but cannot select an ALSA card, so it is only a
    # non-routed fallback. (ffplay used to be listed in the mp3 loop above,
    # where its arg-builder closed over the loop's `binary` — a latent
    # NameError on any other code path through this function.)
    ffplay = shutil.which("ffplay")
    if ffplay:
        return ffplay, lambda path, dev: [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", path]

    return None, None


def _decode_mp3_to_wav(data: bytes) -> bytes:
    """Decode MP3 to 16-bit PCM WAV in memory. Returns b"" if not possible.

    gTTS returns MP3, but `aplay` — the only player on a stock Raspberry Pi OS
    image that can target a specific ALSA card — speaks WAV only. Without this
    conversion, Confidential Mode's private/speaker routing was impossible for
    the default (gTTS) voice unless mpg123 happened to be installed.
    """
    try:
        import io

        import soundfile as sf

        samples, rate = sf.read(io.BytesIO(data), dtype="int16")
        out = io.BytesIO()
        sf.write(out, samples, rate, format="WAV", subtype="PCM_16")
        return out.getvalue()
    except Exception as e:
        logger.debug(f"In-memory MP3 decode unavailable: {e}")
        return b""


def _routed_commands(path: str, fmt: str, device: str) -> list:
    """Playback commands that genuinely honour `device`, best first."""
    commands = []
    aplay = shutil.which("aplay")
    ffmpeg = shutil.which("ffmpeg")

    if fmt == "wav" and aplay:
        commands.append([aplay, "-q", "-D", device, path])
    if fmt == "mp3":
        for name in ("mpg123", "mpg321"):
            binary = shutil.which(name)
            if binary:
                commands.append([binary, "-q", "-a", device, path])
                break
        if ffmpeg:
            # ffmpeg can write straight to an ALSA device as an output sink.
            commands.append([ffmpeg, "-loglevel", "quiet", "-i", path,
                             "-f", "alsa", device])
    return commands


def switch_output_device(device_name: str) -> bool:
    """Route audio to a named output device (ALSA name on the Pi).

    Records the requested device; _play_routed() applies it per-utterance with
    a player that can actually target an ALSA card.
    """
    global _current_device
    if device_name == _current_device:
        return True

    # Selecting the device is just bookkeeping now: _play_routed() applies it
    # per-utterance with a player that can actually target an ALSA card.
    #
    # This used to tear down and re-init pygame's mixer with
    # devicename=<ALSA name>. That could never work — SDL enumerates outputs by
    # friendly name ("USB Audio Device Analog Stereo"), not by ALSA name — so
    # every switch threw, re-inited the default device, and returned False
    # while still claiming the device had been remembered. The mixer churn also
    # risked leaving playback broken for unrelated speech.
    _current_device = device_name
    logger.info(f"Audio output device set to {device_name}")
    return True


def use_speaker_output() -> bool:
    return switch_output_device(SPEAKER_DEVICE)


def use_bone_conduction_output() -> bool:
    return switch_output_device(BONE_DEVICE)


class _Utterance:
    """One queued speech request, with a per-item completion event.

    A per-item event lets speak(block=True) wait for *this* utterance only.
    Queue.join() waited for the entire backlog, so a blocking prompt could
    hang behind unrelated queued speech.
    """

    __slots__ = ("text", "lang", "done")

    def __init__(self, text, lang):
        self.text = text
        self.lang = lang
        self.done = threading.Event()


class TTSManager:
    def __init__(self):
        self.settings = self._load_settings()
        self.queue = Queue()
        self.running = True
        self._current_rate = self.settings.get("tts_rate", 150)
        self._speaking = threading.Event()
        self._engine_lock = threading.Lock()
        self._proc = None            # active CLI player process (for stop())
        self._proc_lock = threading.Lock()
        self._gtts_failed_at = 0.0   # backoff so every line doesn't retry a dead network

        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._current_rate)
            self._engine.setProperty("volume", self.settings.get("tts_volume", 1.0))
            self._engine_available = True
        except Exception as e:
            logger.warning(f"pyttsx3 init failed: {e}. Offline speech unavailable.")
            self._engine = None
            self._engine_available = False

        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        logger.info("TTS Manager ready.")

    def _load_settings(self) -> dict:
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {"tts_rate": 150, "tts_volume": 1.0, "tts_engine": "gtts"}

    # ── SYNTHESIS ────────────────────────────────────────────
    def _synthesize(self, text: str, lang: str):
        """Return (audio_bytes, format) — format is 'mp3' or 'wav'. ("", None) on failure."""
        engine_choice = self.settings.get("tts_engine", "pyttsx3")

        # gTTS produces MP3. Skip it for 30s after a failure so an offline
        # device doesn't pay a network timeout on every single line it speaks.
        if engine_choice == "gtts" and (time.time() - self._gtts_failed_at) > 30:
            try:
                import io
                from gtts import gTTS

                lang_map = {"eng": "en", "hin": "hi", "guj": "gu",
                            "en": "en", "hi": "hi", "gu": "gu"}
                g_lang = lang_map.get(lang.lower(), "en")
                tld = "co.in" if g_lang == "en" else "com"

                fp = io.BytesIO()
                gTTS(text=text, lang=g_lang, tld=tld).write_to_fp(fp)
                fp.seek(0)
                data = fp.read()
                if data:
                    return data, "mp3"
            except Exception as e:
                self._gtts_failed_at = time.time()
                logger.warning(f"gTTS unavailable ({e}); using offline engine for 30s.")

        # Offline pyttsx3 → WAV on disk.
        if not self._engine_available:
            return b"", None

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            with self._engine_lock:
                self._engine.setProperty("rate", self._current_rate)
                self._engine.save_to_file(text, tmp_path)
                self._engine.runAndWait()
            with open(tmp_path, "rb") as f:
                data = f.read()
            return (data, "wav") if data else (b"", None)
        except Exception as e:
            logger.error(f"pyttsx3 synthesis failed: {e}")
            return b"", None
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # ── PLAYBACK CHAIN ───────────────────────────────────────
    def _play_pygame(self, data: bytes, fmt: str) -> bool:
        if not PYGAME_AVAILABLE:
            return False
        try:
            import io
            if fmt == "mp3":
                # mixer.Sound cannot decode MP3 from a buffer — that is what
                # made gTTS output unplayable. MP3 must go through mixer.music.
                pygame.mixer.music.load(io.BytesIO(data))
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and self.running:
                    pygame.time.wait(10)
            else:
                sound = pygame.mixer.Sound(io.BytesIO(data))
                sound.play()
                while pygame.mixer.get_busy() and self.running:
                    pygame.time.wait(10)
            return True
        except Exception as e:
            logger.warning(f"pygame playback failed ({e}); trying system player.")
            return False

    def _play_system(self, data: bytes, fmt: str) -> bool:
        binary, build_args = _find_player(fmt)
        if not binary:
            return False

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            device = None if (IS_MACOS or _current_device in (None, "default")) else _current_device
            proc = subprocess.Popen(build_args(tmp_path, device),
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with self._proc_lock:
                self._proc = proc
            proc.wait(timeout=120)
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            logger.error("System audio player timed out.")
            return False
        except Exception as e:
            logger.warning(f"System player failed ({e}).")
            return False
        finally:
            with self._proc_lock:
                self._proc = None
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _play_routed(self, data: bytes, fmt: str, device: str) -> bool:
        """Play `data` on a specific ALSA card, or return False.

        This exists because pygame cannot do it. SDL enumerates outputs by
        friendly name ("USB Audio Device Analog Stereo"), not by ALSA name, so
        `pygame.mixer.init(devicename="plughw:2,0")` never matches a device and
        routing silently fell back to the default output. For Confidential Mode
        that is a privacy failure, not a cosmetic one: "private" speech went to
        whichever card ALSA defaulted to. If none of these commands work we
        return False so the caller can decide, rather than playing the text out
        of the wrong speaker.
        """
        payload, play_fmt = data, fmt
        if fmt == "mp3" and not shutil.which("mpg123") and not shutil.which("mpg321"):
            decoded = _decode_mp3_to_wav(data)
            if decoded:
                payload, play_fmt = decoded, "wav"

        commands = _routed_commands("<path>", play_fmt, device)
        if not commands:
            logger.warning(
                f"No player can route {play_fmt} to {device}. Install mpg123 "
                "(`sudo apt install mpg123`) or python soundfile for routed audio.")
            return False

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{play_fmt}", delete=False) as tmp:
                tmp.write(payload)
                tmp_path = tmp.name

            for template in _routed_commands(tmp_path, play_fmt, device):
                try:
                    proc = subprocess.Popen(template, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL)
                    with self._proc_lock:
                        self._proc = proc
                    proc.wait(timeout=120)
                    if proc.returncode == 0:
                        return True
                    logger.debug(f"Routed playback failed: {' '.join(template[:2])}")
                except subprocess.TimeoutExpired:
                    logger.error("Routed audio player timed out.")
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    return False
                except Exception as e:
                    logger.debug(f"Routed player error: {e}")
            return False
        finally:
            with self._proc_lock:
                self._proc = None
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _play_pyttsx3_direct(self, text: str) -> bool:
        """Last audible resort: speak straight out of pyttsx3, no file involved."""
        if not self._engine_available:
            return False
        try:
            with self._engine_lock:
                self._engine.setProperty("rate", self._current_rate)
                self._engine.say(text)
                self._engine.runAndWait()
            return True
        except Exception as e:
            logger.error(f"Direct pyttsx3 playback failed: {e}")
            return False

    def _deliver(self, text: str, lang: str):
        """Speak `text`, trying every backend. Never silently drops the message."""
        self._speaking.set()
        try:
            data, fmt = self._synthesize(text, lang)

            if data and fmt:
                # When a specific card has been selected (Confidential Mode),
                # routing is a correctness requirement — try the device-aware
                # players first and only fall back to the default output if
                # none of them can drive that card at all.
                device = None if IS_MACOS else _current_device
                if device not in (None, "default"):
                    if self._play_routed(data, fmt, device):
                        return
                    logger.error(
                        f"Could not play audio on {device}; falling back to the "
                        "default output. Confidential routing is NOT in effect.")

                if self._play_pygame(data, fmt):
                    return
                if self._play_system(data, fmt):
                    return
                logger.warning("All file-based playback failed; using direct engine.")

            if self._play_pyttsx3_direct(text):
                return

            # Nothing could speak. Make the failure loud rather than silent —
            # a blind user must never be left guessing whether the device died.
            logger.error(f"NO AUDIO BACKEND AVAILABLE — could not speak: {text[:80]!r}")
            print(f"[TTS — NO AUDIO OUTPUT] {text}", flush=True)
        finally:
            self._speaking.clear()

    # ── WORKER ───────────────────────────────────────────────
    def _process_queue(self):
        while self.running:
            try:
                item = self.queue.get(timeout=0.25)
            except Empty:
                continue

            if item is None:          # shutdown sentinel
                self.queue.task_done()
                break

            try:
                logger.info(f'Speaking: "{item.text[:60]}"')
                self._deliver(item.text, item.lang)
            except Exception as e:
                logger.error(f"TTS error: {e}")
                print(f"[TTS FALLBACK] {item.text}", flush=True)
            finally:
                item.done.set()
                self.queue.task_done()

    # ── PUBLIC API ───────────────────────────────────────────
    def speak(self, text: str, lang: str = "eng", block: bool = False, timeout: float = 120.0):
        if not text or not str(text).strip():
            return
        if not self.running:
            print(f"[TTS after shutdown] {text}", flush=True)
            return

        item = _Utterance(str(text), lang)
        self.queue.put(item)
        if block:
            item.done.wait(timeout=timeout)

    def set_rate(self, rate: int):
        self._current_rate = max(80, min(300, int(rate)))

    def get_rate(self) -> int:
        return self._current_rate

    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def flush(self):
        """Drop everything still queued (already-released waiters are set).

        Needed so a long answer isn't preceded by stale "still working…"
        reassurance messages that were queued while the AI was thinking.
        """
        dropped = 0
        while True:
            try:
                item = self.queue.get_nowait()
            except Empty:
                break
            if item is not None:
                item.done.set()
                dropped += 1
            self.queue.task_done()
        if dropped:
            logger.info(f"Flushed {dropped} pending utterance(s).")

    def stop(self):
        """Stop what is playing now and discard anything pending."""
        self.flush()
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.stop()
                pygame.mixer.music.stop()
            except Exception:
                pass
        with self._proc_lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
        self._speaking.clear()

    def shutdown(self):
        if not self.running:
            return
        self.running = False
        self.queue.put(None)
        self.worker_thread.join(timeout=5)
        with self._proc_lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
        logger.info("TTS Manager shut down.")


_tts_manager = None
_tts_lock = threading.Lock()


def _get_manager() -> TTSManager:
    global _tts_manager
    if _tts_manager is None:
        with _tts_lock:
            if _tts_manager is None:
                _tts_manager = TTSManager()
    return _tts_manager


def speak(text: str, lang: str = "eng", block: bool = False):
    _get_manager().speak(text, lang, block)


def set_rate(rate: int):
    _get_manager().set_rate(rate)


def get_rate() -> int:
    return _get_manager().get_rate()


def is_speaking() -> bool:
    return _get_manager().is_speaking()


def stop():
    _get_manager().stop()


def flush():
    _get_manager().flush()


def wait_until_idle(timeout: float = 30.0):
    """Block until queued speech has finished playing (used before shutdown)."""
    deadline = time.time() + timeout
    manager = _get_manager()
    while time.time() < deadline:
        if manager.queue.empty() and not manager.is_speaking():
            return True
        time.sleep(0.05)
    return False


def shutdown():
    """Module-level shutdown. main.py calls this on exit."""
    if _tts_manager is not None:
        _tts_manager.shutdown()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: (shutdown(), sys.exit(0)))
    print("TTS Test — type text, QUIT to exit")
    print(f"  pygame:  {PYGAME_AVAILABLE}")
    print(f"  engine:  {_get_manager().settings.get('tts_engine')}")
    player, _ = _find_player("mp3")
    print(f"  player:  {player or 'none'}")
    while True:
        try:
            inp = input("Text: ").strip()
            if inp.upper() == "QUIT":
                break
            if inp:
                speak(inp, block=True)
        except (EOFError, KeyboardInterrupt):
            break
    shutdown()
