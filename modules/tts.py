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
    use_speaker_output() / use_earphone_output() / switch_output_device(name)
"""

import json
import logging
import os
import platform
import re
import shutil
import signal
import subprocess
from modules.privacy_policy import current_policy, processing_session, PRIVATE
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
    from modules.config_loader import load_settings
    return load_settings()


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


def _validate_device(configured: str, role: str, fallback_index: int,
                     avoid: str = None) -> str:
    """Return `configured` if the hardware has it, else the best substitute.

    `avoid` names a card the substitute must not land on. It exists for the
    earphone: with cards 2 and 3 present and the speaker on 3, a positional
    fallback picks index 1 — the speaker — so a mistyped earphone_device made
    private speech come out of the loudspeaker while still reporting success.
    Preferring any other real output keeps the private/public split intact.
    """
    if IS_MACOS or not configured or configured == "default":
        return configured
    if not _AVAILABLE_OUTPUTS:
        return configured                 # cannot verify; trust the config
    if configured in _AVAILABLE_OUTPUTS:
        return configured

    if avoid is not None:
        logger.error(
            "Earphone output %r is unavailable (available: %s). Set "
            "earphone_device to the headset output in settings.json; "
            "private playback is disabled.", configured, ", ".join(_AVAILABLE_OUTPUTS))
        return None
    candidates = _AVAILABLE_OUTPUTS
    substitute = candidates[min(fallback_index, len(candidates) - 1)]
    logger.error(
        f"{role} device {configured!r} from settings.json does not exist on this "
        f"machine (available: {', '.join(_AVAILABLE_OUTPUTS)}). Falling back to "
        f"{substitute!r} — fix {role.lower()}_device in settings.json."
    )
    if avoid and substitute == avoid:
        logger.error(
            "The earphone and the speaker are now the same card: private "
            "speech WILL be audible on the speaker. Set earphone_device in "
            "settings.json to a second output card.")
    return substitute


SPEAKER_DEVICE = _validate_device(
    _static_settings.get("speaker_device", "plughw:3,0"), "Speaker", 0)

# The private output is a wired earphone. `bone_device` is the old name for
# this setting, from when the hardware was going to be a bone-conduction pad;
# there is no such unit on this device. It is still read so an existing
# settings.json keeps working, but earphone_device is the name to use.
_configured_earphone = (_static_settings.get("earphone_device")
                        or _static_settings.get("bone_device")
                        or "plughw:2,0")
if not _static_settings.get("earphone_device") and _static_settings.get("bone_device"):
    logger.info("settings.json uses the old key 'bone_device'; rename it to "
                "'earphone_device'. Reading it as the earphone for now.")

EARPHONE_DEVICE = _validate_device(_configured_earphone, "Earphone", 0,
                                   avoid=SPEAKER_DEVICE)


class AudioRoutingError(RuntimeError):
    pass


class SpeechCancelled(AudioRoutingError):
    """Playback deliberately stopped; another backend must not resume it."""


def require_private_output(device=None):
    selected = device or EARPHONE_DEVICE
    available = _alsa_playback_devices()
    if (IS_MACOS or SPEAKER_DEVICE not in available or not selected or selected == "default"
            or selected == SPEAKER_DEVICE or selected not in available):
        raise AudioRoutingError("A separate, verified private audio output is required.")
    return selected

# Kept so any caller still using the old name gets the same card.
BONE_DEVICE = EARPHONE_DEVICE

# ── OPTIONAL BACKENDS ────────────────────────────────────────
PYGAME_AVAILABLE = False
_current_device = None
try:
    import pygame

    # Importing TTS must not reserve the default ALSA card. Routed playback
    # uses a separate process, which cannot open a card held by SDL's mixer.
    # Open the mixer only for pygame playback and close it when that ends.
    PYGAME_AVAILABLE = True
    _current_device = "default"
    logger.info("pygame available; mixer will open only during default playback.")
except ImportError:
    logger.info("pygame not installed — falling back to system audio players.")
except Exception as e:
    logger.warning(f"pygame import failed: {e}")


def _release_mixer():
    """Release SDL's audio device before another player needs the hardware."""
    if PYGAME_AVAILABLE and pygame.mixer.get_init():
        pygame.mixer.quit()


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

    Several decoders are tried because the obvious one is not dependable here.
    `soundfile` is in requirements.txt, but it only gained MP3 support in
    libsndfile 1.1.0, and on Raspberry Pi OS the wheel links against the
    *system* libsndfile — 1.0.31 on Bullseye — which rejects MP3 outright. A
    device that had soundfile installed therefore still could not route the
    gTTS voice to a chosen card, and the failure surfaced only as silence.
    """
    errors = []

    # 1. soundfile — works where libsndfile is 1.1.0+ (Bookworm and later).
    try:
        import io

        import soundfile as sf

        samples, rate = sf.read(io.BytesIO(data), dtype="int16")
        out = io.BytesIO()
        sf.write(out, samples, rate, format="WAV", subtype="PCM_16")
        return out.getvalue()
    except Exception as e:
        errors.append(f"soundfile: {e}")

    # 2. Any MP3-capable CLI decoder, writing a temp WAV. mpg123 and ffmpeg are
    #    checked here as well as in _routed_commands: this path is what lets an
    #    old-libsndfile device still reach `aplay -D`, which is the only
    #    card-targeting player guaranteed to exist on Raspberry Pi OS.
    decoders = []
    for name in ("mpg123", "mpg321"):
        binary = shutil.which(name)
        if binary:
            decoders.append(lambda src, dst, b=binary: [b, "-q", "-w", dst, src])
            break
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        decoders.append(lambda src, dst: [ffmpeg, "-loglevel", "quiet", "-y",
                                          "-i", src, "-f", "wav", dst])

    for build in decoders:
        src = dst = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(data)
                src = tmp.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                dst = tmp.name
            subprocess.run(build(src, dst), timeout=60,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.getsize(dst) > 44:
                with open(dst, "rb") as f:
                    return f.read()
        except Exception as e:
            errors.append(f"cli: {e}")
        finally:
            for path in (src, dst):
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    logger.debug(f"In-memory MP3 decode unavailable ({'; '.join(errors) or 'no decoder'}).")
    return b""


# espeak-ng ships real Hindi and Gujarati voices. They are the only offline
# way to say a translation out loud, so the language has to reach this far
# down: the default English voice renders Devanagari and Gujarati script as
# silence or as spelled-out nonsense.
_ESPEAK_VOICES = {"eng": "en", "en": "en",
                  "hin": "hi", "hi": "hi",
                  "guj": "gu", "gu": "gu"}


def _espeak_to_wav(text: str, rate: int, lang: str = "eng") -> bytes:
    """espeak → WAV bytes, or b"". The floor of the routed-playback chain.

    pyttsx3 can be installed but unconfigured (no driver, no voice), so it is
    not a dependable last resort on its own. Shelling out to espeak is: it is
    present on every Pi image this runs on, and it is exactly what used to
    deliver routed speech here before the natural voice was wired up. This
    keeps the worst case equal to the old behaviour — an audible answer on the
    correct card — instead of silence.
    """
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if not espeak:
        return b""

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        voice = _ESPEAK_VOICES.get(str(lang).lower(), "en")
        subprocess.run([espeak, "-v", voice,
                        "-s", str(int(rate)), "-a", "175", "-g", "4",
                        "-w", tmp_path, text], timeout=60,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.getsize(tmp_path) > 44:
            with open(tmp_path, "rb") as f:
                return f.read()
    except Exception as e:
        logger.debug(f"espeak WAV synthesis failed: {e}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return b""


# ALSA ring-buffer sizing for the routed players. The defaults are a few tens
# of milliseconds, which underrun — audibly, as crackling and clicks — whenever
# the Pi is busy synthesising, inferring or driving the camera at the same time
# as it plays. Half a second of buffer costs nothing perceptible for speech and
# makes playback immune to ordinary CPU spikes.
_ALSA_BUFFER_US = "500000"     # 500 ms ring buffer
_ALSA_PERIOD_US = "100000"     # 100 ms per period


def _aplay_args(aplay: str, device: str, path: str) -> list:
    return [aplay, "-q", "-D", device,
            "--buffer-time", _ALSA_BUFFER_US,
            "--period-time", _ALSA_PERIOD_US, path]


def _routed_commands(path: str, fmt: str, device: str) -> list:
    """Playback commands that genuinely honour `device`, best first."""
    commands = []
    aplay = shutil.which("aplay")
    ffmpeg = shutil.which("ffmpeg")

    if fmt == "wav" and aplay:
        commands.append(_aplay_args(aplay, device, path))
        # Same player without the buffer tuning, in case an exotic card
        # rejects the explicit timings. Never leave the user with no audio.
        commands.append([aplay, "-q", "-D", device, path])
    if fmt == "mp3":
        for name in ("mpg123", "mpg321"):
            binary = shutil.which(name)
            if binary:
                # -b prebuffers 1 MB of decoded audio so a stalled decoder
                # cannot starve the sound card mid-sentence.
                commands.append([binary, "-q", "-b", "1024", "-a", device, path])
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


def use_earphone_output() -> bool:
    return switch_output_device(EARPHONE_DEVICE)


def use_bone_conduction_output() -> bool:
    """Deprecated name for use_earphone_output(); the hardware is an earphone."""
    return use_earphone_output()


class _Utterance:
    """One queued speech request, with a per-item completion event.

    A per-item event lets speak(block=True) wait for *this* utterance only.
    Queue.join() waited for the entire backlog, so a blocking prompt could
    hang behind unrelated queued speech.
    """

    __slots__ = ("text", "lang", "done", "device", "policy", "error")

    def __init__(self, text, lang, device=None):
        self.text = text
        self.lang = lang
        # The route is captured when the utterance is queued. Looking up the
        # global route later, when the worker finally starts playback, allows
        # a mode transition to redirect an earlier message to the next mode's
        # output. A value pins *this* utterance to one card without disturbing
        # the global selection.
        self.device = _current_device if device is None else device
        self.done = threading.Event()
        self.policy = current_policy()
        self.error = None


def _pick_pyttsx3_voice(engine, lang: str):
    """Voice id on `engine` that can speak `lang`, or None.

    pyttsx3 exposes no language API, only a list of driver-specific voices, so
    this matches on the language tags and then on the id/name text. Returns
    None for English, where the default voice is already correct.
    """
    want = _ESPEAK_VOICES.get(str(lang).lower(), "en")
    if want == "en":
        return None
    try:
        for voice in engine.getProperty("voices"):
            tags = [str(t).lower() for t in (getattr(voice, "languages", None) or [])]
            if any(want in t for t in tags):
                return voice.id
            ident = f"{getattr(voice, 'id', '')} {getattr(voice, 'name', '')}".lower()
            # Match a whole tag, so "gu" never matches "gujarati"'s neighbours
            # like "guarani" — or, worse, the "gu" inside some unrelated path.
            if any(part == want for part in re.split(r"[^a-z]+", ident)):
                return voice.id
    except Exception as e:
        logger.debug(f"Could not enumerate pyttsx3 voices: {e}")
    return None


class TTSManager:
    def __init__(self):
        self.settings = self._load_settings()
        self.queue = Queue(maxsize=32)
        self._generation = 0
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
        from modules.config_loader import load_settings
        return load_settings()

    def _check_cancelled(self, generation):
        if (generation != getattr(self, "_generation", 0)
                or not getattr(self, "running", True)):
            raise SpeechCancelled("Speech cancelled.")

    # ── SYNTHESIS ────────────────────────────────────────────
    def _prepare_audio(self, data: bytes, fmt: str):
        """Apply speech loudness and online voice tempo before any player."""
        settings = self.settings
        gain = float(settings.get("tts_gain", 1.0))
        normalize = settings.get("tts_normalize_enabled", False)
        # Offline synthesis already uses tts_rate. Google returns fixed-rate
        # MP3, so its tempo needs a separate, pitch-preserving adjustment.
        speed = float(settings.get("tts_playback_speed", 1.0)) if fmt == "mp3" else 1.0
        if gain == 1.0 and not normalize and speed == 1.0:
            return data, fmt

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            filters = []
            if speed != 1.0:
                filters.append(f"atempo={speed}")
            if normalize:
                filters.append("loudnorm=I=-18:TP=-2:LRA=11")
            filters.extend((f"volume={gain}", "alimiter=limit=0.95:level=false"))
            try:
                result = subprocess.run(
                    [ffmpeg, "-hide_banner", "-loglevel", "error",
                     "-i", "pipe:0", "-af", ",".join(filters),
                     "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le",
                     "-f", "wav", "pipe:1"],
                    input=data, capture_output=True, timeout=30)
                if result.returncode == 0 and len(result.stdout) > 44:
                    return result.stdout, "wav"
                logger.warning("Speech audio adjustment failed; using PCM loudness adjustment.")
            except Exception as e:
                logger.warning("Speech audio adjustment unavailable (%s).", type(e).__name__)

        # Keep volume adjustment available on images without ffmpeg.
        # Decode through the existing soundfile/mpg123 chain when necessary.
        if speed != 1.0:
            logger.warning("Install ffmpeg to apply tts_playback_speed to the Google voice.")
        wav = _decode_mp3_to_wav(data) if fmt == "mp3" else data
        if not wav:
            return data, fmt
        try:
            import io
            import wave
            import numpy as np

            with wave.open(io.BytesIO(wav), "rb") as source:
                params = source.getparams()
                if source.getsampwidth() != 2:
                    return data, fmt
                samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float64)
            if samples.size:
                peak = float(np.max(np.abs(samples)))
                scale = gain
                if normalize:
                    rms = float(np.sqrt(np.mean(samples * samples)))
                    if rms > 0:
                        scale *= min(8.0, (32768 * 10 ** (-18 / 20)) / rms)
                if peak > 0:
                    scale = min(scale, 0.95 * 32767 / peak)
                samples *= scale
            out = io.BytesIO()
            with wave.open(out, "wb") as target:
                target.setparams(params)
                target.writeframes(samples.astype("<i2").tobytes())
            return out.getvalue(), "wav"
        except Exception as e:
            logger.warning("PCM loudness adjustment unavailable (%s).", type(e).__name__)
            return data, fmt

    def _synthesize(self, text: str, lang: str):
        """Return (audio_bytes, format) — format is 'mp3' or 'wav'. ("", None) on failure."""
        engine_choice = self.settings.get("tts_engine", "pyttsx3")

        # gTTS produces MP3. Skip it for 30s after a failure so an offline
        # device doesn't pay a network timeout on every single line it speaks.
        if current_policy().allow_cloud and engine_choice == "gtts" and (time.time() - self._gtts_failed_at) > 30:
            try:
                import io
                from gtts import gTTS

                lang_map = {"eng": "en", "hin": "hi", "guj": "gu",
                            "en": "en", "hi": "hi", "gu": "gu"}
                g_lang = lang_map.get(lang.lower(), "en")
                tld = "co.in" if g_lang == "en" else "com"

                fp = io.BytesIO()
                gTTS(text=text, lang=g_lang, tld=tld, timeout=(3, 8)).write_to_fp(fp)
                fp.seek(0)
                data = fp.read()
                if data:
                    return self._prepare_audio(data, "mp3")
            except Exception as e:
                self._gtts_failed_at = time.time()
                logger.warning(f"gTTS unavailable ({e}); using offline engine for 30s.")

        data = self._synthesize_offline_wav(text, lang)
        return self._prepare_audio(data, "wav") if data else (b"", None)

    def _select_voice(self, lang: str):
        """Switch the shared engine to `lang`. Returns (chosen, previous).

        Call under _engine_lock. The previous voice is read here rather than
        cached at init because some drivers report it as "" (a real value
        meaning "the default"), and a cached falsy id made the restore below
        silently do nothing — leaving every later English prompt in a Hindi
        voice.
        """
        prev_voice = None
        voice_id = _pick_pyttsx3_voice(self._engine, lang)
        if voice_id:
            try:
                prev_voice = self._engine.getProperty("voice")
            except Exception:
                prev_voice = None
            try:
                self._engine.setProperty("voice", voice_id)
            except Exception as e:
                logger.debug(f"Could not select {lang} voice: {e}")
                return None, None
        return voice_id, prev_voice

    def _restore_voice(self, voice_id, prev_voice) -> None:
        """Put the shared engine back, so the next utterance is unaffected."""
        if voice_id and prev_voice is not None:
            try:
                self._engine.setProperty("voice", prev_voice)
            except Exception as e:
                logger.debug(f"Could not restore default voice: {e}")

    def _synthesize_offline_wav(self, text: str, lang: str = "eng") -> bytes:
        """pyttsx3 → WAV bytes, or b"". Split out so routed playback can fall
        back to a format `aplay -D` can definitely handle."""
        # For Hindi and Gujarati, go to espeak-ng first. pyttsx3's default
        # voice is whatever the driver picked at init — English on every image
        # this runs on — and handing it Devanagari produces silence. espeak-ng
        # has the actual voices, so an offline device can still speak a
        # translation instead of just displaying one.
        if _ESPEAK_VOICES.get(str(lang).lower(), "en") != "en":
            data = _espeak_to_wav(text, self._current_rate, lang)
            if data:
                return data

        if not self._engine_available:
            return b""

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            with self._engine_lock:
                self._engine.setProperty("rate", self._current_rate)
                voice_id, prev_voice = self._select_voice(lang)
                try:
                    self._engine.save_to_file(text, tmp_path)
                    self._engine.runAndWait()
                finally:
                    self._restore_voice(voice_id, prev_voice)
            with open(tmp_path, "rb") as f:
                data = f.read()
            return data
        except Exception as e:
            logger.error(f"pyttsx3 synthesis failed: {e}")
            return b""
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
            if not pygame.mixer.get_init():
                pygame.mixer.init(
                    frequency=24000, size=-16, channels=1,
                    buffer=int(_static_settings.get("tts_mixer_buffer", 1024)))
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
        finally:
            _release_mixer()

    def _play_system(self, data: bytes, fmt: str) -> bool:
        generation = getattr(self, "_generation", 0)
        self._check_cancelled(generation)
        _release_mixer()
        binary, build_args = _find_player(fmt)
        if not binary:
            return False

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            device = None if (IS_MACOS or _current_device in (None, "default")) else _current_device
            with self._proc_lock:
                self._check_cancelled(generation)
                proc = subprocess.Popen(build_args(tmp_path, device),
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self._proc = proc
            proc.wait(timeout=120)
            self._check_cancelled(generation)
            return proc.returncode == 0
        except SpeechCancelled:
            raise
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
        generation = getattr(self, "_generation", 0)
        self._check_cancelled(generation)
        _release_mixer()
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
                    with self._proc_lock:
                        self._check_cancelled(generation)
                        proc = subprocess.Popen(template, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
                        self._proc = proc
                    stdout, stderr = proc.communicate(timeout=120)
                    self._check_cancelled(generation)
                    if proc.returncode == 0:
                        return True
                    logger.error(
                       f"Routed playback failed (exit {proc.returncode}): "
                       f"{' '.join(template)}\nSTDOUT: {stdout}\nSTDERR: {stderr}"
                )
                except SpeechCancelled:
                    raise
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

    def _play_pyttsx3_direct(self, text: str, lang: str = "eng") -> bool:
        """Last audible resort: speak straight out of pyttsx3, no file involved."""
        if not self._engine_available:
            return False
        _release_mixer()
        try:
            with self._engine_lock:
                self._engine.setProperty("rate", self._current_rate)
                voice_id, prev_voice = self._select_voice(lang)
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                finally:
                    self._restore_voice(voice_id, prev_voice)
            return True
        except Exception as e:
            logger.error(f"Direct pyttsx3 playback failed: {e}")
            return False

    def _deliver(self, text: str, lang: str, device_override: str = None):
        """Speak `text`, trying every backend. Never silently drops the message."""
        self._speaking.set()
        try:
            generation = getattr(self, "_generation", 0)
            self._check_cancelled(generation)
            private = current_policy().private or (
                bool(device_override) and device_override == EARPHONE_DEVICE)
            if private:
                device = require_private_output(device_override)
                # Private text must never be sent to a network synthesizer.
                with processing_session(PRIVATE):
                    data, fmt = self._synthesize(text, lang)
                self._check_cancelled(generation)
                if data and fmt and self._play_routed(data, fmt, device):
                    return True
                self._check_cancelled(generation)
                raise AudioRoutingError("Private playback failed; content was not sent to a public output.")
            data, fmt = self._synthesize(text, lang)
            self._check_cancelled(generation)

            if data and fmt:
                # When a specific card has been selected (Confidential Mode),
                # routing is a correctness requirement — try the device-aware
                # players first and only fall back to the default output if
                # none of them can drive that card at all.
                device = None if IS_MACOS else (device_override or _current_device)
                if device not in (None, "default"):
                    if self._play_routed(data, fmt, device):
                        return
                    self._check_cancelled(generation)

                    # The gTTS voice is MP3, and on a Pi with no mpg123/ffmpeg
                    # and a pre-1.1 libsndfile there is nothing that can decode
                    # it — so nothing that can put it on a named card. Rather
                    # than abandon the card (silence on a headless device whose
                    # default sink is dead HDMI, and a privacy breach when the
                    # text was meant for the earphone), re-say it in the
                    # offline voice, which produces WAV that `aplay -D` always
                    # handles. Worse voice, right device, still audible.
                    if fmt == "mp3":
                        # Lazily: espeak must not run when pyttsx3 already
                        # produced usable audio — the user is waiting on this.
                        for synth in (lambda: self._synthesize_offline_wav(text, lang),
                                      lambda: _espeak_to_wav(text, self._current_rate, lang)):
                            self._check_cancelled(generation)
                            wav = synth()
                            if wav:
                                wav, _ = self._prepare_audio(wav, "wav")
                            self._check_cancelled(generation)
                            if wav and self._play_routed(wav, "wav", device):
                                logger.warning(
                                    f"Played {device} in the offline voice: no "
                                    "MP3 decoder here. `sudo apt install "
                                    "mpg123` restores the natural gTTS voice "
                                    "on this card.")
                                return

                    self._check_cancelled(generation)
                    logger.error(
                        f"Could not play audio on {device}; falling back to the "
                        "default output. Confidential routing is NOT in effect.")

                # Which backend actually produced sound is the single most
                # useful fact when the device is silent, and it used to be
                # logged only on failure — so a silent-but-"successful" path
                # was invisible.
                self._check_cancelled(generation)
                if self._play_pygame(data, fmt):
                    logger.debug(f"Played {fmt} via pygame (default output).")
                    return
                self._check_cancelled(generation)
                if self._play_system(data, fmt):
                    logger.debug(f"Played {fmt} via system player (default output).")
                    return
                logger.warning("All file-based playback failed; using direct engine.")

            self._check_cancelled(generation)
            if self._play_pyttsx3_direct(text, lang):
                logger.debug("Played via pyttsx3 direct.")
                return

            # Nothing could speak. Make the failure loud rather than silent —
            # a blind user must never be left guessing whether the device died.
            logger.error("No audio backend available.")
            raise AudioRoutingError("No audio output is available.")
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
                logger.info("Speaking utterance (%s characters).", len(item.text))
                with processing_session(item.policy):
                    self._deliver(item.text, item.lang, item.device)
            except SpeechCancelled as e:
                item.error = e
                logger.info("Speech playback cancelled.")
            except Exception as e:
                item.error = AudioRoutingError("Speech playback failed.")
                logger.error("Speech playback failed (%s).", type(e).__name__)
            finally:
                item.done.set()
                self.queue.task_done()

    # ── PUBLIC API ───────────────────────────────────────────
    def speak(self, text: str, lang: str = "eng", block: bool = False,
              timeout: float = 120.0, device: str = None):
        if not text or not str(text).strip():
            return
        if not self.running:
            raise AudioRoutingError("Speech engine has shut down.")

        # Capture the selected output at enqueue time. This is important for
        # non-blocking speech: callers commonly queue a status message and
        # immediately switch between speaker and earphone for the next step.
        # Without this snapshot, the worker can play that old message on the
        # newly selected route.
        queued_device = _current_device if device is None else device
        if current_policy().private:
            queued_device = require_private_output()
        item = _Utterance(str(text), lang, queued_device)
        self.queue.put(item, timeout=1)
        if block:
            if not item.done.wait(timeout=timeout):
                self.stop()
                raise TimeoutError("Speech playback timed out.")
            if item.error:
                raise item.error
        return True

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
                item.error = AudioRoutingError("Speech cancelled.")
                item.done.set()
                dropped += 1
            self.queue.task_done()
        if dropped:
            logger.info(f"Flushed {dropped} pending utterance(s).")

    def stop(self):
        """Stop what is playing now and discard anything pending."""
        # Serialize cancellation with launching/registering a CLI player so
        # stop cannot miss a process created between Popen and registration.
        with self._proc_lock:
            self._generation = getattr(self, "_generation", 0) + 1
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
        self.flush()
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.stop()
                pygame.mixer.music.stop()
            except Exception:
                pass
        self._speaking.clear()

    def shutdown(self):
        if not self.running:
            return
        self.stop()
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
    return _get_manager().speak(text, lang, block)


def speak_on_device(text: str, device: str, lang: str = "eng",
                    block: bool = True, timeout: float = 300.0):
    """Speak `text` on one named ALSA card, using the normal synthesis chain.

    This is the good voice (gTTS, falling back to pyttsx3) delivered to a
    chosen card. Callers that need routing — Confidential Mode's
    earphone/speaker split, and the AI answer — used to synthesise with espeak
    themselves purely because espeak-to-WAV plus `aplay -D` was the only thing
    they knew could target a card. That made the AI's answer the one utterance
    on the device spoken by a formant synthesiser: buzzy and creaky, and
    obviously different from every other prompt the user hears.

    Routing here is per-utterance, so it never mutates the global device
    selection, and it goes through the same queue as ordinary speech, so a
    routed answer cannot overlap with a queued prompt.

    Defaults to block=True: every caller that routes speech has to know when
    the utterance finished (it is about to ask the user a question).
    """
    return _get_manager().speak(text, lang, block=block, timeout=timeout, device=device)

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
