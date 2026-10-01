"""
live_translator.py — BlindAssist Project
=========================================
Project  : Accessible Educational Terminal for Visually Impaired
Module   : Real-Time Live Conversation Translation (Gemini Live API)

Bidirectional real-time speech-to-speech interpreter for two speakers:
  - Speaker A speaks English  -> immediately spoken in Gujarati/Hindi
  - Speaker B speaks Gujarati/Hindi -> immediately spoken in English

Uses Google Gemini Live API with low-latency WebSocket streaming.
Compatible with:
  - Laptop simulation (MacBook Air / PC mic + speakers via sounddevice)
  - Raspberry Pi 5 (USB mic + speaker / earphones)
"""

import os
import sys
import json
import time
import queue
import signal
import select
import asyncio
import logging
import threading
from pathlib import Path
from typing import Optional, Callable

import numpy as np

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except Exception:
    sd = None
    SD_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    types = None
    GENAI_AVAILABLE = False

# ──────────────────────────────────────────────────────────────
# HARDWARE FLAGS (Pi Flag Pattern)
# ──────────────────────────────────────────────────────────────
HEADLESS = False
USE_PICAMERA = False
USE_GPIO = False
USE_CORAL = False

# ──────────────────────────────────────────────────────────────
# PATH CONFIGURATION & LOGGING
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "live_translator.log"
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("LiveTranslator")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    try:
        fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass


def _load_settings() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load settings from {CONFIG_PATH}: {e}")
        return {}


# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
SAMPLE_RATE_IN = 16000     # Gemini Live expects 16kHz mono 16-bit PCM
SAMPLE_RATE_OUT = 24000    # Gemini Live returns 24kHz mono 16-bit PCM
CHUNK_MS = 100             # 100ms streaming chunks
CHUNK_FRAMES = int(SAMPLE_RATE_IN * (CHUNK_MS / 1000.0))  # 1600 samples = 3200 bytes
# A fixed 8x gain clips ordinary USB/Mac microphones.  Clipped PCM is hard for
# Live API voice activity detection to recognise, so leave input untouched by
# default and allow a carefully chosen gain in settings.json when necessary.
DEFAULT_MIC_GAIN = 1.0

# `gemini-3.8-live` supports microphone audio plus a system instruction.  Do
# not replace this with `gemini-2.5-flash-native-audio-latest`: that model
# accepts the WebSocket connection but rejects audio when used with this
# interpreter configuration (the failure that previously made this mode look
# connected while doing no translation).
DEFAULT_LIVE_MODEL = "gemini-3.8-live"

# Never let a stalled/closed WebSocket accumulate unlimited microphone audio.
# Five seconds is enough to absorb short network jitter without adding a
# noticeable delay to a conversation.
MAX_PENDING_AUDIO_CHUNKS = 50


class LiveTranslator:
    """Real-time bi-directional speech-to-speech interpreter."""

    def __init__(self, lang_pair: str = "en-gu"):
        """
        lang_pair:
          'en-gu': English <-> Gujarati
          'en-hi': English <-> Hindi
          'hi-gu': Hindi <-> Gujarati
        """
        self.settings = _load_settings()
        self.api_key = self.settings.get("gemini_api_key", "").strip()
        self.model = str(
            self.settings.get("live_translation_model", DEFAULT_LIVE_MODEL)
            or DEFAULT_LIVE_MODEL
        ).strip()
        self.mic_gain = max(
            0.1, min(float(self.settings.get("live_mic_gain", DEFAULT_MIC_GAIN)), 4.0)
        )
        # `None` tells sounddevice to use the operating system defaults.  This
        # works on macOS and Windows; Pi deployments may set ALSA-compatible
        # sounddevice device names/indices here when their defaults differ.
        self.input_device = self.settings.get("live_input_device") or None
        self.output_device = self.settings.get("live_output_device") or None
        self.lang_pair = lang_pair
        self.running = False
        self._session_dead = False
        self.last_error: Optional[str] = None
        self._chunks_sent = 0
        self.audio_in_queue = asyncio.Queue(maxsize=MAX_PENDING_AUDIO_CHUNKS)
        self.playback_queue = queue.Queue()
        self._playback_thread = None
        self._stop_event = threading.Event()
        # Prevent the microphone from re-submitting Gemini's own speaker
        # output as a fresh user turn.  This is especially important on a
        # laptop where the mic and speakers are physically close together.
        self._model_speaking = threading.Event()
        self._turn_end_marker = object()

    def _get_system_prompt(self) -> str:
        if self.lang_pair == "en-gu":
            return (
                "You are an expert real-time live interpreter between English and Gujarati. "
                "You will listen to a live conversation between two people. "
                "RULE 1: When you hear English speech, translate it immediately into natural spoken Gujarati. "
                "RULE 2: When you hear Gujarati speech, translate it immediately into natural spoken English. "
                "RULE 3: Output ONLY the spoken translation. Do not add any conversational commentary, "
                "introductions, or pleasantries like 'Sure', 'Here is the translation'. "
                "Speak clearly and with appropriate tone and inflection."
            )
        elif self.lang_pair == "en-hi":
            return (
                "You are an expert real-time live interpreter between English and Hindi. "
                "You will listen to a live conversation between two people. "
                "RULE 1: When you hear English speech, translate it immediately into natural spoken Hindi. "
                "RULE 2: When you hear Hindi speech, translate it immediately into natural spoken English. "
                "RULE 3: Output ONLY the spoken translation. Do not add any conversational commentary. "
                "Speak clearly and with natural inflection."
            )
        elif self.lang_pair == "hi-gu":
            return (
                "You are an expert real-time live interpreter between Hindi and Gujarati. "
                "When you hear Hindi, speak the translation in Gujarati. "
                "When you hear Gujarati, speak the translation in Hindi. "
                "Output ONLY the spoken translation without conversational commentary."
            )
        else:
            return (
                "You are a real-time speech interpreter. "
                "Translate the spoken language to the other conversation language immediately. "
                "Output ONLY the spoken translation."
            )

    def _playback_worker(self):
        """Worker thread that plays audio response chunks smoothly on the speakers."""
        if not SD_AVAILABLE or sd is None:
            return

        out_stream = None
        try:
            out_stream = sd.OutputStream(
                samplerate=SAMPLE_RATE_OUT,
                channels=1,
                dtype="int16",
                device=self.output_device,
            )
            out_stream.start()
            logger.info("Live playback ready on %s.", self._audio_device_name(
                self.output_device, "output"))

            while not self._stop_event.is_set():
                try:
                    chunk = self.playback_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if chunk is None:  # Sentinel
                    break

                if chunk is self._turn_end_marker:
                    self._model_speaking.clear()
                    self.playback_queue.task_done()
                    continue

                if len(chunk) > 0 and not self._stop_event.is_set():
                    out_stream.write(chunk)
                self.playback_queue.task_done()
        except Exception as e:
            self.last_error = f"Speaker playback failed: {e}"
            logger.error(self.last_error)
            self._session_dead = True
            self.running = False
        finally:
            if out_stream is not None:
                try:
                    out_stream.stop()
                    out_stream.close()
                except Exception:
                    pass

    async def _send_audio_chunk(self, session, audio: bytes):
        """Send one PCM chunk on both current and older google-genai SDKs.

        `send_realtime_input(audio=...)` is the current public SDK API.
        The project originally used the older `session.send()` envelope; keep
        that fallback so existing Raspberry Pi installations do not fail until
        their dependencies are upgraded.
        """
        blob = types.Blob(
            data=audio,
            mime_type=f"audio/pcm;rate={SAMPLE_RATE_IN}",
        )
        send_realtime_input = getattr(session, "send_realtime_input", None)
        if callable(send_realtime_input):
            await send_realtime_input(audio=blob)
            return

        realtime_input = types.LiveClientRealtimeInput(media_chunks=[blob])
        await session.send(input=realtime_input)

    async def _send_audio_loop(self, session):
        """Continuously reads audio from the mic queue and sends it to the Gemini Live session."""
        while self.running and not self._session_dead:
            try:
                data = await self.audio_in_queue.get()
                if data is None:
                    break

                # Use clean PCM by default.  Excessive gain clips the waveform
                # and makes speech detection less reliable than quiet audio.
                samples = np.frombuffer(data, dtype=np.int16).astype(np.int32)
                samples = np.clip(samples * self.mic_gain, -32768, 32767).astype(np.int16)
                amplified = samples.tobytes()

                await self._send_audio_chunk(session, amplified)
                self._chunks_sent += 1
                self.audio_in_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log once and mark the session as dead so we don't spam
                # hundreds of identical error lines.
                self.last_error = f"Audio stream failed: {e}"
                logger.error(self.last_error)
                self._session_dead = True
                self.running = False
                break

    async def _receive_audio_loop(self, session):
        """Continuously receives translated audio stream from Gemini and queues for playback."""
        turn_audio_bytes = 0
        in_turn = False
        received_audio = False

        try:
            # `session.receive()` is a stream for one model turn in current
            # google-genai versions.  Calling it only once silently ends the
            # receiver after the first translation, leaving later microphone
            # chunks queued but never played.  Re-enter it for every turn.
            while self.running and not self._session_dead:
                async for response in session.receive():
                    if not self.running:
                        break

                    sc = response.server_content
                    if sc is not None:
                        input_transcription = getattr(sc, "input_transcription", None)
                        if input_transcription and getattr(input_transcription, "text", None):
                            print(f"\n  You: {input_transcription.text}", flush=True)

                        output_transcription = getattr(sc, "output_transcription", None)
                        if output_transcription and getattr(output_transcription, "text", None):
                            print(f"\n  Gemini: {output_transcription.text}", flush=True)

                        # Model audio response
                        if sc.model_turn:
                            if not in_turn:
                                in_turn = True
                                print("\n🔊 [Translating & Speaking]... ", end="", flush=True)

                            for part in sc.model_turn.parts:
                                if part.inline_data and part.inline_data.data:
                                    pcm_chunk = np.frombuffer(part.inline_data.data, dtype=np.int16)
                                    turn_audio_bytes += len(part.inline_data.data)
                                    self._model_speaking.set()
                                    self.playback_queue.put(pcm_chunk)
                                    if not received_audio:
                                        received_audio = True
                                        logger.info(
                                            "Received translated audio from Gemini; sending it to %s.",
                                            self._audio_device_name(self.output_device, "output"),
                                        )

                        # Interruption detection (barge-in)
                        if sc.interrupted:
                            logger.info("User interrupted translation.")
                            # Clear pending playback buffer
                            while not self.playback_queue.empty():
                                try:
                                    self.playback_queue.get_nowait()
                                    self.playback_queue.task_done()
                                except queue.Empty:
                                    break
                            in_turn = False
                            turn_audio_bytes = 0
                            self._model_speaking.clear()

                        # Turn complete.  The enclosing loop immediately
                        # starts listening for the next user turn.
                        if sc.turn_complete:
                            if in_turn:
                                print("✓ Done.")
                                in_turn = False
                                turn_audio_bytes = 0
                                # Put the marker after all response chunks so
                                # capture resumes only after the speaker drains.
                                self.playback_queue.put(self._turn_end_marker)
                            else:
                                self._model_speaking.clear()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self.running:
                self.last_error = f"Translation stream failed: {e}"
                logger.error(self.last_error)
                self._session_dead = True
                self.running = False

    def _queue_microphone_audio(self, pcm_bytes: bytes):
        """Called on the asyncio loop after sounddevice's callback returns."""
        if not self.running or self._session_dead:
            return
        try:
            self.audio_in_queue.put_nowait(pcm_bytes)
        except asyncio.QueueFull:
            # Prefer fresh speech to delayed speech.  Dropping a 100 ms chunk
            # is much less harmful than translating several seconds late.
            logger.warning("Live audio queue is full; dropping a stale 100 ms chunk.")

    @staticmethod
    def _audio_device_name(device, kind: str) -> str:
        """Return a human-readable device name without exposing an exception."""
        try:
            info = sd.query_devices(device=device, kind=kind)
            return str(info["name"])
        except Exception:
            return "the system default device"

    def _check_audio_devices(self) -> bool:
        """Fail before connecting when no microphone or speaker can open."""
        try:
            sd.check_input_settings(
                device=self.input_device,
                channels=1,
                samplerate=SAMPLE_RATE_IN,
                dtype="int16",
            )
            sd.check_output_settings(
                device=self.output_device,
                channels=1,
                samplerate=SAMPLE_RATE_OUT,
                dtype="int16",
            )
        except Exception as e:
            self.last_error = f"Live audio device is unavailable: {e}"
            logger.error(self.last_error)
            print(f"Error: {self.last_error}")
            print("Select a working microphone and speaker in your operating system, "
                  "then grant this terminal microphone permission.")
            return False

        print("Audio input: " + self._audio_device_name(self.input_device, "input"))
        print("Audio output: " + self._audio_device_name(self.output_device, "output"))
        return True

    @staticmethod
    def _enter_pressed() -> bool:
        """Return true for an Enter keypress without blocking the audio loop."""
        try:
            if not sys.stdin or not sys.stdin.isatty():
                return False
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if ready:
                sys.stdin.readline()
                return True
        except (OSError, ValueError):
            # A serial/no-TTY deployment uses the physical stop button.
            pass
        return False

    def run_live(self, stop_check: Optional[Callable[[], bool]] = None):
        """
        Run the live interpreter.
        Blocks until the user exits (via stop_check, pressing Enter, or Ctrl+C).
        """
        if not GENAI_AVAILABLE:
            print("Error: google-genai library is required.")
            return

        if not self.api_key:
            print("Error: gemini_api_key not found in config/settings.json.")
            return

        if not SD_AVAILABLE or sd is None:
            print("Error: sounddevice library is required for audio input/output.")
            return

        if not self._check_audio_devices():
            return

        self.running = True
        self._session_dead = False
        self.last_error = None
        # asyncio queues bind to the loop that first awaits them.  A fresh
        # queue makes it safe to leave and re-enter live mode in one process.
        self.audio_in_queue = asyncio.Queue(maxsize=MAX_PENDING_AUDIO_CHUNKS)
        self._stop_event.clear()

        # Start playback worker thread
        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._playback_thread.start()

        client = genai.Client(api_key=self.api_key)

        config_kwargs = {
            "response_modalities": ["AUDIO"],
            "system_instruction": types.Content(
                parts=[types.Part.from_text(text=self._get_system_prompt())]
            ),
        }
        # These fields were added after the old 1.x SDK.  They make the live
        # path observable while remaining compatible with older installations.
        if hasattr(types, "AudioTranscriptionConfig"):
            config_kwargs["input_audio_transcription"] = types.AudioTranscriptionConfig()
            config_kwargs["output_audio_transcription"] = types.AudioTranscriptionConfig()
        config = types.LiveConnectConfig(**config_kwargs)

        async def _main_async():
            logger.info(f"Connecting to Gemini Live API ({self.model})...")
            try:
                async with client.aio.live.connect(model=self.model, config=config) as session:
                    logger.info("Connected to Gemini Live session!")
                    print("=" * 60)
                    print(f"🎙️  LIVE TRANSLATION ACTIVE: {self.lang_pair.upper()}")
                    print(f"   Mic gain: {self.mic_gain:g}x")
                    print("   - Speak English → Translates into spoken Gujarati/Hindi")
                    print("   - Speak Gujarati/Hindi → Translates into spoken English")
                    print("   - Press Ctrl+C to stop.")
                    print("=" * 60)
                    print("🎤 Listening... (speak now)")

                    send_task = asyncio.create_task(self._send_audio_loop(session))
                    recv_task = asyncio.create_task(self._receive_audio_loop(session))

                    # Audio input stream from sounddevice
                    loop = asyncio.get_running_loop()
                    last_level_print = [0.0]  # track last level bar time

                    def mic_callback(indata, frames, time_info, status):
                        if not self.running:
                            return
                        if self._model_speaking.is_set():
                            # Do not send speaker echo back to Gemini. The
                            # next turn is accepted as soon as playback ends.
                            return
                        if status:
                            logger.debug(f"Mic status: {status}")
                        pcm_bytes = indata.tobytes()
                        loop.call_soon_threadsafe(self._queue_microphone_audio, pcm_bytes)

                        # Show live mic level bar every 0.5s
                        import time as _time
                        now = _time.time()
                        if now - last_level_print[0] > 0.5:
                            last_level_print[0] = now
                            peak = int(np.max(np.abs(indata)))
                            # After gain
                            effective = min(int(peak * self.mic_gain), 32767)
                            bar_len = int((effective / 32767) * 30)
                            bar = '█' * bar_len + '░' * (30 - bar_len)
                            sent = self._chunks_sent
                            print(f"\r  MIC |{bar}| peak={peak} eff={effective} sent={sent}  ", end='', flush=True)

                    in_stream = sd.InputStream(
                        samplerate=SAMPLE_RATE_IN,
                        channels=1,
                        dtype="int16",
                        blocksize=CHUNK_FRAMES,
                        device=self.input_device,
                        callback=mic_callback
                    )

                    with in_stream:
                        while self.running:
                            if stop_check and stop_check():
                                logger.info("Stop check triggered.")
                                break
                            if self._enter_pressed():
                                logger.info("Enter pressed; stopping live translation.")
                                break
                            await asyncio.sleep(0.1)

                    # Cancel tasks on exit
                    send_task.cancel()
                    recv_task.cancel()
                    await asyncio.gather(send_task, recv_task, return_exceptions=True)

            except Exception as e:
                self.last_error = f"Could not start live translation: {e}"
                logger.error(self.last_error)
                print(f"\nLive session ended: {e}")

        try:
            asyncio.run(_main_async())
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
        finally:
            self.stop()
            if self.last_error:
                print(f"\nLive translation error: {self.last_error}")

    def stop(self):
        """Stops the translation session and releases resources."""
        self.running = False
        self._stop_event.set()
        self.playback_queue.put(None)
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)


# ──────────────────────────────────────────────────────────────
# STANDALONE CLI TESTER
# ──────────────────────────────────────────────────────────────
def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║    🦯 BlindAssist — Real-Time Live Audio Translator 🦯    ║
║        Powered by Google Gemini Live Multimodal API       ║
╚═══════════════════════════════════════════════════════════╝
    """)

    print("Select Language Pair:")
    print("  1 → English <-> Gujarati (Default)")
    print("  2 → English <-> Hindi")
    print("  3 → Hindi <-> Gujarati")

    choice = input("Choice (1-3, default 1): ").strip()
    pair_map = {"1": "en-gu", "2": "en-hi", "3": "hi-gu"}
    selected_pair = pair_map.get(choice, "en-gu")

    translator = LiveTranslator(lang_pair=selected_pair)
    translator.run_live()


if __name__ == "__main__":
    main()
