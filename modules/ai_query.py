"""
ai_query.py — BlindAssist Project (v4)
======================================
Concurrent AI query: Groq and OpenAI race, then Gemini, then the offline model.
The answer is then delivered to whichever output the user chooses.

WHAT v4 ADDS
------------
The answer is no longer read out on whatever card the TTS module happens to be
pointed at. Once an answer is ready, the *speaker* asks where it should go:

    Button 1 -> the earphone   (confidential)
    Button 2 -> the speaker
    no answer in 20 seconds -> the speaker

The question is asked on the speaker even though the answer may end up in the
earphone: a user who is not already wearing the earphone would otherwise never
hear that they had been asked, and would be left in silence wondering whether
the device had crashed.

The default on timeout is the speaker rather than the earphone. That is the
less private option, so it is worth being explicit about why: the earphone may
not be plugged in or not be in the user's ear, and an answer played into an
earphone nobody is wearing is silently lost. A user who wanted privacy has 20
seconds and a single button press to say so — and the prompt itself is the
warning that the answer is about to be audible.

Device routing and button reading are borrowed from voice.py rather than
reimplemented, because that module already resolves which card is the speaker,
which is the earphone, and how to talk to the Pico W. If voice.py cannot be
imported, delivery falls back to the caller's speak_fn and nothing breaks.

WHAT WAS BROKEN BEFORE v3
-------------------------
1. ask_ai() did not accept speak_fn / flush_fn, so main.py's Mode 3 died with
   "ask_ai() got an unexpected keyword argument 'speak_fn'" after the question
   had already been recorded and transcribed.
2. gemini-3.5-flash does not exist; it was the default in two places, so every
   Gemini call 404'd silently.
3. The offline model preloaded at import even when settings.json disabled it —
   20-30s of CPU behind every startup.
4. Four ThreadPoolExecutors were built per question on a 4-core Pi.

Public API
----------
    ask_ai(prompt, context='', simplify=False,
           speak_fn=None, flush_fn=None)              -> str
    ask_ai_and_speak(prompt, context='', simplify=False,
                     speak_fn=None, flush_fn=None,
                     privacy_timeout=20.0)            -> str
    deliver_answer(answer, speak_fn=None,
                   privacy_timeout=20.0)              -> str  (where it played)
    index_text_in_rag(text) / index_file_in_rag(path)
"""

import concurrent.futures
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

from pathlib import Path
from typing import Callable, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# modules/ needs to import from services/
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

LOG_PATH = BASE_DIR / "logs" / "ai_query.log"
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("AIQueryModule")

_settings = {}
try:
    with open(CONFIG_PATH, "r") as f:
        _settings = json.load(f)
except Exception as e:
    logger.warning(f"Settings load failed: {e}")


# ── RAG pipeline (optional, fails gracefully) ───────────────
_rag_pipeline_instance = None
_rag_lock = threading.Lock()


def get_rag_pipeline():
    global _rag_pipeline_instance
    if _rag_pipeline_instance is not None:
        return _rag_pipeline_instance

    with _rag_lock:
        if _rag_pipeline_instance is not None:
            return _rag_pipeline_instance
        try:
            from services.embedder import Embedder
            from services.vector_store import VectorStore
            from services.retriever import Retriever
            from services.gemini_agent import GeminiAgent
            from services.rag_pipeline import RAGPipeline

            emb = Embedder(
                provider=_settings.get("embedding_provider", "local"),
                api_key=_settings.get("gemini_api_key") or os.getenv("GEMINI_API_KEY"),
            )
            store = VectorStore(store_path=str(BASE_DIR / "data" / "vector_store"),
                                dimension=emb.dimension)
            retriever = Retriever(
                embedder=emb,
                vector_store=store,
                cohere_api_key=(_settings.get("cohere_api_key")
                                or os.getenv("COHERE_API_KEY")),
            )
            gemini_agent = GeminiAgent(
                api_key=_settings.get("gemini_api_key") or os.getenv("GEMINI_API_KEY"),
                # was "gemini-3.5-flash", which does not exist
                model_name=_settings.get("gemini_model_name", "gemini-1.5-flash"),
            )
            _rag_pipeline_instance = RAGPipeline(emb, store, retriever, gemini_agent)
            logger.info("RAG pipeline initialised.")
        except Exception as e:
            # Not fatal: a question can still be answered without retrieval.
            logger.warning(f"RAG pipeline unavailable: {e}")
    return _rag_pipeline_instance


SYSTEM_PROMPT = (
    "You are BlindAssist, a concise assistant for visually impaired users. "
    "Answer in 1-3 short sentences. No markdown, no lists, no special characters. "
    "Speak naturally."
)

# ── Client handles (lazy, cached) ───────────────────────────
_groq_client = None
_openai_client = None
_gemini_client = None
_gemini_model_name = "gemini-1.5-flash"    # was "gemini-3.5-flash" — no such model
_offline_model = None
_offline_loading = threading.Event()
_offline_ready = False
MAX_PROMPT_CHARS = 8000

GROQ_TIMEOUT_S = float(_settings.get("groq_timeout_s", 6.0))
OPENAI_TIMEOUT_S = float(_settings.get("openai_timeout_s", 8.0))
GEMINI_TIMEOUT_S = float(_settings.get("gemini_timeout_s", 10.0))
RACE_TIMEOUT_S = float(_settings.get("ai_race_timeout_s", 10.0))

# How long the "is this confidential?" question waits before defaulting.
_privacy_settings = _settings.get("privacy")
PRIVACY_TIMEOUT_S = float(
    _privacy_settings.get("answer_route_timeout_seconds", 20.0)
    if isinstance(_privacy_settings, dict) else 20.0)

# One pool for the lifetime of the process. Building a fresh ThreadPoolExecutor
# per provider per call meant four pools for one question on a 4-core Pi.
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="ai_query")


# ── Latency narration ───────────────────────────────────────
LATENCY_MESSAGES = [
    "Your question is being processed.",
    "Still working on your answer, please wait.",
    "Almost there, just a moment.",
    "This is taking a bit longer than usual. Hang on.",
    "Still processing. Thank you for your patience.",
]
LATENCY_FIRST_DELAY = 3.0
LATENCY_INTERVAL = 4.0


class LatencyNarrator:
    """Speak reassurance while a slow call runs, then get out of the way.

    A blind user has no spinner: silence after "Thinking..." is
    indistinguishable from a crash. The first line is deliberately delayed so a
    fast answer is never spoken over, and flush_fn drops anything still queued
    so narration cannot be read out in front of the answer.
    """

    def __init__(self, speak_fn: Optional[Callable] = None,
                 messages: Optional[List[str]] = None,
                 first_delay: float = LATENCY_FIRST_DELAY,
                 interval: float = LATENCY_INTERVAL,
                 flush_fn: Optional[Callable] = None):
        self._speak_fn = speak_fn
        self._messages = messages or LATENCY_MESSAGES
        self._first_delay = first_delay
        self._interval = interval
        self._flush_fn = flush_fn
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._spoke_any = False

    def _narrate_loop(self):
        if self._stop_event.wait(timeout=self._first_delay):
            return                           # answered before the first line
        idx = 0
        while not self._stop_event.is_set():
            msg = self._messages[idx % len(self._messages)]
            if self._speak_fn:
                try:
                    self._speak_fn(msg)
                    self._spoke_any = True
                except Exception as e:
                    logger.debug(f"Narrator speak error: {e}")
            else:
                logger.info(f"[Narrator] {msg}")
            idx += 1
            if self._stop_event.wait(timeout=self._interval):
                return

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._narrate_loop, daemon=True,
                                        name="latency_narrator")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._spoke_any and self._flush_fn:
            try:
                self._flush_fn()
            except Exception as e:
                logger.debug(f"Narrator flush error: {e}")
        self._spoke_any = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


def _bounded_text(text: str, limit: int = MAX_PROMPT_CHARS) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "\n[Text truncated.]"


# ── ANSWER DELIVERY: ask the speaker, then route ────────────
_voice_module = None
_voice_lookup_done = False


def _voice():
    """voice.py, or None. Imported lazily so a broken voice.py cannot stop AI.

    voice.py already resolves SPEAKER_DEVICE / EARPHONE_DEVICE from
    settings.json and owns the Pico W connection, so routing an answer reuses
    that rather than keeping a second, drifting copy of the same knowledge.
    """
    global _voice_module, _voice_lookup_done
    if _voice_lookup_done:
        return _voice_module
    _voice_lookup_done = True
    for path in ("modules.voice", "voice"):
        try:
            _voice_module = __import__(path, fromlist=["*"])
            return _voice_module
        except Exception:
            continue
    logger.info("voice.py not importable — answers will use the default output.")
    return None


_tts_module = None
_tts_lookup_done = False


def _tts():
    """tts.py, or None. Imported lazily, exactly like _voice().

    tts.py is the project's one speech backend, and the only thing here that
    knows how to synthesise the natural gTTS voice the rest of the device
    speaks with. A missing or broken tts.py must never stop an answer being
    delivered, so this is best-effort and the espeak path below remains as the
    fallback.
    """
    global _tts_module, _tts_lookup_done
    if _tts_lookup_done:
        return _tts_module
    _tts_lookup_done = True
    for path in ("modules.tts", "tts"):
        try:
            module = __import__(path, fromlist=["*"])
            if hasattr(module, "speak_on_device"):
                _tts_module = module
                return _tts_module
            logger.info(f"{path} has no speak_on_device() — using espeak for routed speech.")
            return None
        except Exception:
            continue
    logger.info("tts.py not importable — routed speech will use espeak.")
    return None


def _speaker_device() -> Optional[str]:
    v = _voice()
    return getattr(v, "SPEAKER_DEVICE", None) if v else None


def _earphone_device() -> Optional[str]:
    v = _voice()
    return getattr(v, "EARPHONE_DEVICE", None) if v else None


def _say_on_device(text: str, device: Optional[str],
                   speak_fn: Optional[Callable] = None) -> bool:
    """Say something on a specific card, in the device's normal voice.

    tts.py is tried first. It synthesises with gTTS — the same voice every
    other prompt on this device uses — and plays it on `device`. This module
    used to synthesise the answer itself with espeak, which made the AI's
    answer the *only* utterance produced by a formant synthesiser: audibly
    buzzy and creaky next to the gTTS prompt that had just introduced it.
    espeak was never a voice choice, only the one thing known to hit a
    specific card; tts.speak_on_device() does that with the good voice.

    espeak-to-WAV plus `aplay -D` stays as the offline fallback, and speak_fn
    as the last resort — it will not honour the chosen card, but a spoken
    answer on the wrong speaker beats no answer at all.
    """
    if not device:
        if speak_fn:
            speak_fn(text)
            return True
        return False

    tts = _tts()
    if tts is not None:
        try:
            # block=True: callers ask the user a question straight afterwards
            # and must not race the audio.
            tts.speak_on_device(text, device, block=True)
            return True
        except Exception as e:
            logger.warning(f"tts.py could not speak on {device} ({e}); using espeak.")

    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    aplay = shutil.which("aplay")
    if espeak and aplay:
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                path = tmp.name
            rate = str(int(_settings.get("tts_rate", 130)))
            # -a 175 lifts espeak's quiet default, -g 4 spaces words slightly.
            # Both make the fallback voice easier to follow; neither can fail
            # in a way that produces no file, and the size check below catches
            # it if the build rejects a flag.
            base = [espeak, "-s", rate, "-a", "175", "-g", "4", "-w", path, text]
            subprocess.run(base, timeout=60,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.getsize(path) <= 44:
                subprocess.run([espeak, "-s", rate, "-w", path, text], timeout=60,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.getsize(path) > 44:
                # An explicit 500 ms ALSA ring buffer. aplay's default is a few
                # tens of milliseconds, which underruns — heard as crackling
                # mid-sentence — whenever the Pi is busy. Retry without the
                # timings if a card refuses them.
                tuned = [aplay, "-q", "-D", device, "--buffer-time", "500000",
                         "--period-time", "100000", path]
                result = subprocess.run(tuned, timeout=600,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                if result.returncode != 0:
                    subprocess.run([aplay, "-q", "-D", device, path], timeout=600,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
        except Exception as e:
            logger.warning(f"Routed speech failed on {device}: {e}")
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    if speak_fn:
        try:
            speak_fn(text)
            return True
        except Exception as e:
            logger.debug(f"speak_fn failed: {e}")
    return False


def _drain_stdin():
    """Discard input typed before the question was asked — it is not an answer.

    Keystrokes sit in the terminal buffer until something reads them, so an
    impatient ENTER pressed while the AI was thinking would otherwise be
    consumed here, and a stale "2" would put a confidential answer on the open
    speaker. voice.py drains for the same reason before it records.
    """
    if sys.stdin is None or not sys.stdin.isatty():
        return
    try:
        import termios

        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception as e:
        logger.debug(f"Could not drain stdin: {e}")


def _wait_for_route_choice(timeout: float) -> Optional[str]:
    """Wait for Button 1 or Button 2, or a typed 1/2 at a terminal."""
    v = _voice()
    deadline = time.time() + timeout
    tty = sys.stdin is not None and sys.stdin.isatty()
    has_buttons = False

    if tty:
        _drain_stdin()

    if v is not None and hasattr(v, "_wait_for_button"):
        try:
            has_buttons = v._get_serial() is not None
        except Exception:
            has_buttons = False

    while time.time() < deadline:
        if has_buttons:
            try:
                choice = v._wait_for_button(("1", "2"), timeout=min(0.5, timeout))
                if choice:
                    return choice
            except Exception as e:
                logger.debug(f"Button read error: {e}")
                has_buttons = False
        if tty:
            import select

            if select.select([sys.stdin], [], [], 0.2)[0]:
                typed = sys.stdin.readline().strip()
                if typed in ("1", "2"):
                    return typed
        elif not has_buttons:
            # Nothing can answer — do not burn the full timeout in a tight loop.
            time.sleep(min(0.5, max(0.0, deadline - time.time())))
    return None


def deliver_answer(answer: str, speak_fn: Optional[Callable] = None,
                   privacy_timeout: Optional[float] = None) -> str:
    """Ask on the speaker where the answer should go, then say it there.

    Returns "earphone" or "speaker" so the caller can log what happened.
    """
    if not answer:
        return "none"

    wait_s = float(privacy_timeout if privacy_timeout is not None
                   else PRIVACY_TIMEOUT_S)
    speaker = _speaker_device()
    earphone = _earphone_device()

    if not speaker and not earphone:
        # voice.py unavailable: nothing to route between, so just answer.
        if speak_fn:
            speak_fn(answer)
        return "default"

    question = ("Your answer is ready. Is this confidential? "
                "Press button 1 to hear it in the earphone, "
                "or button 2 to hear it on the speaker.")
    _say_on_device(question, speaker, speak_fn)

    if sys.stdin is not None and sys.stdin.isatty():
        print(f"\n🔒 {question}")
        print(f"   (or type 1 / 2 here — {wait_s:.0f}s, then the speaker)")

    choice = _wait_for_route_choice(wait_s)

    if choice == "1":
        device, where = earphone or speaker, "earphone"
    else:
        device, where = speaker or earphone, "speaker"
        if choice is None:
            logger.info(f"No answer in {wait_s:.0f}s — using the speaker.")

    logger.info(f"Delivering the answer on the {where} ({device}).")
    _say_on_device(answer, device, speak_fn)
    return where


# ── Provider initialisers ───────────────────────────────────
def _init_groq() -> bool:
    global _groq_client
    if _groq_client:
        return True
    key = _settings.get("groq_api_key") or os.environ.get("GROQ_API_KEY", "")
    if not key or key.startswith("YOUR_"):
        return False
    try:
        from groq import Groq

        _groq_client = Groq(api_key=key)
        return True
    except Exception as e:
        logger.debug(f"Groq init failed: {e}")
        return False


def _init_openai() -> bool:
    global _openai_client
    if _openai_client:
        return True
    key = _settings.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
    if not key or key.startswith("YOUR_"):
        return False
    try:
        from openai import OpenAI

        _openai_client = OpenAI(api_key=key)
        return True
    except Exception as e:
        logger.debug(f"OpenAI init failed: {e}")
        return False


def _init_gemini() -> bool:
    global _gemini_client, _gemini_model_name
    if _gemini_client:
        return True
    key = _settings.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY", "")
    if not key or key.startswith("YOUR_"):
        return False
    try:
        from google import genai

        _gemini_client = genai.Client(api_key=key)
        _gemini_model_name = _settings.get("gemini_model_name", "gemini-1.5-flash")
        return True
    except Exception as e:
        logger.debug(f"Gemini init failed: {e}")
        return False


def _with_timeout(client, timeout: float):
    """Apply a per-request timeout when the SDK supports it."""
    try:
        return client.with_options(timeout=timeout)
    except Exception:
        return client


# ── Offline model ───────────────────────────────────────────
def _preload_offline_model():
    global _offline_model, _offline_ready
    model_path = _settings.get("offline_model_path", "")
    if not model_path:
        _offline_loading.set()
        return
    if not Path(model_path).exists():
        alt = BASE_DIR / "models" / Path(model_path).name
        if alt.exists():
            model_path = str(alt)
        else:
            logger.info("Offline model file not found — skipping preload.")
            _offline_loading.set()
            return
    try:
        from llama_cpp import Llama

        logger.info("Preloading offline model …")
        t0 = time.time()
        _offline_model = Llama(model_path=model_path, n_ctx=2048, n_threads=4,
                               verbose=False, n_batch=512)
        _offline_ready = True
        logger.info(f"Offline model ready in {time.time() - t0:.1f}s")
    except Exception as e:
        logger.warning(f"Offline preload failed: {e}")
    finally:
        _offline_loading.set()


_preload_thread = None
_preload_lock = threading.Lock()


def _start_offline_preload():
    global _preload_thread
    with _preload_lock:
        if _preload_thread is not None:
            return
        _preload_thread = threading.Thread(target=_preload_offline_model,
                                           daemon=True, name="offline_preload")
        _preload_thread.start()


# Opt-in. The old version started this at import unconditionally, so a Pi that
# had set offline_model_preload: false still spent 20-30s loading TinyLlama
# behind every startup.
if _settings.get("offline_model_preload", False):
    _start_offline_preload()


# ── Provider callers ────────────────────────────────────────
def _ask_groq(prompt: str, simplify: bool = False,
              timeout: float = GROQ_TIMEOUT_S) -> Optional[str]:
    if not _init_groq():
        return None
    try:
        system = SYSTEM_PROMPT + (" Use very simple words." if simplify else "")
        resp = _with_timeout(_groq_client, timeout).chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Groq error: {e}")
        return None


def _ask_openai(prompt: str, simplify: bool = False,
                timeout: float = OPENAI_TIMEOUT_S) -> Optional[str]:
    if not _init_openai():
        return None
    try:
        system = SYSTEM_PROMPT + (" Use very simple words." if simplify else "")
        resp = _with_timeout(_openai_client, timeout).chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"OpenAI error: {e}")
        return None


def _ask_gemini(prompt: str, simplify: bool = False,
                timeout: float = GEMINI_TIMEOUT_S) -> Optional[str]:
    if not _init_gemini():
        return None
    try:
        from google.genai import types

        system = SYSTEM_PROMPT + (" Use very simple words." if simplify else "")
        full = ("Explain simply: " + prompt) if simplify else prompt

        try:
            config = types.GenerateContentConfig(
                system_instruction=system,
                http_options=types.HttpOptions(timeout=int(timeout * 1000)),
            )
        except Exception:
            # Older google-genai has no HttpOptions; the call still works.
            config = types.GenerateContentConfig(system_instruction=system)

        resp = _gemini_client.models.generate_content(
            model=_gemini_model_name, contents=full, config=config)
        return resp.text.strip() if resp and resp.text else None
    except Exception as e:
        logger.warning(f"Gemini error: {e}")
        return None


def _ask_offline(prompt: str, simplify: bool = False) -> Optional[str]:
    if not _settings.get("offline_model_path"):
        return None
    _start_offline_preload()
    _offline_loading.wait(timeout=float(_settings.get("offline_load_timeout_s", 60)))
    if not _offline_ready or _offline_model is None:
        return None
    try:
        mp = _settings.get("offline_model_path", "").lower()
        if "qwen" in mp:
            full = (f"<|im_start|>system\n{SYSTEM_PROMPT}\n"
                    f"<|im_start|>user\n{prompt}\n<|im_start|>assistant\n")
            stops = ["<|im_end|>", "<|im_start|>"]
        elif "llama-3" in mp:
            full = (f"<|begin_of_text|>system\n{SYSTEM_PROMPT}\n<|eot_id|>"
                    f"user\n{prompt}\n<|eot_id|>assistant\n")
            stops = ["<|eot_id|>"]
        else:
            full = f"User: {prompt}\nAssistant:"
            stops = ["User:", "\n\n"]

        resp = _offline_model(full, max_tokens=150, stop=stops, echo=False,
                              temperature=0.7)
        return resp["choices"][0]["text"].strip()
    except Exception as e:
        logger.error(f"Offline error: {e}")
        return None


# ── Public API ──────────────────────────────────────────────
def ask_ai(prompt: str, context: str = "", simplify: bool = False,
           speak_fn: Optional[Callable] = None,
           flush_fn: Optional[Callable] = None) -> str:
    """Answer a question: Groq and OpenAI race, then Gemini, then offline.

    Returns the text and speaks nothing except latency reassurance. Use
    ask_ai_and_speak() when the answer should also be delivered.
    """
    if not prompt or not prompt.strip():
        return "I didn't receive a question. Please try again."

    if not context:
        pipeline = get_rag_pipeline()
        if pipeline:
            try:
                context = pipeline.retriever.get_relevant_context(prompt, k=3)
                if context:
                    logger.info(f"RAG context: {len(context)} chars")
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")

    prompt = _bounded_text(prompt, 2000)
    context = _bounded_text(context, 6000)
    full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}" if context else prompt
    logger.info(f'Query: "{prompt[:60]}"')

    narrator = LatencyNarrator(speak_fn=speak_fn, flush_fn=flush_fn)
    narrator.start()
    try:
        # Phase 1 — race the two fast providers, first real answer wins.
        futures = {
            _EXECUTOR.submit(_ask_groq, full_prompt, simplify): "groq",
            _EXECUTOR.submit(_ask_openai, full_prompt, simplify): "openai",
        }
        try:
            for future in concurrent.futures.as_completed(futures,
                                                          timeout=RACE_TIMEOUT_S):
                source = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.debug(f"{source} exception: {e}")
                    continue
                if result:
                    logger.info(f"Answer from {source}")
                    return result
        except concurrent.futures.TimeoutError:
            logger.warning("Fast providers timed out.")

        # Phase 2 — Gemini.
        result = _ask_gemini(full_prompt, simplify)
        if result:
            logger.info("Answer from gemini")
            return result

        # Phase 3 — offline, the only one that works with no internet.
        result = _ask_offline(full_prompt, simplify)
        if result:
            logger.info("Answer from offline model")
            return result

        return ("I'm sorry, I cannot answer right now. "
                "Please check your internet connection.")
    finally:
        narrator.stop()


def ask_ai_and_speak(prompt: str, context: str = "", simplify: bool = False,
                     speak_fn: Optional[Callable] = None,
                     flush_fn: Optional[Callable] = None,
                     privacy_timeout: Optional[float] = None) -> str:
    """Answer the question, ask where it should be heard, then say it there.

    Returns the answer text as well, so the caller can log or display it — but
    the caller must NOT speak the return value again, or the user hears it
    twice.
    """
    answer = ask_ai(prompt, context=context, simplify=simplify,
                    speak_fn=speak_fn, flush_fn=flush_fn)
    where = deliver_answer(answer, speak_fn=speak_fn,
                           privacy_timeout=privacy_timeout)
    logger.info(f"Answer delivered on the {where}.")
    return answer


def index_text_in_rag(text: str):
    pipeline = get_rag_pipeline()
    if pipeline:
        pipeline.index_document(text)


def index_file_in_rag(file_path: str) -> bool:
    pipeline = get_rag_pipeline()
    if pipeline:
        return pipeline.index_file(file_path)
    return False


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("BlindAssist AI Query — racing Groq vs OpenAI")
    print(f"  speaker  {_speaker_device()}")
    print(f"  earphone {_earphone_device()}")
    print(f"  espeak   {bool(shutil.which('espeak-ng') or shutil.which('espeak'))}")
    print(f"  voice    {'tts.py / gTTS' if _tts() else 'espeak (fallback)'}")

    while True:
        try:
            q = input("\nQuestion (QUIT to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.upper() == "QUIT":
            break
        if q:
            t0 = time.time()
            answer = ask_ai_and_speak(q, speak_fn=print)
            print(f"Answer ({time.time() - t0:.2f}s): {answer}")
