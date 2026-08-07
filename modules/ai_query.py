"""
ai_query.py — BlindAssist Project (OPTIMIZED v2)
===============================================
Async API calls with concurrent fallback.
Preloads offline model at import time (hidden in thread).
Non-blocking for online APIs.

v2 additions:
- LatencyNarrator: Speaks periodic reassurance messages to the user during
  long AI processing times, preventing blind users from thinking the device
  has frozen. Configurable interval and message pool.
"""

import sys
import signal
import logging
import json
import os
import threading
import concurrent.futures
import time

from pathlib import Path
from typing import Optional, Callable, List

from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Ensure cross-package imports work (modules/ → services/) ──
import sys as _sys
if str(BASE_DIR) not in _sys.path:
    _sys.path.insert(0, str(BASE_DIR))

LOG_PATH = BASE_DIR / "logs" / "ai_query.log"
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("AIQueryModule")

_settings = {}
try:
    with open(CONFIG_PATH, 'r') as f:
        _settings = json.load(f)
except Exception as e:
    logger.warning(f"Settings load failed: {e}")

_rag_pipeline_instance = None

def get_rag_pipeline():
    global _rag_pipeline_instance
    if _rag_pipeline_instance is None:
        try:
            from services.embedder import Embedder
            from services.vector_store import VectorStore
            from services.retriever import Retriever
            from services.gemini_agent import GeminiAgent
            from services.rag_pipeline import RAGPipeline

            emb = Embedder(
                provider=_settings.get("embedding_provider", "local"),
                api_key=_settings.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
            )
            store_path = str(BASE_DIR / "data" / "vector_store")
            store = VectorStore(store_path=store_path, dimension=emb.dimension)
            retriever = Retriever(
                embedder=emb,
                vector_store=store,
                cohere_api_key=_settings.get("cohere_api_key") or os.getenv("COHERE_API_KEY")
            )
            gemini_agent = GeminiAgent(
                api_key=_settings.get("gemini_api_key") or os.getenv("GEMINI_API_KEY"),
                model_name=_settings.get("gemini_model_name", "gemini-3.5-flash")
            )
            _rag_pipeline_instance = RAGPipeline(emb, store, retriever, gemini_agent)
            logger.info("RAG Pipeline successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline in ai_query: {e}")
    return _rag_pipeline_instance


SYSTEM_PROMPT = (
    "You are BlindAssist, a concise assistant for visually impaired users. "
    "Answer in 1-3 short sentences. No markdown, no lists, no special characters. "
    "Speak naturally."
)

# ── CLIENT INITIALIZATION (lazy but cached) ─────────────────
_groq_client = None
_openai_client = None
_gemini_client = None
_gemini_model_name = "gemini-3.5-flash"
_offline_model = None
_offline_loading = threading.Event()
_offline_ready = False
MAX_PROMPT_CHARS = 8000

# ── LATENCY NARRATOR ────────────────────────────────────────
# Speaks periodic reassurance messages to the user while AI processes their query.
# This is critical for accessibility — a blind user has no visual loading spinner.

# Default narration messages (rotated in order, then loops)
LATENCY_MESSAGES = [
    "Your question is being processed.",
    "Still working on your answer, please wait.",
    "Almost there, just a moment.",
    "This is taking a bit longer than usual. Hang on.",
    "Still processing. Thank you for your patience.",
]

# How many seconds of silence before the first narration message
LATENCY_FIRST_DELAY = 3.0

# How many seconds between subsequent narration messages
LATENCY_INTERVAL = 4.0


class LatencyNarrator:
    """
    Background narrator that speaks periodic reassurance messages while
    the AI query is processing. Keeps blind users engaged and informed
    during high-latency API calls.

    Usage:
        narrator = LatencyNarrator(speak_fn=tts.speak)
        narrator.start()
        # ... do slow work ...
        narrator.stop()  # stops narration immediately

    Or as a context manager:
        with LatencyNarrator(speak_fn=tts.speak):
            # ... do slow work ...
    """

    def __init__(
        self,
        speak_fn: Optional[Callable] = None,
        messages: Optional[List[str]] = None,
        first_delay: float = LATENCY_FIRST_DELAY,
        interval: float = LATENCY_INTERVAL,
        flush_fn: Optional[Callable] = None,
    ):
        self._speak_fn = speak_fn
        self._messages = messages or LATENCY_MESSAGES
        self._first_delay = first_delay
        self._interval = interval
        # TTS is a FIFO queue, so reassurance messages queued while the AI was
        # thinking would otherwise play *after* the answer is ready. flush_fn
        # drops anything still pending the moment we stop narrating.
        self._flush_fn = flush_fn
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._spoke_any = False

    def _narrate_loop(self):
        """Background thread: waits for first_delay, then speaks messages at interval."""
        # Wait for the initial delay before first message
        if self._stop_event.wait(timeout=self._first_delay):
            return  # Stopped before first message was needed

        msg_index = 0
        while not self._stop_event.is_set():
            message = self._messages[msg_index % len(self._messages)]
            if self._speak_fn:
                try:
                    self._speak_fn(message)
                    self._spoke_any = True
                except Exception as e:
                    logger.debug(f"Narrator speak error: {e}")
            else:
                logger.info(f"[Narrator] {message}")

            msg_index += 1

            # Wait for interval before next message (or stop signal)
            if self._stop_event.wait(timeout=self._interval):
                return

    def start(self):
        """Start the background narration thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._narrate_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop narration immediately and discard queued reassurance messages.

        Safe to call multiple times.
        """
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
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[Text truncated for safety.]"

def _init_groq():
    global _groq_client
    if _groq_client:
        return True
    key = _settings.get("groq_api_key") or os.environ.get("GROQ_API_KEY")
    if not key or key.startswith("YOUR_"):
        return False
    try:
        from groq import Groq
        _groq_client = Groq(api_key=key)
        return True
    except Exception as e:
        logger.debug(f"Groq unavailable: {e}")
        return False

def _init_openai():
    global _openai_client
    if _openai_client:
        return True
    key = _settings.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
    if not key or key.startswith("YOUR_"):
        return False
    try:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=key)
        return True
    except Exception as e:
        logger.debug(f"OpenAI unavailable: {e}")
        return False

def _init_gemini():
    global _gemini_client, _gemini_model_name
    if _gemini_client:
        return True
    key = _settings.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
    if not key or key.startswith("YOUR_"):
        return False
    try:
        from google import genai
        _gemini_client = genai.Client(api_key=key)
        _gemini_model_name = _settings.get("gemini_model_name", "gemini-3.5-flash")
        return True
    except Exception as e:
        logger.debug(f"Gemini unavailable: {e}")
        return False

def _preload_offline_model():
    """Background thread: preload offline model to avoid 10-30s cold start."""
    global _offline_model, _offline_ready
    model_path = _settings.get("offline_model_path", "")
    if not model_path:
        _offline_loading.set()
        return
    
    # Try relative path
    if not Path(model_path).exists():
        alt = BASE_DIR / "models" / Path(model_path).name
        if alt.exists():
            model_path = str(alt)
        else:
            logger.info("Offline model not found, skipping preload.")
            _offline_loading.set()
            return
    
    try:
        from llama_cpp import Llama
        logger.info("Preloading offline model (this may take 20-30s)...")
        start = time.time()
        _offline_model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            verbose=False,
            n_batch=512,
        )
        _offline_ready = True
        logger.info(f"Offline model ready in {time.time()-start:.1f}s")
    except Exception as e:
        logger.warning(f"Offline preload failed: {e}")
    finally:
        _offline_loading.set()

# Offline model preload.
#
# The preload thread used to be created and then never started (the .start()
# call was commented out), which left the _offline_loading event permanently
# unset — so the offline fallback sat in `_offline_loading.wait(timeout=60)`
# for a full minute before returning None. With no network that meant a
# 60-second silence followed by an error, on every single question.
#
# Preloading is now opt-in via settings (it costs 20-30s of CPU and ~700MB of
# RAM at startup), and _ask_offline starts the load on demand when it wasn't
# preloaded, so the wait is never wasted.
_preload_thread = None
_preload_lock = threading.Lock()


def _start_offline_preload():
    """Start the offline model load once, in the background. Idempotent."""
    global _preload_thread
    with _preload_lock:
        if _preload_thread is not None:
            return
        _preload_thread = threading.Thread(
            target=_preload_offline_model, daemon=True, name="offline_preload"
        )
        _preload_thread.start()


if _settings.get("offline_model_preload", False):
    _start_offline_preload()

# ── API CALLERS (with timeouts) ─────────────────────────────

# NOTE ON TIMEOUTS
# The previous version wrapped each provider call in its own
# `with ThreadPoolExecutor() as ex:` block and relied on
# `future.result(timeout=N)` to bound it. That never worked: leaving the
# `with` block calls shutdown(wait=True), which blocks until the HTTP call
# finishes anyway — a measured "5 second timeout" still took the full call
# duration. Timeouts are now enforced where they actually work: on the SDK
# clients themselves (real socket timeouts that abort the request), with a
# single top-level race in ask_ai().
GROQ_TIMEOUT_S = float(_settings.get("groq_timeout_s", 6.0))
OPENAI_TIMEOUT_S = float(_settings.get("openai_timeout_s", 8.0))
GEMINI_TIMEOUT_S = float(_settings.get("gemini_timeout_s", 10.0))
RACE_TIMEOUT_S = float(_settings.get("ai_race_timeout_s", 10.0))

# One long-lived pool. Creating a pool per call leaked threads and made
# every call pay pool setup/teardown.
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="ai_query"
)


def _ask_groq(prompt: str, simplify: bool = False, timeout: float = GROQ_TIMEOUT_S) -> Optional[str]:
    if not _init_groq():
        return None
    try:
        system = SYSTEM_PROMPT
        if simplify:
            system += " Use very simple words."
        response = _groq_client.with_options(timeout=timeout).chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Groq unavailable: {e}")
        return None


def _ask_openai(prompt: str, simplify: bool = False, timeout: float = OPENAI_TIMEOUT_S) -> Optional[str]:
    if not _init_openai():
        return None
    try:
        system = SYSTEM_PROMPT
        if simplify:
            system += " Use very simple words."
        response = _openai_client.with_options(timeout=timeout).chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"OpenAI unavailable: {e}")
        return None


def _ask_gemini(prompt: str, simplify: bool = False, timeout: float = GEMINI_TIMEOUT_S) -> Optional[str]:
    if not _init_gemini():
        return None
    try:
        from google.genai import types
        full = "Explain simply: " + prompt if simplify else prompt
        system = SYSTEM_PROMPT
        if simplify:
            system += " Use very simple words."
        response = _gemini_client.models.generate_content(
            model=_gemini_model_name,
            contents=full,
            config=types.GenerateContentConfig(
                system_instruction=system,
                # google-genai takes request timeouts in milliseconds.
                http_options=types.HttpOptions(timeout=int(timeout * 1000)),
            ),
        )
        return response.text.strip() if response and response.text else None
    except Exception as e:
        logger.warning(f"Gemini unavailable: {e}")
        return None

def _ask_offline(prompt: str, simplify: bool = False) -> Optional[str]:
    global _offline_ready

    # Nothing configured — return immediately instead of waiting on a load
    # that will never happen.
    if not _settings.get("offline_model_path"):
        return None

    _start_offline_preload()          # no-op if already loading/loaded
    _offline_loading.wait(timeout=float(_settings.get("offline_load_timeout_s", 60)))
    if not _offline_ready or _offline_model is None:
        return None


    try:
        model_path = _settings.get("offline_model_path", "").lower()
        if "qwen" in model_path:
            full = (
                "<|im_start|>system\n" + SYSTEM_PROMPT + "\n"
                "<|im_start|>user\n" + prompt + "\n"
                "<|im_start|>assistant\n"
            )
            stops = ["<|im_end|>", "<|im_start|>"]
        elif "llama-3" in model_path:
            full = (
                "<|begin_of_text|>system\n" + SYSTEM_PROMPT + "\n<|eot_id|>"
                "user\n" + prompt + "\n<|eot_id|>"
                "assistant\n"
            )
            stops = ["<|eot_id|>"]
        else:
            full = f"User: {prompt}\nAssistant:"
            stops = ["User:", "\n\n"]
        
        response = _offline_model(
            full,
            max_tokens=150,
            stop=stops,
            echo=False,
            temperature=0.7,
        )
        return response['choices'][0]['text'].strip()
    except Exception as e:
        logger.error(f"Offline error: {e}")
        return None

# ── PUBLIC API ──────────────────────────────────────────────

def ask_ai(prompt: str, context: str = '', simplify: bool = False,
           speak_fn: Optional[Callable] = None,
           flush_fn: Optional[Callable] = None) -> str:
    """
    Concurrent AI query with fastest-response-wins strategy.
    Tries Groq + OpenAI simultaneously, uses whichever answers first.
    Falls back to Gemini, then offline.

    Args:
        speak_fn: Optional TTS function for latency narration. If provided,
                  the system will speak periodic reassurance messages to the
                  user while the AI processes their query (e.g., "Your question
                  is being processed", "Still working on it"). This prevents
                  blind users from thinking the device has frozen during
                  high-latency API calls.
        flush_fn: Optional TTS flush function. Called when narration stops so
                  reassurance messages still sitting in the speech queue are
                  dropped instead of playing after the answer.
    """
    if not prompt or not prompt.strip():
        return "I didn't receive a question. Please try again."

    if not context:
        pipeline = get_rag_pipeline()
        if pipeline:
            try:
                context = pipeline.retriever.get_relevant_context(prompt, k=3)
                if context:
                    logger.info(f"Retrieved RAG context: {len(context)} chars.")
            except Exception as e:
                logger.warning(f"Failed to retrieve RAG context in ask_ai: {e}")

    prompt = _bounded_text(prompt, 2000)
    context = _bounded_text(context, 6000)
    full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}" if context else prompt
    logger.info(f"Query: \"{prompt[:50]}...\"")

    # Start the latency narrator — speaks periodic updates while AI processes
    narrator = LatencyNarrator(speak_fn=speak_fn, flush_fn=flush_fn)
    narrator.start()

    try:
        # Phase 1: Race Groq vs OpenAI (fastest valid answer wins).
        # The pool is module-level and is never shut down here, so a slow
        # loser can finish in the background instead of blocking the winner.
        futures = {
            _EXECUTOR.submit(_ask_groq, full_prompt, simplify): 'groq',
            _EXECUTOR.submit(_ask_openai, full_prompt, simplify): 'openai',
        }
        try:
            for future in concurrent.futures.as_completed(futures, timeout=RACE_TIMEOUT_S):
                source = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.debug(f"{source} returned no answer: {e}")
                    continue
                if result:
                    logger.info(f"First response from {source}")
                    return result
        except concurrent.futures.TimeoutError:
            logger.warning("Timed out waiting for fast AI providers.")

        # Phase 2: Try Gemini
        result = _ask_gemini(full_prompt, simplify)
        if result:
            return result

        # Phase 3: Offline (already preloaded)
        result = _ask_offline(full_prompt, simplify)
        if result:
            return result

        return "I'm sorry, I cannot answer right now. Please check your connection."

    finally:
        # Always stop narration before returning the answer
        narrator.stop()

def index_text_in_rag(text: str):
    """Indexes raw text (e.g. OCR scan) into the RAG vector database."""
    pipeline = get_rag_pipeline()
    if pipeline:
        pipeline.index_document(text)

def index_file_in_rag(file_path: str) -> bool:
    """Indexes a text file (e.g. NCERT textbook chapter) into the RAG vector database."""
    pipeline = get_rag_pipeline()
    if pipeline:
        return pipeline.index_file(file_path)
    return False


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    print("AI Query Optimized — racing Groq vs OpenAI")
    while True:
        q = input("Question (QUIT): ").strip()
        if q.upper() == "QUIT":
            break
        if q:
            print("Racing APIs...")
            start = time.time()
            print(f"Answer ({time.time()-start:.2f}s): {ask_ai(q, speak_fn=print)}\n")
