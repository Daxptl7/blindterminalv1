"""
translator.py — BlindAssist Project
=====================================
Project  : Accessible Educational Terminal for Visually Impaired
Team     : Dhruv Vaghela & Dax Patel  |  CSR / Infineon 2025
Module   : Multilingual Translation

Translates text between English, Hindi and Gujarati.

The device is used in classrooms where the network is unreliable or absent,
so translation is a chain rather than a single call. Each link is tried in
turn and the first that answers wins:

    cache  →  Google  →  LibreTranslate  →  MyMemory  →  Argos  →  phrasebook

The three network links are skipped outright — not attempted and timed out —
whenever the device is known to be offline, which is what turned a failed
translation into a 10-second stall for a user who cannot see a progress bar.
Everything that succeeds online is written to an on-disk cache, so a phrase
translated once keeps working after the network goes away.

Callers that need to know *how* the answer was produced (to speak it in the
right voice, or to admit that no translation happened) should use
translate_ex(); translate() keeps the original string-in/string-out contract.
"""

import os
import re
import sys
import json
import time
import signal
import socket
import logging
import tempfile
import threading

import requests

from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

# --- NEW IMPORT FOR VOICE INPUT ---
try:
    from modules.voice import listen
except ImportError:
    try:
        from voice import listen
    except ImportError:
        listen = None
        logging.warning("Voice module not found. Option 3 will be disabled.")

# ──────────────────────────────────────────────────────────────
# HARDWARE FLAGS (Pi Flag Pattern)
# ──────────────────────────────────────────────────────────────
HEADLESS = False
USE_PICAMERA = False
USE_GPIO = False
USE_CORAL = False

# ──────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "translator.log"
CONFIG_PATH = BASE_DIR / "config" / "settings.json"
CACHE_PATH = BASE_DIR / "cache" / "translations.json"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────
logger = logging.getLogger("TranslatorModule")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        _fh = logging.FileHandler(LOG_PATH)
        _fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(_fh)
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────────────────────────
def _load_settings() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


_settings = _load_settings()

# A LibreTranslate instance running on the Pi itself (or on the classroom LAN)
# is the only way to get full-quality translation with no internet at all.
# Left empty it is simply skipped.
LIBRE_URL = str(_settings.get("libretranslate_url", "") or "").rstrip("/")
LIBRE_KEY = str(_settings.get("libretranslate_api_key", "") or "")

HTTP_TIMEOUT = float(_settings.get("translate_timeout_s", 6.0))
# Hard ceiling on the whole online chain. A sighted user watching a spinner
# will wait; someone listening to silence has no way to tell a slow network
# from a dead device, so the chain gives up and says something instead.
ONLINE_BUDGET_S = float(_settings.get("translate_online_budget_s", 14.0))
OFFLINE_BACKOFF_S = float(_settings.get("translate_offline_backoff_s", 30.0))
MAX_CACHE_ENTRIES = int(_settings.get("translate_cache_entries", 2000))

# ──────────────────────────────────────────────────────────────
# SUPPORTED LANGUAGES
# ──────────────────────────────────────────────────────────────
SUPPORTED_LANGS = {
    'en': 'English',
    'hi': 'Hindi',
    'gu': 'Gujarati',
}

# Map settings.json language codes to Google Translate language codes
SETTINGS_TO_GOOGLE = {
    'eng': 'en',
    'hin': 'hi',
    'guj': 'gu',
    'en': 'en',
    'hi': 'hi',
    'gu': 'gu',
}

# ISO code → the code tts.speak() expects, so a caller can hand our result
# straight to the speaker without a second lookup table.
LANG_TO_TTS = {'en': 'eng', 'hi': 'hin', 'gu': 'guj'}

TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"
MYMEMORY_URL = "https://api.mymemory.translated.net/get"

# MyMemory rejects anonymous queries longer than this.
MYMEMORY_MAX_CHARS = 480


# ──────────────────────────────────────────────────────────────
# RESULT TYPE
# ──────────────────────────────────────────────────────────────
@dataclass
class TranslationResult:
    """What came back, and how.

    text        — the best text available to show or speak
    lang        — the language `text` is actually in ('en'/'hi'/'gu').
                  On failure this is the SOURCE language, because `text` is
                  then the untranslated original: speaking it with the target
                  language's voice would be wrong.
    translated  — True only if `text` is a genuine translation
    source      — which link in the chain answered
    error       — short reason when translated is False
    """
    text: str
    lang: str
    translated: bool
    source: str = "none"
    error: Optional[str] = None

    def __str__(self) -> str:       # keeps f"{translate_ex(...)}" sane
        return self.text


# ──────────────────────────────────────────────────────────────
# CONNECTIVITY
# ──────────────────────────────────────────────────────────────
# Probing by raw IP deliberately skips DNS: a Pi on a captive or dead network
# blocks for seconds inside getaddrinfo(), which is exactly the stall this is
# here to avoid. Any one of these answering means "worth trying HTTP".
_NET_PROBES = (("1.1.1.1", 53), ("8.8.8.8", 53), ("1.1.1.1", 443))
_DNS_PROBE_HOST = "translate.googleapis.com"
_NET_PROBE_TIMEOUT = 1.5
_NET_CACHE_TTL = 15.0

_net_lock = threading.Lock()
_net_state: Optional[bool] = None
_net_checked_at = 0.0
_net_failed_at = 0.0


def _with_deadline(fn, seconds: float, *args):
    """Run fn(*args) but never block the caller longer than `seconds`.

    requests' `timeout=` covers connect and read, but NOT name resolution:
    getaddrinfo() is a blocking libc call, and on a Pi whose DNS server has
    gone away it sits there for the resolver's own 30s timeout no matter what
    was passed to requests. That is the stall this whole module exists to
    avoid, so every outbound call gets a real wall-clock ceiling. An abandoned
    thread is a daemon and dies with the process.
    """
    box = {}

    def _run():
        try:
            box["value"] = fn(*args)
        except BaseException as e:      # re-raised on the caller's thread
            box["error"] = e

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeoutError(f"no response within {seconds:.0f}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _probe_network() -> bool:
    # Reachability by raw IP: proves packets get out, without touching DNS.
    reachable = False
    for host, port in _NET_PROBES:
        try:
            with socket.create_connection((host, port), _NET_PROBE_TIMEOUT):
                reachable = True
                break
        except OSError:
            continue
    if not reachable:
        return False

    # DNS is a separate failure: a captive or half-configured network routes
    # packets fine and resolves nothing, which is precisely the state the
    # field logs show. Without this check every provider is tried and each one
    # blocks in getaddrinfo().
    try:
        _with_deadline(socket.getaddrinfo, _NET_PROBE_TIMEOUT * 2,
                       _DNS_PROBE_HOST, 443)
        return True
    except Exception:
        logger.debug("Network reachable but DNS is not resolving.")
        return False


def is_online(force: bool = False) -> bool:
    """True if the network looks usable. Cached for a few seconds.

    Also honours a backoff: after an online provider fails with a network
    error we stop trying all of them for OFFLINE_BACKOFF_S, the same way
    tts.py backs off gTTS, so a long document doesn't pay one timeout per
    sentence.
    """
    global _net_state, _net_checked_at

    with _net_lock:
        now = time.time()
        if not force and (now - _net_failed_at) < OFFLINE_BACKOFF_S:
            return False
        if not force and _net_state is not None and (now - _net_checked_at) < _NET_CACHE_TTL:
            return _net_state

    state = _probe_network()

    with _net_lock:
        _net_state = state
        _net_checked_at = time.time()
    return state


def _mark_network_down(reason: str = "") -> None:
    global _net_state, _net_failed_at
    with _net_lock:
        _net_state = False
        _net_failed_at = time.time()
    logger.warning(
        f"Network unavailable ({reason}); online translation paused for "
        f"{OFFLINE_BACKOFF_S:.0f}s.")


def _is_network_error(exc: Exception) -> bool:
    """True only for "the network is gone", not "the server said no".

    The distinction matters: a dead network means skip every remaining online
    provider, while an HTTP 429 or 503 from one of them means try the next.
    requests.RequestException subclasses OSError, so a bare OSError check
    would quietly swallow every HTTP error into the first category and take
    the whole chain down over one rate-limited endpoint.
    """
    if isinstance(exc, requests.HTTPError):
        return False
    return isinstance(exc, (requests.ConnectionError, requests.Timeout,
                            socket.gaierror, socket.timeout,
                            ConnectionError, TimeoutError))


# ──────────────────────────────────────────────────────────────
# NORMALISATION  (cache + phrasebook keys)
# ──────────────────────────────────────────────────────────────
_TRIM_PUNCT = " \t\n\r.?!,;:।॥"   # includes danda / double danda


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip(_TRIM_PUNCT).lower()


# ──────────────────────────────────────────────────────────────
# CACHE
# ──────────────────────────────────────────────────────────────
_cache_lock = threading.Lock()
_cache: Optional[dict] = None


def _cache_key(text: str, src: str, dest: str) -> str:
    return f"{src}|{dest}|{_normalize(text)}"


def _load_cache() -> dict:
    global _cache
    with _cache_lock:
        if _cache is not None:
            return _cache
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            _cache = loaded if isinstance(loaded, dict) else {}
            logger.info(f"Translation cache loaded ({len(_cache)} entries).")
        except FileNotFoundError:
            _cache = {}
        except Exception as e:
            logger.warning(f"Translation cache unreadable ({e}); starting empty.")
            _cache = {}
        return _cache


def _cache_get(text: str, src: str, dest: str) -> Optional[str]:
    return _load_cache().get(_cache_key(text, src, dest))


def _cache_put(text: str, src: str, dest: str, translated: str) -> None:
    cache = _load_cache()
    with _cache_lock:
        cache[_cache_key(text, src, dest)] = translated
        # dict preserves insertion order, so the oldest keys are simply the
        # first ones. Bounded because this file lives on an SD card.
        if len(cache) > MAX_CACHE_ENTRIES:
            for stale in list(cache)[:len(cache) - MAX_CACHE_ENTRIES]:
                cache.pop(stale, None)
        snapshot = dict(cache)

    # Write via a temp file in the same directory: a power cut mid-write on a
    # Pi must not leave a truncated JSON file that poisons every later run.
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=str(CACHE_PATH.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False)
        os.replace(tmp_path, CACHE_PATH)
        tmp_path = None
    except Exception as e:
        logger.debug(f"Could not persist translation cache: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────
# OFFLINE PHRASEBOOK
# ──────────────────────────────────────────────────────────────
# Last resort, and deliberately exact-match only: a word-by-word substitution
# would produce confident nonsense, which is worse than admitting failure to
# someone who cannot proof-read the output. Keyed on English.
_PHRASEBOOK = {
    # greetings & courtesy
    "hello":              ("नमस्ते", "નમસ્તે"),
    "hi":                 ("नमस्ते", "નમસ્તે"),
    "thank you":          ("धन्यवाद", "આભાર"),
    "thanks":             ("धन्यवाद", "આભાર"),
    "please":             ("कृपया", "કૃપા કરીને"),
    "sorry":              ("माफ़ कीजिए", "માફ કરશો"),
    "yes":                ("हाँ", "હા"),
    "no":                 ("नहीं", "ના"),
    "good morning":       ("सुप्रभात", "સુપ્રભાત"),
    "good night":         ("शुभ रात्रि", "શુભ રાત્રિ"),
    "how are you":        ("आप कैसे हैं", "તમે કેમ છો"),
    "i am fine":          ("मैं ठीक हूँ", "હું ઠીક છું"),
    "what is your name":  ("आपका नाम क्या है", "તમારું નામ શું છે"),
    "my name is":         ("मेरा नाम है", "મારું નામ છે"),
    # needs & safety
    "help":               ("मदद", "મદદ"),
    "water":              ("पानी", "પાણી"),
    "food":               ("खाना", "ખોરાક"),
    "doctor":             ("डॉक्टर", "ડૉક્ટર"),
    "hospital":           ("अस्पताल", "હોસ્પિટલ"),
    "danger":             ("खतरा", "ભય"),
    "careful":            ("सावधान", "સાવધાન"),
    "stop":               ("रुको", "રોકો"),
    "start":              ("शुरू", "શરૂ"),
    "left":               ("बाएँ", "ડાબે"),
    "right":              ("दाएँ", "જમણે"),
    "front":              ("सामने", "આગળ"),
    "back":               ("पीछे", "પાછળ"),
    # people & places
    "home":               ("घर", "ઘર"),
    "house":              ("घर", "ઘર"),
    "mother":             ("माता", "માતા"),
    "father":             ("पिता", "પિતા"),
    "friend":             ("मित्र", "મિત્ર"),
    "teacher":            ("शिक्षक", "શિક્ષક"),
    "student":            ("विद्यार्थी", "વિદ્યાર્થી"),
    "school":             ("विद्यालय", "શાળા"),
    "name":               ("नाम", "નામ"),
    # classroom objects
    "book":               ("किताब", "પુસ્તક"),
    "pen":                ("कलम", "પેન"),
    "paper":              ("कागज़", "કાગળ"),
    "table":              ("मेज़", "ટેબલ"),
    "chair":              ("कुर्सी", "ખુરશી"),
    "door":               ("दरवाज़ा", "દરવાજો"),
    "computer":           ("कंप्यूटर", "કમ્પ્યુટર"),
    # subjects & study verbs
    "science":            ("विज्ञान", "વિજ્ઞાન"),
    "mathematics":        ("गणित", "ગણિત"),
    "maths":              ("गणित", "ગણિત"),
    "english":            ("अंग्रेज़ी", "અંગ્રેજી"),
    "read":               ("पढ़ना", "વાંચવું"),
    "write":              ("लिखना", "લખવું"),
    "listen":             ("सुनना", "સાંભળવું"),
    "speak":              ("बोलना", "બોલવું"),
    "photosynthesis":     ("प्रकाश संश्लेषण", "પ્રકાશસંશ્લેષણ"),
    "what is photosynthesis": ("प्रकाश संश्लेषण क्या है", "પ્રકાશસંશ્લેષણ શું છે"),
    # time
    "time":               ("समय", "સમય"),
    "today":              ("आज", "આજ"),
    "tomorrow":           ("कल", "કાલે"),
    "yesterday":          ("बीता कल", "ગઈકાલે"),
    # numbers
    "one":                ("एक", "એક"),
    "two":                ("दो", "બે"),
    "three":              ("तीन", "ત્રણ"),
    "four":               ("चार", "ચાર"),
    "five":               ("पाँच", "પાંચ"),
    "six":                ("छह", "છ"),
    "seven":              ("सात", "સાત"),
    "eight":              ("आठ", "આઠ"),
    "nine":               ("नौ", "નવ"),
    "ten":                ("दस", "દસ"),
}

# (src, dest) → {normalized source phrase: translation}, built once.
_PHRASE_INDEX: dict = {}


def _build_phrase_index() -> dict:
    if _PHRASE_INDEX:
        return _PHRASE_INDEX

    en_hi, en_gu = {}, {}
    for eng, (hin, guj) in _PHRASEBOOK.items():
        en_hi[_normalize(eng)] = hin
        en_gu[_normalize(eng)] = guj

    def invert(forward: dict) -> dict:
        # First English spelling wins, so "hello" beats "hi" as the reverse of
        # नमस्ते and the user hears the fuller word.
        out = {}
        for eng, other in forward.items():
            out.setdefault(_normalize(other), eng)
        return out

    hi_en, gu_en = invert(en_hi), invert(en_gu)

    # hi↔gu is pivoted through English rather than listed twice.
    hi_gu = {k: en_gu[_normalize(v)] for k, v in hi_en.items() if _normalize(v) in en_gu}
    gu_hi = {k: en_hi[_normalize(v)] for k, v in gu_en.items() if _normalize(v) in en_hi}

    _PHRASE_INDEX.update({
        ("en", "hi"): en_hi, ("en", "gu"): en_gu,
        ("hi", "en"): hi_en, ("gu", "en"): gu_en,
        ("hi", "gu"): hi_gu, ("gu", "hi"): gu_hi,
    })
    return _PHRASE_INDEX


def _phrasebook_lookup(text: str, src: str, dest: str) -> Optional[str]:
    return _build_phrase_index().get((src, dest), {}).get(_normalize(text))


# ──────────────────────────────────────────────────────────────
# PROVIDERS
# ──────────────────────────────────────────────────────────────
def _request_translation(text: str, src: str, dest: str):
    """Raw Google endpoint call. Kept public-ish: detect_language() uses it."""
    params = {
        "client": "gtx",
        "sl": src,
        "tl": dest,
        "dt": "t",
        "q": text,
    }
    response = requests.get(TRANSLATE_URL, params=params, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _provider_google(text: str, src: str, dest: str) -> Optional[str]:
    result = _request_translation(text, src, dest)
    return "".join(
        part[0] for part in result[0]
        if isinstance(part, list) and part and part[0]
    ).strip()


def _provider_libre(text: str, src: str, dest: str) -> Optional[str]:
    if not LIBRE_URL:
        return None
    payload = {"q": text, "source": src, "target": dest, "format": "text"}
    if LIBRE_KEY:
        payload["api_key"] = LIBRE_KEY
    response = requests.post(f"{LIBRE_URL}/translate", json=payload, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    return (response.json().get("translatedText") or "").strip()


def _provider_mymemory(text: str, src: str, dest: str) -> Optional[str]:
    if len(text) > MYMEMORY_MAX_CHARS:
        return None
    response = requests.get(
        MYMEMORY_URL,
        params={"q": text, "langpair": f"{src}|{dest}"},
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    body = response.json()
    if int(body.get("responseStatus", 0)) != 200:
        return None
    out = (body.get("responseData", {}).get("translatedText") or "").strip()
    # MyMemory reports quota and other refusals as ALL-CAPS ASCII prose in the
    # translation field itself, which would otherwise be spoken to the user as
    # if it were Hindi.
    if not out or out.isupper():
        return None
    return out


_argos_ready: Optional[bool] = None


def _argos_available() -> bool:
    global _argos_ready
    if _argos_ready is None:
        try:
            import argostranslate.translate  # noqa: F401
            _argos_ready = True
            logger.info("Argos Translate available — offline translation enabled.")
        except Exception:
            _argos_ready = False
    return _argos_ready


def _provider_argos(text: str, src: str, dest: str) -> Optional[str]:
    """Fully offline neural translation, when the language pair is installed.

    Optional dependency: `pip install argostranslate`, then install the pair
    (e.g. en→hi). Absent, this link is skipped silently.
    """
    if not _argos_available():
        return None
    import argostranslate.translate as at
    out = (at.translate(text, src, dest) or "").strip()
    # Argos returns the input unchanged when the pair isn't installed.
    return None if not out or _normalize(out) == _normalize(text) else out


# name, callable, needs_network
_PROVIDERS: Tuple[Tuple[str, object, bool], ...] = (
    ("google",     _provider_google,   True),
    ("libre",      _provider_libre,    True),
    ("mymemory",   _provider_mymemory, True),
    ("argos",      _provider_argos,    False),
)


# ──────────────────────────────────────────────────────────────
# PUBLIC API — translate_ex()
# ──────────────────────────────────────────────────────────────
def translate_ex(text: str, from_lang: str = 'en', to_lang: str = 'hi') -> TranslationResult:
    """Translate, reporting how it went.

    Never raises and never blocks longer than the chain allows: when the
    device is offline the network providers are skipped, not timed out.
    """
    if not text or not text.strip():
        return TranslationResult("No text to translate.", "en", False,
                                 "none", "empty input")

    src = SETTINGS_TO_GOOGLE.get(from_lang, from_lang)
    dest = SETTINGS_TO_GOOGLE.get(to_lang, to_lang)

    if src not in SUPPORTED_LANGS:
        logger.warning(f"Unsupported source language: {from_lang}")
        return TranslationResult(text, "en", False, "none",
                                 f"unsupported source language: {from_lang}")

    if dest not in SUPPORTED_LANGS:
        logger.warning(f"Unsupported target language: {to_lang}")
        return TranslationResult(text, src, False, "none",
                                 f"unsupported target language: {to_lang}")

    if src == dest:
        logger.info("Source and target language are the same.")
        return TranslationResult(text, src, True, "identity")

    text = text.strip()
    logger.info(
        f"Translating from {SUPPORTED_LANGS[src]} to "
        f"{SUPPORTED_LANGS[dest]}: \"{text[:80]}\""
    )

    # 1 — cache: instant, and the only thing that makes a repeated phrase work
    #     on a device that has since gone offline.
    cached = _cache_get(text, src, dest)
    if cached:
        logger.info(f"Cache hit → \"{cached[:80]}\"")
        return TranslationResult(cached, dest, True, "cache")

    # 2 — provider chain
    online = is_online()
    if not online:
        logger.info("Device is offline; trying offline providers only.")

    last_error = None
    budget_ends = time.time() + ONLINE_BUDGET_S

    for name, provider, needs_network in _PROVIDERS:
        if needs_network:
            if not online:
                continue
            remaining = budget_ends - time.time()
            if remaining <= 1.0:
                logger.warning(
                    f"Online translation budget of {ONLINE_BUDGET_S:.0f}s spent; "
                    f"skipping {name}.")
                online = False
                continue

        try:
            if needs_network:
                out = _with_deadline(provider, min(remaining, HTTP_TIMEOUT + 2.0),
                                     text, src, dest)
            else:
                out = provider(text, src, dest)
        except Exception as e:
            last_error = f"{name}: {e}"
            if _is_network_error(e):
                logger.warning(f"{name} unreachable: {e}")
                _mark_network_down(name)
                online = False          # don't retry the rest of the network links
            else:
                logger.warning(f"{name} failed: {e}")
            continue

        if out:
            logger.info(f"Translated via {name} → \"{out[:80]}\"")
            _cache_put(text, src, dest, out)
            return TranslationResult(out, dest, True, name)

    # 3 — phrasebook: exact matches only, so a hit is trustworthy.
    phrase = _phrasebook_lookup(text, src, dest)
    if phrase:
        logger.info(f"Translated from offline phrasebook → \"{phrase}\"")
        return TranslationResult(phrase, dest, True, "phrasebook")

    reason = last_error or ("no network and no offline translator installed"
                            if not online else "all providers returned nothing")
    logger.error(f"Translation failed ({reason}).")
    # `text` is still in the SOURCE language — say so, so the caller speaks it
    # with the right voice instead of reading Gujarati in an English accent.
    return TranslationResult(text, src, False, "none", reason)


# ──────────────────────────────────────────────────────────────
# PUBLIC API — translate()
# ──────────────────────────────────────────────────────────────
def translate(text: str, from_lang: str = 'en', to_lang: str = 'hi') -> str:
    """Translate text between supported languages.

    Args:
        text: The text to translate.
        from_lang: Source language code ('en', 'hi', 'gu', 'eng', 'hin', 'guj').
        to_lang: Target language code ('en', 'hi', 'gu', 'eng', 'hin', 'guj').

    Returns:
        Translated text, or the original text with an error message if
        translation failed. Prefer translate_ex() when the caller can act on
        the difference.
    """
    result = translate_ex(text, from_lang, to_lang)
    if result.translated:
        return result.text
    if result.error == "empty input":
        return result.text
    if result.error and result.error.startswith("unsupported"):
        # Capitalise the reason the old API used to return verbatim.
        return result.error[0].upper() + result.error[1:]
    return f"Translation failed. Original text: {result.text}"


# ──────────────────────────────────────────────────────────────
# PUBLIC API — detect_language()
# ──────────────────────────────────────────────────────────────
# Hindi is written in Devanagari and Gujarati in the Gujarati block; neither
# shares a script with English or with each other. For these three languages
# the script *is* the answer, so detection works with no network at all.
_SCRIPT_RANGES = (
    ("hi", 0x0900, 0x097F),
    ("gu", 0x0A80, 0x0AFF),
)


def detect_language(text: str) -> Optional[str]:
    """
    Detect the language of given text.

    Returns:
        Language code ('en', 'hi', 'gu') or None.
    """
    if not text or not text.strip():
        return None

    counts = {"hi": 0, "gu": 0, "en": 0}
    for ch in text:
        cp = ord(ch)
        for code, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[code] += 1
                break
        else:
            if ch.isalpha() and cp < 0x0250:
                counts["en"] += 1

    if any(counts.values()):
        lang = max(counts, key=counts.get)
        logger.info(f"Detected language: {lang} (by script)")
        return lang

    # Digits/punctuation only — ask the network if there is one.
    if not is_online():
        return None
    try:
        result = _request_translation(text, "auto", "en")
        lang = result[2] if len(result) > 2 else None
        if lang:
            logger.info(f"Detected language: {lang}")
            return lang
    except Exception as e:
        if _is_network_error(e):
            _mark_network_down("detect_language")
        logger.error(f"Language detection failed: {e}")
    return None


# ──────────────────────────────────────────────────────────────
# SIGNAL HANDLER
# ──────────────────────────────────────────────────────────────
def signal_handler(sig, frame):
    logger.info("Shutting down Translator module.")
    sys.exit(0)


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("\n" + "=" * 50)
    print("   BlindAssist Translation Module — Test")
    print("   Project: CSR-DES-INFINEON-2025")
    print("=" * 50)
    print("Supported Languages:")
    print(" 1. English (en)")
    print(" 2. Hindi (hi)")
    print(" 3. Gujarati (gu)")
    print(f"Network      : {'online' if is_online() else 'OFFLINE'}")
    print(f"Offline MT   : {'Argos ready' if _argos_available() else 'not installed'}")
    print(f"Cached pairs : {len(_load_cache())}")
    print(f"LibreTranslate: {LIBRE_URL or 'not configured'}")
    print("Type QUIT at any time to exit.")
    print("=" * 50 + "\n")

    lang_map = {'1': 'en', '2': 'hi', '3': 'gu'}

    while True:
        try:
            # --- NEW 3-OPTION MENU MENU ---
            print("\nTranslation mode. Select input method:")
            print(" 1: Type text manually")
            print(" 2: Enter via Morse code")
            print(" 3: Speak text (Voice input)")
            choice = input("Input method (1/2/3) or QUIT: ").strip().lower()

            if choice == "quit":
                break

            text = ""

            if choice == '1':
                text = input("Text to translate: ").strip()

            elif choice == '2':
                # Placeholder for Morse input logic testing
                print("Morse input selected.")
                text = input("Enter translated morse text here: ").strip()

            elif choice == '3':
                if listen:
                    print("\nStarting voice input...")
                    # Calls the manual listen function that waits for ENTER
                    captured_text = listen('en-IN')
                    if captured_text:
                        print(f"\nCaptured text: {captured_text}")
                        text = captured_text
                    else:
                        print("\nFailed to capture voice. Returning to menu.")
                        continue
                else:
                    print("\nVoice module (modules.voice) not found! Cannot use Option 3.")
                    continue
            else:
                print("Invalid choice. Please select 1, 2, or 3.")
                continue

            # Ensure we actually have text before proceeding
            if not text or text.upper() == "QUIT":
                continue

            # --- LANGUAGE SELECTION ---
            detected = detect_language(text)
            if detected:
                print(f"(detected source language: {SUPPORTED_LANGS.get(detected, detected)})")

            src = input("From language (1=EN, 2=HI, 3=GU): ").strip()
            dest = input("To language   (1=EN, 2=HI, 3=GU): ").strip()

            src_code = lang_map.get(src, detected or 'en')
            dest_code = lang_map.get(dest, 'hi')

            # --- RUN TRANSLATION ---
            result = translate_ex(text, src_code, dest_code)
            if result.translated:
                print(f"\n>>> Translation Result: {result.text}")
                print(f"    (via {result.source})\n")
            else:
                print(f"\n>>> Could not translate: {result.error}")
                print(f"    Original text: {result.text}\n")

        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nExiting Translator Test...")
            break

    print("Translator Module Closed.")
