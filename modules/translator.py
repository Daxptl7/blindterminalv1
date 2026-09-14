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

    validated cache → IndicTrans2 → Argos → LibreTranslate → Google
                    → MyMemory → phrasebook

The three network links are skipped outright — not attempted and timed out —
whenever the device is known to be offline, which is what turned a failed
translation into a 10-second stall for a user who cannot see a progress bar.
Everything that succeeds online is written to an on-disk cache, so a phrase
translated once keeps working after the network goes away.

Callers that need to know *how* the answer was produced (to speak it in the
right voice, or to admit that no translation happened) should use
translate_ex(); translate() keeps the original string-in/string-out contract.
Pass ``privacy=True`` to forbid cloud providers and persistent text caching.
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
import ipaddress

import requests

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse

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

# Local engines are intentionally first: translation should remain private,
# predictable and available when the network disappears.  Deployments can
# override this with either a JSON list or a comma-separated string.
_DEFAULT_PROVIDER_ORDER = "indictrans2,argos,libre,google,mymemory"
_configured_order = _settings.get("translate_provider_order", _DEFAULT_PROVIDER_ORDER)
if isinstance(_configured_order, str):
    PROVIDER_ORDER = tuple(
        item.strip().lower() for item in _configured_order.split(",") if item.strip()
    )
else:
    PROVIDER_ORDER = tuple(str(item).strip().lower() for item in _configured_order)

PROVIDER_BACKOFF_S = float(_settings.get("translate_provider_backoff_s", 30.0))
PROVIDER_FAILURE_THRESHOLD = max(
    1, int(_settings.get("translate_provider_failure_threshold", 1))
)
CHUNK_MAX_CHARS = max(100, int(_settings.get("translate_chunk_max_chars", 600)))
VALIDATION_MIN_COVERAGE = float(
    _settings.get("translate_validation_min_coverage", 0.35)
)
VALIDATION_MAX_COVERAGE = float(
    _settings.get("translate_validation_max_coverage", 3.0)
)
VALIDATION_MIN_TARGET_SCRIPT = float(
    _settings.get("translate_validation_min_target_script", 0.50)
)
DEFAULT_PRIVACY = bool(_settings.get("translate_privacy_default", False))
LIBRE_TRUSTED_LOCAL = _settings.get("libretranslate_trusted_local")

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
    model: Optional[str] = None
    validation: Optional[dict] = None

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


def _split_long_piece(text: str, limit: int) -> List[str]:
    """Split a sentence without cutting words unless one word exceeds limit."""
    words = re.findall(r"\S+", text)
    if not words:
        return []
    chunks: List[str] = []
    current = ""
    for word in words:
        if len(word) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(word[i:i + limit] for i in range(0, len(word), limit))
            continue
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > limit:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _sentence_chunks(text: str, limit: int = CHUNK_MAX_CHARS) -> List[Tuple[str, str]]:
    """Return ``(text, separator_after)`` chunks in document reading order.

    Sentence punctuation and paragraph separators are retained.  The provider
    never receives a fragment larger than ``limit``, which avoids URL limits
    and model truncation without blindly slicing through words.
    """
    limit = max(20, int(limit))
    chunks: List[Tuple[str, str]] = []
    parts = re.split(r"(\n+)", text.strip())
    for index in range(0, len(parts), 2):
        paragraph = parts[index].strip()
        newline = parts[index + 1] if index + 1 < len(parts) else ""
        if not paragraph:
            if chunks and newline:
                prior_text, prior_sep = chunks[-1]
                chunks[-1] = (prior_text, prior_sep + newline)
            continue

        sentences = [
            item.strip() for item in re.findall(
                r"[^.!?।॥]+(?:[.!?।॥]+|$)", paragraph
            ) if item.strip()
        ] or [paragraph]
        paragraph_chunks: List[str] = []
        current = ""
        for sentence in sentences:
            for piece in _split_long_piece(sentence, limit):
                candidate = f"{current} {piece}".strip()
                if current and len(candidate) > limit:
                    paragraph_chunks.append(current)
                    current = piece
                else:
                    current = candidate
        if current:
            paragraph_chunks.append(current)

        for pos, item in enumerate(paragraph_chunks):
            separator = " " if pos < len(paragraph_chunks) - 1 else newline
            chunks.append((item, separator))
    return chunks


_DATE_RE = re.compile(
    r"(?<!\d)(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4})(?!\d)"
)
_NUMBER_RE = re.compile(r"(?<![\w])[-+]?\d+(?:[.,]\d+)?%?(?![\w])")
_UNIT_RE = re.compile(
    r"(?i)(?<![\w])(?:kg|g|mg|km|m|cm|mm|l|ml|°c|°f|rs|inr|₹)(?![\w])"
)


def _protected_tokens(text: str) -> Counter:
    """Tokens whose accidental mutation can make a translation dangerous."""
    dates = _DATE_RE.findall(text)
    without_dates = _DATE_RE.sub(" ", text)
    numbers = _NUMBER_RE.findall(without_dates)
    units = [unit.lower() for unit in _UNIT_RE.findall(text)]
    return Counter(dates + numbers + units)


def _script_counts(text: str) -> Dict[str, int]:
    counts = {"en": 0, "hi": 0, "gu": 0, "other": 0}
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            counts["hi"] += 1
        elif 0x0A80 <= cp <= 0x0AFF:
            counts["gu"] += 1
        elif ch.isalpha() and cp < 0x0250:
            counts["en"] += 1
        elif ch.isalpha():
            counts["other"] += 1
    return counts


def validate_translation(source: str, translated: str, src: str,
                         dest: str) -> Tuple[bool, dict]:
    """Apply deterministic safety checks before speech or persistent cache.

    This does not claim semantic correctness.  It catches high-impact failure
    modes that can be measured without a second translation service: wrong
    script, unchanged text, missing numbers/dates/units, and severe truncation.
    """
    reasons: List[str] = []
    source = (source or "").strip()
    translated = (translated or "").strip()
    if not translated:
        reasons.append("empty output")

    source_letters = sum(1 for char in source if char.isalpha())
    target_letters = sum(1 for char in translated if char.isalpha())
    coverage = target_letters / max(1, source_letters)
    if source_letters >= 8 and coverage < VALIDATION_MIN_COVERAGE:
        reasons.append(f"output coverage too low ({coverage:.2f})")
    if source_letters >= 8 and coverage > VALIDATION_MAX_COVERAGE:
        reasons.append(f"output coverage too high ({coverage:.2f})")

    unchanged = _normalize(source) == _normalize(translated)
    if src != dest and unchanged and source_letters >= 3:
        reasons.append("output is unchanged")

    scripts = _script_counts(translated)
    if dest in ("hi", "gu") and target_letters >= 4:
        target_ratio = scripts[dest] / max(1, target_letters)
        if scripts[dest] == 0 or target_ratio < VALIDATION_MIN_TARGET_SCRIPT:
            reasons.append(
                f"wrong target script ({dest} ratio {target_ratio:.2f})"
            )
    elif dest == "en" and target_letters >= 4:
        target_ratio = scripts["en"] / max(1, target_letters)
        if scripts["en"] == 0 or target_ratio < VALIDATION_MIN_TARGET_SCRIPT:
            reasons.append(
                f"wrong target script (en ratio {target_ratio:.2f})"
            )
    else:
        target_ratio = 1.0

    expected_tokens = _protected_tokens(source)
    actual_tokens = _protected_tokens(translated)
    missing_tokens = list((expected_tokens - actual_tokens).elements())
    if missing_tokens:
        reasons.append("missing protected tokens: " + ", ".join(missing_tokens[:8]))

    report = {
        "passed": not reasons,
        "reasons": reasons,
        "coverage": round(coverage, 4),
        "target_script_ratio": round(target_ratio, 4),
        "protected_tokens_preserved": not missing_tokens,
    }
    return not reasons, report


# ──────────────────────────────────────────────────────────────
# CACHE
# ──────────────────────────────────────────────────────────────
_cache_lock = threading.Lock()
_cache: Optional[dict] = None
_CACHE_SCHEMA_VERSION = 2


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
            if (isinstance(loaded, dict)
                    and loaded.get("schema_version") == _CACHE_SCHEMA_VERSION
                    and isinstance(loaded.get("entries"), dict)):
                _cache = loaded
            elif isinstance(loaded, dict):
                # Legacy cache: values were plain strings.  Preserve them in
                # memory and revalidate on use; the next successful write
                # atomically upgrades the file to the versioned schema.
                entries = {}
                for key, value in loaded.items():
                    if isinstance(value, str) and value.strip():
                        parts = key.split("|", 2)
                        entries[key] = {
                            "translation": value,
                            "src": parts[0] if len(parts) > 0 else None,
                            "dest": parts[1] if len(parts) > 1 else None,
                            "provider": "legacy-cache",
                            "model": None,
                            "validated": False,
                            "validation": None,
                            "created_at": None,
                        }
                _cache = {"schema_version": _CACHE_SCHEMA_VERSION,
                          "entries": entries}
            else:
                _cache = {"schema_version": _CACHE_SCHEMA_VERSION, "entries": {}}
            logger.info(
                f"Translation cache loaded ({len(_cache['entries'])} entries)."
            )
        except FileNotFoundError:
            _cache = {"schema_version": _CACHE_SCHEMA_VERSION, "entries": {}}
        except Exception as e:
            logger.warning(f"Translation cache unreadable ({e}); starting empty.")
            _cache = {"schema_version": _CACHE_SCHEMA_VERSION, "entries": {}}
        return _cache


def _cache_get(text: str, src: str, dest: str) -> Optional[str]:
    record = _cache_get_record(text, src, dest)
    return record.get("translation") if record else None


def _cache_get_record(text: str, src: str, dest: str) -> Optional[dict]:
    record = _load_cache().get("entries", {}).get(_cache_key(text, src, dest))
    if isinstance(record, str):  # defensive compatibility with injected caches
        record = {"translation": record, "provider": "legacy-cache",
                  "model": None, "validated": False, "validation": None}
    if not isinstance(record, dict):
        return None
    translated = str(record.get("translation", "") or "").strip()
    if not translated:
        return None
    passed, validation = validate_translation(text, translated, src, dest)
    if not passed:
        logger.warning("Rejected an invalid translation cache entry.")
        return None
    result = dict(record)
    result["translation"] = translated
    result["validation"] = validation
    result["validated"] = True
    return result


def _cache_put(text: str, src: str, dest: str, translated: str,
               provider: str = "unknown", model: Optional[str] = None,
               validation: Optional[dict] = None) -> None:
    cache = _load_cache()
    with _cache_lock:
        entries = cache.setdefault("entries", {})
        entries[_cache_key(text, src, dest)] = {
            "translation": translated,
            "src": src,
            "dest": dest,
            "provider": provider,
            "model": model,
            "validated": bool(validation and validation.get("passed")),
            "validation": validation,
            "created_at": int(time.time()),
        }
        # dict preserves insertion order, so the oldest keys are simply the
        # first ones. Bounded because this file lives on an SD card.
        if len(entries) > MAX_CACHE_ENTRIES:
            for stale in list(entries)[:len(entries) - MAX_CACHE_ENTRIES]:
                entries.pop(stale, None)
        snapshot = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "entries": dict(entries),
        }

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
    try:
        language = at.get_from_code(src)
        translation = language.get_translation(at.get_from_code(dest)) if language else None
    except Exception:
        translation = None
    if translation is None:
        return None
    out = (translation.translate(text) or "").strip()
    # Argos returns the input unchanged when the pair isn't installed.
    return None if not out or _normalize(out) == _normalize(text) else out


_indictrans2_engine = None
_indictrans2_lock = threading.Lock()


def _get_indictrans2_engine():
    global _indictrans2_engine
    if _indictrans2_engine is None:
        with _indictrans2_lock:
            if _indictrans2_engine is None:
                try:
                    from modules.indictrans2_engine import IndicTrans2Engine
                except ImportError:
                    from indictrans2_engine import IndicTrans2Engine
                _indictrans2_engine = IndicTrans2Engine(_settings, BASE_DIR)
    return _indictrans2_engine


def _provider_indictrans2(text: str, src: str, dest: str) -> Optional[str]:
    return _get_indictrans2_engine().translate(text, src, dest)


def _libre_is_trusted_local() -> bool:
    """True when Libre is explicitly trusted or clearly local/private."""
    if LIBRE_TRUSTED_LOCAL is not None:
        return bool(LIBRE_TRUSTED_LOCAL)
    if not LIBRE_URL:
        return False
    hostname = (urlparse(LIBRE_URL).hostname or "").lower().strip("[]")
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        return True
    if hostname and "." not in hostname:
        return True  # conventional single-label classroom LAN hostname
    try:
        address = ipaddress.ip_address(hostname)
        return address.is_private or address.is_loopback or address.is_link_local
    except ValueError:
        return False


_provider_state_lock = threading.Lock()
_provider_failures: Dict[str, int] = {}
_provider_blocked_until: Dict[str, float] = {}


def _provider_blocked(name: str) -> bool:
    with _provider_state_lock:
        return time.time() < _provider_blocked_until.get(name, 0.0)


def _provider_succeeded(name: str) -> None:
    with _provider_state_lock:
        _provider_failures.pop(name, None)
        _provider_blocked_until.pop(name, None)


def _provider_failed(name: str) -> None:
    with _provider_state_lock:
        failures = _provider_failures.get(name, 0) + 1
        _provider_failures[name] = failures
        if failures >= PROVIDER_FAILURE_THRESHOLD:
            _provider_blocked_until[name] = time.time() + PROVIDER_BACKOFF_S
            _provider_failures[name] = 0


def _provider_model_id(name: str, src: str, dest: str) -> Optional[str]:
    if name == "indictrans2":
        return _get_indictrans2_engine().model_id(src, dest)
    if name == "google":
        return "google-web-gtx"
    if name == "mymemory":
        return "mymemory-api"
    if name == "libre":
        return LIBRE_URL or None
    return None


# name, callable, needs_network
_PROVIDERS: Tuple[Tuple[str, object, bool], ...] = (
    ("indictrans2", _provider_indictrans2, False),
    ("argos",      _provider_argos,    False),
    # A configured Libre server may be localhost or LAN-only.  It has its own
    # reachability/circuit-breaker and must not depend on public internet.
    ("libre",      _provider_libre,    False),
    ("google",     _provider_google,   True),
    ("mymemory",   _provider_mymemory, True),
)

_PROVIDER_MAP = {name: (provider, needs_network)
                 for name, provider, needs_network in _PROVIDERS}


def _ordered_providers() -> Tuple[Tuple[str, object, bool], ...]:
    ordered = []
    seen = set()
    for name in PROVIDER_ORDER:
        if name in seen or name not in _PROVIDER_MAP:
            continue
        provider, needs_network = _PROVIDER_MAP[name]
        ordered.append((name, provider, needs_network))
        seen.add(name)
    return tuple(ordered)


def _provider_chunk_limit(name: str) -> int:
    configured = _settings.get(f"translate_{name}_chunk_chars")
    if configured is not None:
        return max(100, int(configured))
    if name == "mymemory":
        return MYMEMORY_MAX_CHARS
    if name in ("indictrans2", "argos"):
        return CHUNK_MAX_CHARS
    return max(CHUNK_MAX_CHARS, 2000)


def _call_provider_chunked(name: str, provider, text: str,
                           src: str, dest: str, deadline: float) -> Optional[str]:
    rendered: List[str] = []
    chunks = _sentence_chunks(text, _provider_chunk_limit(name))
    for chunk, separator in chunks:
        if name in ("google", "libre", "mymemory"):
            remaining = deadline - time.time()
            if remaining <= 0.25:
                raise TimeoutError(f"{name} translation budget exhausted")
            out = _with_deadline(
                provider, min(remaining, HTTP_TIMEOUT + 2.0), chunk, src, dest
            )
        else:
            out = provider(chunk, src, dest)
        if not out:
            return None
        rendered.append(out.strip())
        rendered.append(separator)
    return "".join(rendered).strip() or None


# ──────────────────────────────────────────────────────────────
# PUBLIC API — translate_ex()
# ──────────────────────────────────────────────────────────────
def translate_ex(text: str, from_lang: str = 'en', to_lang: str = 'hi',
                 *, privacy: Optional[bool] = None) -> TranslationResult:
    """Translate, reporting how it went.

    Never raises and never blocks longer than the chain allows: when the
    device is offline the network providers are skipped, not timed out.
    """
    private = DEFAULT_PRIVACY if privacy is None else bool(privacy)

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
        return TranslationResult(
            text, src, True, "identity", validation={"passed": True,
                                                       "reasons": []}
        )

    text = text.strip()
    if private:
        logger.info(
            f"Private translation from {SUPPORTED_LANGS[src]} to "
            f"{SUPPORTED_LANGS[dest]} ({len(text)} characters)."
        )
    else:
        logger.info(
            f"Translating from {SUPPORTED_LANGS[src]} to "
            f"{SUPPORTED_LANGS[dest]}: \"{text[:80]}\""
        )

    # 1 — cache: instant, and the only thing that makes a repeated phrase work
    #     on a device that has since gone offline.
    # Private requests never touch persistent text storage in either direction.
    if not private:
        cached = _cache_get_record(text, src, dest)
        if cached:
            translated = cached["translation"]
            logger.info(f"Cache hit → \"{translated[:80]}\"")
            return TranslationResult(
                translated, dest, True, "cache", model=cached.get("model"),
                validation=cached.get("validation")
            )

    # 2 — provider chain
    errors: List[str] = []
    public_online: Optional[bool] = None
    cloud_budget_ends: Optional[float] = None

    for name, provider, needs_network in _ordered_providers():
        is_cloud = name in ("google", "mymemory") or (
            name == "libre" and not _libre_is_trusted_local()
        )
        if private and is_cloud:
            errors.append(f"{name}: skipped by privacy policy")
            continue
        if name == "libre" and not LIBRE_URL:
            errors.append("libre: not configured")
            continue
        if _provider_blocked(name):
            errors.append(f"{name}: temporarily paused after failures")
            continue

        if needs_network:
            if public_online is None:
                public_online = is_online()
                if not public_online:
                    logger.info("Public internet is offline; cloud providers skipped.")
            if not public_online:
                errors.append(f"{name}: public internet unavailable")
                continue
            if cloud_budget_ends is None:
                cloud_budget_ends = time.time() + ONLINE_BUDGET_S
            if time.time() >= cloud_budget_ends:
                errors.append(f"{name}: online translation budget exhausted")
                continue

        if name == "libre":
            # Libre may be reachable only over localhost/LAN.  Give it an
            # independent deadline even when the public internet probe fails.
            provider_deadline = time.time() + HTTP_TIMEOUT + 2.0
        elif needs_network:
            provider_deadline = cloud_budget_ends or (time.time() + HTTP_TIMEOUT + 2.0)
        else:
            provider_deadline = time.time() + max(HTTP_TIMEOUT + 2.0, 60.0)

        try:
            out = _call_provider_chunked(
                name, provider, text, src, dest, provider_deadline
            )
        except Exception as e:
            error = f"{name}: {e}"
            errors.append(error)
            _provider_failed(name)
            logger.warning(f"{name} failed: {e}")
            continue

        if out:
            passed, validation = validate_translation(text, out, src, dest)
            if not passed:
                reason = "; ".join(validation["reasons"])
                errors.append(f"{name}: validation rejected output ({reason})")
                _provider_failed(name)
                logger.warning(f"Rejected {name} translation: {reason}")
                continue

            _provider_succeeded(name)
            model = _provider_model_id(name, src, dest)
            if private:
                logger.info(f"Private translation completed via {name}.")
            else:
                logger.info(f"Translated via {name} → \"{out[:80]}\"")
                _cache_put(text, src, dest, out, provider=name, model=model,
                           validation=validation)
            return TranslationResult(
                out, dest, True, name, model=model, validation=validation
            )
        if name == "indictrans2":
            diag = _get_indictrans2_engine().diagnostics()
            bundle = _get_indictrans2_engine()._bundle_for(src, dest)
            detail = diag.get("bundles", {}).get(bundle or "", {}).get("error")
            errors.append(f"indictrans2: {detail or 'model unavailable'}")
        elif name == "argos":
            errors.append("argos: package or requested language pair unavailable")

    # 3 — phrasebook: exact matches only, so a hit is trustworthy.
    phrase = _phrasebook_lookup(text, src, dest)
    if phrase:
        passed, validation = validate_translation(text, phrase, src, dest)
        if passed:
            if private:
                logger.info("Private translation completed from offline phrasebook.")
            else:
                logger.info(f"Translated from offline phrasebook → \"{phrase}\"")
            return TranslationResult(
                phrase, dest, True, "phrasebook", model="builtin-v1",
                validation=validation
            )
        errors.append("phrasebook: validation rejected output")

    reason = "; ".join(errors[-5:]) or "all providers returned nothing"
    logger.error(f"Translation failed ({reason}).")
    # `text` is still in the SOURCE language — say so, so the caller speaks it
    # with the right voice instead of reading Gujarati in an English accent.
    return TranslationResult(text, src, False, "none", reason)


# ──────────────────────────────────────────────────────────────
# PUBLIC API — translate()
# ──────────────────────────────────────────────────────────────
def translate(text: str, from_lang: str = 'en', to_lang: str = 'hi',
              *, privacy: Optional[bool] = None) -> str:
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
    result = translate_ex(text, from_lang, to_lang, privacy=privacy)
    if result.translated:
        return result.text
    if result.error == "empty input":
        return result.text
    if result.error and result.error.startswith("unsupported"):
        # Capitalise the reason the old API used to return verbatim.
        return result.error[0].upper() + result.error[1:]
    return f"Translation failed. Original text: {result.text}"


def diagnostics() -> dict:
    """Return translation readiness without sending text to any provider."""
    indic = _get_indictrans2_engine().diagnostics()
    cache = _load_cache()
    bundles = indic.get("bundles", {})
    return {
        "supported_languages": dict(SUPPORTED_LANGS),
        "supported_pairs": [
            f"{src}->{dest}"
            for src in SUPPORTED_LANGS
            for dest in SUPPORTED_LANGS
            if src != dest
        ],
        "provider_order": list(PROVIDER_ORDER),
        "indictrans2": indic,
        "indictrans2_all_pairs_ready": all(
            info.get("available") for info in bundles.values()
        ),
        "argos_runtime": _argos_available(),
        "libre_url": LIBRE_URL or None,
        "libre_trusted_local": _libre_is_trusted_local(),
        "cache_schema": cache.get("schema_version"),
        "cache_entries": len(cache.get("entries", {})),
        "privacy_default": DEFAULT_PRIVACY,
        "chunk_max_chars": CHUNK_MAX_CHARS,
    }


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
