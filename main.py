"""
main.py — BlindAssist Project
Production-ready orchestrator with graceful degradation.
"""

import sys
import signal
import logging
import json
import threading
import time

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ── Ensure Python can find modules/ and services/ from any CWD ──
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

LOG_PATH = BASE_DIR / "logs" / "main.log"
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logfire_handler = None
try:
    import os
    import json
    token = None
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r') as f:
                config_data = json.load(f)
                token = config_data.get("logfire_token")
        except Exception:
            pass
    if token:
        os.environ["LOGFIRE_TOKEN"] = token

    import logfire
    logfire.configure(send_to_logfire='if-token-present')
    logfire_handler = logfire.LogfireLoggingHandler()
except Exception as e:
    pass

logging_handlers = [
    logging.FileHandler(LOG_PATH),
    logging.StreamHandler(sys.stdout)
]
if logfire_handler:
    logging_handlers.append(logfire_handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=logging_handlers
)
logger = logging.getLogger("MainController")
_shutdown_done = False


def _load_settings() -> dict:
    """Orchestrator-level settings. Modules load their own; this is only for
    behaviour main.py itself controls (mode time limits, etc.)."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read settings.json ({e}); using defaults.")
        return {}


_settings = _load_settings()

# ── SAFE IMPORTS ────────────────────────────────────────────
# Each module is optional — app starts even if some are missing

_modules = {}

def _safe_import(module_name, alias):
    """Import a module, log if missing, but don't crash."""
    try:
        mod = __import__(f"modules.{module_name}", fromlist=[alias])
        _modules[alias] = getattr(mod, alias) if hasattr(mod, alias) else mod
        logger.info(f"Module loaded: {module_name}")
        return True
    except Exception as e:
        logger.warning(f"Module unavailable: {module_name} — {e}")
        _modules[alias] = None
        return False

# Core modules (app works without these but degraded)
_safe_import("tts", "tts")
_safe_import("voice", "voice")
_safe_import("ai_query", "ai_query")
_safe_import("morse", "morse")
_safe_import("ocr", "ocr")
_safe_import("translator", "translator")
_safe_import("math_solver", "mathsolver")

# Optional modules (app works fine without)
_safe_import("gesture_control", "gesture")
_safe_import("emotion_engine", "emotion")
_safe_import("gps_navigator", "gps")
_safe_import("confidential_mode", "privacy")
_safe_import("object_detection", "objdetect")

# NEW — bridge to the physical Pico W Morse buttons over USB serial.
# Optional: if no Pico W is plugged in (e.g. testing on a laptop),
# this simply stays unavailable and Mode 2 falls back to keyboard typing.
_morse_serial_singleton = None
try:
    from modules.morse_serial import MorseSerial
    _morse_serial_singleton = MorseSerial()
    logger.info("Pico W Morse buttons connected.")
except Exception as e:
    logger.warning(f"Pico W Morse buttons unavailable, using keyboard fallback: {e}")
    _morse_serial_singleton = None

# Hand the Pico connection to voice.py so it reads buttons through this one
# reader instead of opening /dev/ttyACM0 a second time. Two readers on the
# same port steal messages from each other's queue — and voice.py's own
# attempt would simply fail here, because main.py already holds the port.
# Without this, voice.py's confidentiality prompt can never see Button 1 or 2
# and every answer would fall through to its timeout default.
if _morse_serial_singleton is not None and _modules.get("voice") is not None:
    try:
        _modules["voice"].set_morse_serial(_morse_serial_singleton)
    except Exception as e:
        logger.warning(f"Could not share the button connection with voice.py: {e}")

# Helper to check if module is available
def _has(mod_name):
    return _modules.get(mod_name) is not None

def _speak(text, block=False, lang="eng"):
    """Safe TTS — works even if tts module failed to load.

    `lang` picks the voice. It matters for exactly one thing today — reading a
    translation aloud — but it has to be threaded through here, because the
    default English voice reads Devanagari and Gujarati script as either
    silence or nonsense.
    """
    if _has("tts"):
        _modules["tts"].speak(text, lang=lang, block=block)
    else:
        print(f"[TTS OFFLINE] {text}")


def _flush_speech():
    """Drop queued-but-unspoken audio (e.g. stale 'still working…' messages)."""
    if _has("tts"):
        try:
            _modules["tts"].flush()
        except Exception as e:
            logger.debug(f"TTS flush failed: {e}")


def _listen(lang: str = "en-IN", announce: bool = False):
    """Capture one utterance. Returns text or None, never raises.

    Every voice call site used to swallow exceptions and then say the same
    "I didn't catch that" regardless of cause — so a missing microphone, a
    dead internet connection and genuine mishearing were indistinguishable to
    a user who cannot read the logs. voice.get_last_error() gives a reason
    written to be spoken aloud; _speak_voice_failure() delivers it.
    """
    if not _has("voice"):
        return None

    # Let the speaker finish before opening the microphone. tts.speak() is a
    # non-blocking queue, so every "Ask your question now" prompt was still
    # playing when capture started — the device recorded its own voice, the
    # VAD triggered on it, and the AI was asked to answer the prompt it had
    # just spoken. This is why voice input appeared to "hear the wrong thing"
    # even when the microphone was working perfectly.
    if _has("tts"):
        try:
            _modules["tts"].wait_until_idle(timeout=15)
        except Exception as e:
            logger.debug(f"Could not wait for TTS to drain: {e}")

    try:
        return _modules["voice"].listen(lang, speak_fn=_speak if announce else None)
    except Exception as e:
        logger.warning(f"Voice capture error: {e}")
        return None


# How many presses of Button 3 end an open-ended recording, and how long the
# user may take between them. Three presses (rather than one) so that a stray
# knock against the button cannot cut a question short.
_STOP_PRESSES = int(_settings.get("voice_stop_press_count", 3))
_STOP_PRESS_WINDOW_S = float(_settings.get("voice_stop_press_window_seconds", 4.0))


def _button3_stop_signal():
    """Watch Button 3 in the background; return (stop_check, cancel).

    stop_check() is polled by voice.listen() and returns True once Button 3 has
    been pressed _STOP_PRESSES times in a row, each within
    _STOP_PRESS_WINDOW_S of the last. cancel() stops the watcher thread.

    Only presses of Button 3 count. The Pico firmware emits RAW:3 on every
    press of it and follows up with WORD_SPACE or CONFIRM once it has decided
    whether the press was single or double — those follow-ups are ignored here,
    so a triple press registers as exactly three RAW:3 messages regardless of
    how the firmware classifies the gesture.

    Returns (None, no-op) when no Pico W is attached, so callers degrade to the
    keyboard/backstop path instead of recording into a recording that nothing
    can end.
    """
    serial = _morse_serial_singleton      # bound once: one reader, one queue
    if serial is None:
        return None, (lambda: None)

    stopped = threading.Event()
    finished = threading.Event()

    def _watch():
        presses = 0
        last_press = 0.0
        while not finished.is_set():
            try:
                msg = serial.get_message(timeout=0.3)
            except Exception as e:
                logger.debug(f"Button read error while recording: {e}")
                time.sleep(0.3)
                continue
            if msg is None:
                continue

            now = time.time()
            if msg.startswith("RAW:3"):
                presses = presses + 1 if (now - last_press) <= _STOP_PRESS_WINDOW_S else 1
            elif msg == "CONFIRM":
                # The firmware only emits CONFIRM when it has decided two
                # presses were a double-press, and while it is making that
                # decision it stops scanning the pins — so one of the two
                # presses may never have reached us as a RAW:3. Treat CONFIRM
                # as proof that two presses happened, otherwise a user who taps
                # quickly could press four or five times and never reach three.
                presses = max(presses, 2)
            else:
                continue

            last_press = now
            logger.info(f"Stop button press {presses} of {_STOP_PRESSES}")
            if presses >= _STOP_PRESSES:
                stopped.set()
                return

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()

    def _cancel():
        finished.set()
        watcher.join(timeout=1)

    return stopped.is_set, _cancel


def _listen_until_stopped(lang: str = "en-IN", prompt: str | None = None):
    """Record for as long as the user needs, ending on three Button 3 presses.

    The 8-second phrase cap was cutting people off mid-question in every mode
    where they compose a sentence. Those modes now call this instead: the
    recording runs until the user signals they are done (or, with no buttons
    attached, until ENTER on a terminal / the safety backstop in voice.py).
    """
    if not _has("voice"):
        return None

    if _morse_serial_singleton is not None:
        try:
            _drain_button_messages()
        except Exception as e:
            logger.debug(f"Could not clear stale button presses before recording: {e}")

    stop_check, cancel = _button3_stop_signal()

    if prompt:
        _speak(prompt)
    if stop_check is not None:
        _speak(f"Take as long as you need. Press button 3 "
               f"{_STOP_PRESSES} times when you are finished.", block=True)
    elif _STDIN_IS_TTY:
        _speak("Take as long as you need. Press Enter when you are finished.", block=True)

    # Let the speaker finish before the microphone opens — otherwise the device
    # records its own prompt and tries to answer it.
    if _has("tts"):
        try:
            _modules["tts"].wait_until_idle(timeout=15)
        except Exception as e:
            logger.debug(f"Could not wait for TTS to drain: {e}")

    try:
        return _modules["voice"].listen(lang, stop_check=stop_check, manual_stop=True)
    except Exception as e:
        logger.warning(f"Voice capture error: {e}")
        return None
    finally:
        cancel()


def _speak_voice_failure(default: str = "I didn't catch that. Please try again."):
    """Explain the most recent listen() failure in the user's own words."""
    reason = None
    if _has("voice"):
        try:
            reason = _modules["voice"].get_last_error()
        except Exception:
            reason = None
    _speak(reason or default)


# ── HEADLESS-SAFE INPUT ─────────────────────────────────────
# The production device is a headless Pi with no keyboard: stdin is not a TTY
# and input() either blocks forever or raises. Several modes used to call
# input() directly as a "fallback", which on real hardware meant the device
# hung with no way out. Every keyboard read now goes through this guard.
_STDIN_IS_TTY = sys.stdin is not None and sys.stdin.isatty()


def _keyboard_input(prompt: str = "", default=None):
    """Read a line from the keyboard, or return `default` when there is no TTY."""
    if not _STDIN_IS_TTY:
        logger.debug(f"Keyboard input skipped (no TTY): {prompt!r}")
        return default
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return default

# ── MODE HANDLERS ───────────────────────────────────────────

def mode_ocr_scan():
    """Mode 1: OCR Scan → AI → TTS"""
    logger.info("Mode 1: OCR Scan")

    if not _has("ocr"):
        _speak("OCR module is not available.")
        return

    # UPDATED FOR NEW ocr.py: open_camera() now returns a single bool
    # (camera handling is fully internal to ocr.py's CameraManager
    # singleton) instead of a (cap, cam_type) tuple.
    if not _modules["ocr"].open_camera():
        _speak("Camera not available.")
        return

    try:
        while True:
            _drain_button_messages()
            stop_check, stop_watcher = _button3_stop_signal()
            try:
                if stop_check is not None:
                    _speak(f"Press button 3 {_STOP_PRESSES} times to cancel scanning.", block=True)
                def guidance_speak(message):
                    # Wait for each short prompt; the preview reader keeps only
                    # the newest image while speech plays.
                    _speak(message, block=True)
                text = _modules["ocr"].scan_and_read(
                    lang='eng', speak_fn=guidance_speak, stop_check=stop_check)
                if stop_check and stop_check():
                    text = "OCR scan cancelled."
            except KeyboardInterrupt:
                text = "OCR scan cancelled."
            except Exception:
                logger.exception("Guided OCR capture failed")
                text = "OCR guidance unavailable. Please check the camera and try again."
            finally:
                stop_watcher()
            if text != "OCR positioning timed out.":
                break
            _speak("Positioning timed out. Press button 1 to retry or button 2 to cancel.", block=True)
            if _morse_serial_singleton is not None:
                choice = _morse_serial_singleton.wait_for_raw_button(timeout=15)
            else:
                choice = _keyboard_input("1 to retry, Enter to cancel: ", default="")
            if str(choice) != "1":
                _speak("OCR scan cancelled.")
                return
        if text and text.startswith(("OCR scan cancelled", "OCR guidance unavailable")):
            _speak(text)
            return

        # UPDATED: the old check ("Could not" / capital-E "Error") never
        # matched any of the new ocr.py's real failure/status strings, so
        # a genuine failure slipped through silently and got spoken as if
        # it were scanned content. This now matches what ocr.py actually
        # returns on failure:
        #   "Camera not available. Please check the ribbon cable connection."
        #   "Image capture failed. Please try again."
        #   "Image processing error."
        #   "OCR unavailable: ..."
        #   "No text detected in the image."
        #   "OCR error: ..."
        failure_markers = (
            "camera not available", "capture failed", "processing error",
            "ocr unavailable", "no text detected", "ocr error",
        )
        if not text or any(marker in text.lower() for marker in failure_markers):
            _speak("I could not read the text clearly. Please try again.")
            return

        # —— PRIVACY ROUTING (FIX 1 & FIX 2) ——————————————————
        # The original code called ask_privacy() but never passed the morse
        # serial handle, then ALWAYS called _speak() afterwards — meaning the
        # text was read aloud through the speaker regardless of what the user
        # chose. This block replaces both the broken privacy call and the
        # broken "explain this?" response listener.
        #
        # speak_document_with_privacy_check() does three things atomically:
        #   1. Asks "confidential or normal?" through the earphone only
        #   2. Waits for Button 1 (private) or Button 2 (speaker) — with an
        #      8-second microphone fallback that accepts "yes"/"no"
        #   3. Routes the spoken text to ONLY the correct audio path
        # Nothing is read aloud before the user makes their choice.
        try:
            from modules import confidential_mode as _cm
            # Safely resolve the morse serial handle -- supports both variable
            # names used across different versions of this file.
            _ms_ref = _morse_serial_singleton  # always defined at module level
            _cm.speak_document_with_privacy_check(text, "eng", _ms_ref)
        except Exception as e:
            # Never leak recognized document text to the public speaker when
            # privacy routing fails.  The user can safely retry after checking
            # the earphones/audio devices.
            logger.error(f"OCR privacy reader failed: {e}")
            _speak(
                "I read the page, but private audio routing is unavailable. "
                "The document was not spoken. Please check the audio devices "
                "and try again."
            )
            return

        logger.info(f"OCR: {text[:200]}")

        # Index into RAG vector store for semantic retrieval
        if _has("ai_query") and hasattr(_modules["ai_query"], "index_text_in_rag"):
            _modules["ai_query"].index_text_in_rag(text)
            logger.info("OCR text indexed into RAG vector store.")

        # —— ASK IF USER WANTS EXPLANATION (FIX 2) ——————————————
        # Original code used input() which blocks forever on a headless device
        # with no keyboard. Now listens to the physical Pico W buttons first
        # (Button 1 = yes, Button 2 = no), with a 4-second window, then
        # falls back to the microphone, and finally defaults to "yes" so a
        # blind user never gets silently stuck waiting for a keypress.
        _speak("Would you like to explain this? Press 1 for yes, 2 for no.")

        response = "yes"   # Failsafe: ensures it never crashes even if mic fails

        try:
            # 1. Wait 4 seconds for a button press (Button 1 = yes, Button 2 = no)
            if _morse_serial_singleton is not None:
                btn = _morse_serial_singleton.wait_for_raw_button(timeout=4.0)
                if btn == 1:
                    response = "yes"
                elif btn == 2:
                    response = "no"
                else:
                    # 2. No button pressed — try the microphone
                    spoken = _listen("en-IN")
                    if spoken:
                        response = spoken.lower()
            else:
                # Keyboard fallback for laptop/development use
                kb = (_keyboard_input("[Y/N/1/2/Enter=Yes]: ", default="") or "").lower()
                if kb in ("n", "no", "2"):
                    response = "no"

        except Exception as e:
            logger.warning(f"Mic/Button error while asking for explanation: {e}")

        if response not in ("no", "n", "2"):
            _speak("Analyzing...")
            if _has("ai_query"):
                answer = _modules["ai_query"].ask_ai(
                    "Explain this in simple terms for a visually impaired student:",
                    context=text,
                    speak_fn=_speak,
                    flush_fn=_flush_speech,
                )
                _speak(answer)
            else:
                _speak("AI module is not available.")

    finally:
        # UPDATED FOR NEW ocr.py: release_camera() takes no arguments and
        # needs no cam_type branching — ocr.py's CameraManager tracks its
        # own camera type internally (rpicam vs usb) and releases whichever
        # one is active. The old cap.release()/cap.stop() branching is gone
        # because main.py no longer holds a cap/cam_type reference at all.
        _modules["ocr"].release_camera()
        if _has("privacy"):
            try:
                _modules["privacy"].reset_to_normal()
            except Exception:
                pass

def _morse_type_sentence(intro_message, timeout=90):
    """
    Shared helper: types a full sentence via the real Pico W buttons,
    speaking live feedback after every letter and every finished word
    (FIX — previously a blind user got zero feedback until the entire
    sentence was done, with no way to know a button press even
    registered). Falls back to keyboard typing if no Pico W is present.

    Button behaviour while typing (unchanged, just documented clearly):
    - dot/dash on Buttons 1/2  -> builds up the current letter
    - single press Button 3    -> finishes the letter, adds a space,
                                    keeps typing (does NOT send yet)
    - double press Button 3    -> finished — send the whole sentence
    - long press Button 3      -> backspace (deletes the last letter)

    Returns the typed sentence (str), or "" if nothing was typed.
    """
    if _morse_serial_singleton is not None:
        _speak(intro_message)
        _speak("Single-press button 3 for a space, double-press to send, long-press to backspace.")

        def _on_update(event, letter, current_text):
            if event == "letter":
                _speak(letter)
            elif event == "space":
                last_word = current_text.strip().split(" ")[-1] if current_text.strip() else ""
                if last_word:
                    _speak(f"word: {last_word}")
            elif event == "backspace":
                _speak("deleted")

        try:
            text = _morse_serial_singleton.type_word(timeout=timeout, on_update=_on_update).strip()
        except Exception as e:
            logger.error(f"Morse serial read error: {e}")
            text = ""
        if text:
            _speak(f"Full sentence: {text}")
        return text
    else:
        _speak(intro_message + " Type it, then press Enter.")
        try:
            text = _keyboard_input("Type: ", default="")
            if text is None:
                return ""
        except EOFError:
            return ""

        # Simple Morse decode if input looks like Morse (keyboard fallback path)
        if text and all(c in '.- /' for c in text):
            words = text.split(' / ') if ' / ' in text else text.split('  ')
            decoded = []
            morse_mod = _modules.get("morse")
            for word in words:
                for letter in word.strip().split(' '):
                    if letter:
                        decoded.append(morse_mod.MORSE_CODE_DICT.get(letter, '?'))
                decoded.append(' ')
            text = ''.join(decoded).strip()
            _speak(f"You typed: {text}")
        return text


def mode_morse_type():
    """Mode 2: Type a question via buttons (with live feedback) → AI → TTS"""
    logger.info("Mode 2: Morse Type")

    question = _morse_type_sentence("Type your question on the buttons.")
    if not question:
        _speak("No question received.")
        return

    _speak("Thinking...")
    if _has("ai_query"):
        answer = _modules["ai_query"].ask_ai(
            question, speak_fn=_speak, flush_fn=_flush_speech
        )
        _speak(answer)
    else:
        _speak("AI module is not available.")

def mode_voice_ask():
    """Mode 3: Voice → AI → TTS"""
    logger.info("Mode 3: Voice Ask")

    if not _has("voice"):
        _speak("Voice recognition is not available.")
        return

    question = _listen_until_stopped("en-IN", prompt="Voice mode. Ask your question now.")

    if not question:
        _speak_voice_failure()
        return

    _speak(f"You asked: {question}")
    logger.info(f"Voice: {question}")

    # Emotion check (passive)
    if _has("emotion"):
        # Note: emotion engine needs audio file path, not available here
        # In full implementation, voice.py should save the recording
        pass

    _speak("Thinking...")
    if _has("ai_query"):
        # ask_ai_and_speak() asks — on the speaker — whether the answer is
        # confidential, then reads it out on the earphone (Button 1) or the
        # speaker (Button 2, or after the timeout). It speaks the answer
        # itself, so there must be no _speak(answer) here: that would play
        # the whole answer a second time, on the wrong device, immediately
        # after the user has just chosen where to hear it.
        _modules["ai_query"].ask_ai_and_speak(
            question, speak_fn=_speak, flush_fn=_flush_speech
        )
    else:
        _speak("AI module is not available.")

def _decode_morse_input(morse_text: str) -> str:
    """Decode a Morse code string (e.g. '.... . .-.. .-.. --- / .-- --- .-. .-.. -..') to English."""
    morse_mod = _modules.get("morse")
    if not morse_mod:
        return ""

    morse_dict = morse_mod.MORSE_CODE_DICT
    decoded = []

    # Words are separated by ' / ' or triple space '   '
    words = morse_text.split(' / ') if ' / ' in morse_text else morse_text.split('   ')
    for word in words:
        letters = word.strip().split(' ')
        for letter_code in letters:
            letter_code = letter_code.strip()
            if letter_code:
                decoded.append(morse_dict.get(letter_code, '?'))
        decoded.append(' ')

    return ''.join(decoded).strip()


# ── BUTTON-ONLY MENU SELECTION ──────────────────────────────
# Mode 4 answers its menus with the three physical buttons and nothing else.
# It used to try the buttons, then open the microphone and transcribe a spoken
# answer, then read the keyboard. On the real device that meant every menu sat
# through a recording and a speech-to-text round trip — the lag the buttons
# were there to avoid.
#
# Timing note: the Pico flushes its Morse buffer into a LETTER message
# LETTER_GAP_MS (1500 ms) after the last symbol, so a button pressed to answer
# a menu ALSO arrives a moment later as a stray letter. Menus therefore drop
# whatever is already queued before they read, and Morse typing waits for that
# gap to pass so a word never starts with a phantom letter.
_BUTTON_CHOICE_TIMEOUT_S = float(_settings.get("button_choice_timeout_seconds", 15.0))
_BUTTON_DOUBLE_WINDOW_S = float(_settings.get("button_double_press_seconds", 2.0))
_BUTTON_LETTER_SETTLE_S = 1.8          # must exceed the Pico's LETTER_GAP_MS


def _drain_button_messages():
    """Drop button traffic that arrived before this menu started listening.

    Without this, the press that entered a mode is still sitting in the queue
    and instantly answers the next question — the same failure the stdin drain
    fixes for the keyboard.
    """
    serial = _morse_serial_singleton
    if serial is None:
        return
    try:
        while serial.get_message(timeout=0.02) is not None:
            pass
    except Exception as e:
        logger.debug(f"Could not drain the button queue: {e}")


def _button_choice(valid: tuple, timeout: float = None, double: str = None,
                   drain: bool = True) -> str | None:
    """One menu answer from the physical buttons. No microphone, no keyboard.

    Returns "1"/"2"/"3", or `double` repeated (e.g. "22") when that button is
    pressed twice within _BUTTON_DOUBLE_WINDOW_S, or None if nothing was
    pressed in time.
    """
    serial = _morse_serial_singleton
    if serial is None:
        return None

    if drain:
        _drain_button_messages()
    deadline = time.time() + (_BUTTON_CHOICE_TIMEOUT_S if timeout is None else timeout)

    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return None
        try:
            pressed = serial.wait_for_raw_button(timeout=remaining)
        except Exception as e:
            logger.warning(f"Button read failed: {e}")
            return None
        if pressed is None:
            return None

        key = str(pressed)
        if key not in valid:
            continue                    # a button this menu does not use

        if double and key == double:
            # A single press only means what it says once the window for a
            # second press has passed, so resolve that before returning.
            try:
                again = serial.wait_for_raw_button(timeout=_BUTTON_DOUBLE_WINDOW_S)
            except Exception:
                again = None
            if again is not None and str(again) == double:
                return double * 2
            if again is not None:
                logger.debug(f"Ignored button {again} inside the double-press window.")
        return key


def _select_with_buttons(valid: tuple, prompt: str,
                         double: str = None) -> str | None:
    """Speak a menu, then take the answer from the buttons.

    The prompt is spoken with block=True so the answer window does not start
    counting down while the options are still being read out.

    Stale presses are dropped BEFORE the prompt rather than after it. Draining
    afterwards would also throw away a press made while the menu was being
    spoken, forcing anyone who already knows the menu to press twice; draining
    first still removes the press that entered this mode, which is the one that
    would otherwise answer the question by itself.
    """
    _drain_button_messages()
    _speak(prompt, block=True)

    if _morse_serial_singleton is not None:
        return _button_choice(valid, double=double, drain=False)

    # No Pico attached at all — laptop simulation, per CLAUDE.md. This is never
    # reached on the device, where the buttons are the only selector; on a
    # headless Pi with no TTY _keyboard_input() returns the default and the
    # caller cancels cleanly.
    logger.info("No Pico W connected — Mode 4 menu falling back to the keyboard.")
    typed = _keyboard_input("Choice: ", default=None)
    if not typed:
        return None
    typed = typed.strip()
    if double and typed == double * 2:
        return double * 2
    return typed if typed in valid else None


def mode_translate():
    """
    Mode 4: Translate text.

    Step 1 — ask HOW text will be entered  (1=type  2=Morse buttons  3=voice)
    Step 2 — get the text using that method
    Step 3 — choose source and target from English, Hindi and Gujarati
    Step 4 — translate and speak the result

    Both menus are answered with the three physical buttons only. Source and
    target are separate choices, so all six directions are reachable without
    timing-sensitive double presses.
    """
    logger.info("Mode 4: Translate")

    if not _has("translator"):
        _speak("Translation module is not available.")
        return

    # ── STEP 1: Choose input method ───────────────────────────────────────
    print("\n[Mode 4 — Translate]  input method")
    print("  Button 1 → Type text")
    print("  Button 2 → Morse code buttons")
    print("  Button 3 → Voice / microphone")

    input_choice = _select_with_buttons(
        ("1", "2", "3"),
        "Translation mode. "
        "Press button 1 to type your text. "
        "Press button 2 to enter it in Morse code. "
        "Press button 3 to speak it."
    )

    if input_choice not in ("1", "2", "3"):
        _speak("No button was pressed. Translation cancelled.")
        return

    languages = {
        "1": ("en", "English"),
        "2": ("hi", "Hindi"),
        "3": ("gu", "Gujarati"),
    }
    language_names = {code: name for code, name in languages.values()}
    speech_langs = {
        "en": "en-IN",
        "hi": "hi-IN",
        "gu": "gu-IN",
    }

    def _choose_translation_direction() -> tuple[str, str] | None:
        print("\n  Source language: 1=English, 2=Hindi, 3=Gujarati")
        source_choice = _select_with_buttons(
            ("1", "2", "3"),
            "Choose the source language. Press button 1 for English, "
            "button 2 for Hindi, or button 3 for Gujarati."
        )
        if source_choice not in languages:
            return None

        src, source_name = languages[source_choice]
        _speak(f"Source language: {source_name}.", block=True)
        print("  Target language: 1=English, 2=Hindi, 3=Gujarati")
        target_choice = _select_with_buttons(
            ("1", "2", "3"),
            "Choose the target language. Press button 1 for English, "
            "button 2 for Hindi, or button 3 for Gujarati."
        )
        if target_choice not in languages:
            return None

        dest, target_name = languages[target_choice]
        if src == dest:
            _speak(
                "Source and target cannot be the same. Translation cancelled."
            )
            return None
        _speak(f"Target language: {target_name}.", block=True)
        return src, dest

    # ── STEP 2: Get the source text ───────────────────────────────────────
    text = None
    direction_choice = None

    if input_choice == "1":
        # ── Type text on keyboard ──
        _speak("Type your text and press Enter.")
        text = _keyboard_input("Text: ", default=None)
        if text is None:
            _speak("No input received.")
            return
        if not text:
            _speak("No text entered.")
            return

    elif input_choice == "2":
        # ── Morse code via Pico W buttons ──
        if _morse_serial_singleton is None and not _has("morse"):
            _speak("Morse module is not available. Please try option 1 or 3 instead.")
            return

        _prompt_started = time.time()
        _speak(
            "Morse input mode. "
            "Tap Button 1 for dot and Button 2 for dash. "
            "Pause 1.5 seconds to finish a letter. "
            "Double-press Button 3 to confirm the full word and send it.",
            block=True
        )

        # Use MorseSerial if Pico W is connected, else fall back to keyboard Morse
        if _morse_serial_singleton is not None:
            # Button 2 selected this mode, and the Pico is still holding that
            # press as a dash. Let its letter gap expire and throw away what it
            # produces, or every word typed here begins with a phantom "T".
            # Speaking the prompt normally covers the gap on its own; this only
            # tops up the remainder if speech was unavailable and returned at
            # once.
            _waited = time.time() - _prompt_started
            if _waited < _BUTTON_LETTER_SETTLE_S:
                time.sleep(_BUTTON_LETTER_SETTLE_S - _waited)
            _drain_button_messages()

            try:
                typed = _morse_serial_singleton.type_word(timeout=60)
                text = typed.strip() if typed else None
            except Exception as e:
                logger.warning(f"Morse serial read failed: {e}")
                text = None
        else:
            # Keyboard Morse fallback
            print("\nMorse Code keyboard entry:")
            print("  Letters: separate with space  (e.g. .... .)")
            print("  Words:   separate with  /     (e.g. .... . / .-- --- .-. .-.. -..)\n")
            raw_morse = _keyboard_input("Morse: ", default=None)
            if raw_morse is None:
                _speak("No input received.")
                return
            if not raw_morse:
                _speak("No Morse code entered.")
                return
            if not all(c in ".- / " for c in raw_morse):
                _speak("That does not look like Morse code. Use dots, dashes, spaces and slashes only.")
                return
            text = _decode_morse_input(raw_morse)

        if not text or text.replace("?", "").strip() == "":
            _speak("Could not decode the Morse input. Please try again.")
            return

        # Do not read the captured text over the public speaker before the
        # user has chosen a privacy route. The decoded content may be private.
        _speak("Morse text decoded.")
        logger.info("Translate — Morse input decoded successfully.")

    elif input_choice == "3":
        # ── Voice / microphone ──
        if not _has("voice"):
            _speak("Voice module is not available. Please use option 1 instead.")
            return

        # Voice recognition needs the source language before recording starts.
        # The old order captured every translation voice input as English, so
        # Hindi/Gujarati speech could be recorded cleanly and still transcribe
        # as nonsense.
        direction_choice = _choose_translation_direction()
        if direction_choice is None:
            _speak("No valid direction selected. Translation cancelled.")
            return

        src, _dest = direction_choice
        listen_lang = speech_langs.get(src, "en-IN")
        text = _listen_until_stopped(listen_lang, prompt="Speak your text now.")
        if not text:
            _speak_voice_failure("I did not catch anything. Please try again.")
            return
        # The privacy choice happens below. Echoing the recognised sentence
        # here could expose confidential text through the main speaker.
        _speak("Voice input captured.")

    # Reaching here means text is valid
    if not text:
        _speak("No text to translate.")
        return

    # ── STEP 3: Choose translation direction ──────────────────────────────
    if direction_choice is None:
        direction_choice = _choose_translation_direction()

    if direction_choice is None:
        _speak("No valid direction selected. Translation cancelled.")
        return

    src, dest = direction_choice
    if input_choice == "2" and src != "en":
        _speak(
            "Morse input currently supports English source text only. "
            "Please select English as the source or use voice input."
        )
        return
    label = f"{language_names[src]} to {language_names[dest]}"

    # Ask before any source text leaves the device. A private translation may
    # use only local providers and must neither read nor write the persistent
    # translation cache. If Confidential Mode is unavailable, the established
    # public behaviour remains available rather than silently claiming privacy.
    private_translation = False
    if _has("privacy"):
        try:
            privacy_choice = _modules["privacy"].ask_confidentiality(
                _morse_serial_singleton
            )
            private_translation = privacy_choice == "PRIVATE"
        except Exception as e:
            logger.error(f"Translation privacy selection failed: {e}")
            _speak("Privacy selection failed. Translation cancelled for safety.")
            return

    # ── STEP 4: Translate and speak ───────────────────────────────────────
    _speak(f"Translating. {label}.")
    translator = _modules["translator"]
    try:
        result = translator.translate_ex(
            text,
            from_lang=src,
            to_lang=dest,
            privacy=private_translation,
        )
    except Exception as e:
        logger.error(f"Translation error: {e}")
        _speak("Translation failed. Please try again.")
        return

    if not result.text:
        _speak("Translation returned an empty result.")
        return

    # The old code spoke whatever string came back — including the literal
    # words "Translation failed. Original text: ..." — through the English
    # voice. Two things were wrong with that: a failure was announced as
    # though it were a result, and a real Hindi or Gujarati result was read
    # in an English accent. translate_ex() reports both facts, so say the
    # right thing and use the right voice.
    tts_lang = translator.LANG_TO_TTS.get(result.lang, "eng")

    if result.translated:
        if private_translation:
            try:
                with _modules["privacy"].PrivateAudio():
                    _speak("Private translation.", block=True)
                    _speak(result.text, lang=tts_lang, block=True)
            except Exception as e:
                # Fail closed: never move private content to the speaker when
                # the private output route cannot be established.
                logger.error(f"Private translation playback failed: {e}")
                _speak("Private audio is unavailable. The result was not spoken.")
                return
            print("\n  [private translation completed; text hidden]\n")
            logger.info(
                f"Private translation [{src}→{dest}] completed via "
                f"{result.source} ({len(text)} source characters)."
            )
        else:
            _speak("Translation.", block=True)
            _speak(result.text, lang=tts_lang, block=True)
            print(f"\n  {label}: {result.text}\n")
            logger.info(
                f"Translated [{src}→{dest}] via {result.source}: "
                f"{text[:80]} → {result.text[:80]}")
    else:
        # Nothing was translated. Say why, then read back what the user gave
        # us so the session still ends with something useful rather than a
        # dead end.
        if private_translation:
            _speak(
                "A private local translation was not available. "
                "Your text was not sent to an online service.",
                block=True,
            )
        elif not translator.is_online():
            _speak("I could not translate that because there is no internet "
                   "connection. Here is your original text.", block=True)
        else:
            _speak("The translation service did not respond. "
                   "Here is your original text.", block=True)
        if not private_translation:
            _speak(result.text, lang=translator.LANG_TO_TTS.get(src, "eng"), block=True)
            print(f"\n  [not translated — {result.error}]  {result.text}\n")
        else:
            print("\n  [private translation unavailable; text hidden]\n")
        logger.warning(f"Translation unavailable [{src}→{dest}]: {result.error}")


# A gesture never launches a mode on its own, and it is never confirmed with a
# second gesture either. Recognising one hand shape reliably is hard enough on
# a moving hand at an unknown distance; requiring two in a row multiplied the
# ways it could fail. Button 1 is the shutter instead: the user holds the pose,
# presses, and the pose under the shutter is the one that counts. A button
# press cannot be misread.
_GESTURE_ACTIONS = {
    "MODE_SCAN":     "O C R scan",
    "MODE_VOICE":    "voice question",
    "OBJECT_DETECT": "object detection",
    "GPS_CHECK":     "G P S",
}
_GESTURE_GUIDANCE = {
    "NO_HAND": "I cannot see a hand. Put your full hand in front of the camera.",
    "HAND_CROPPED": "Part of your hand is outside the picture. Move it toward the centre.",
    "MOVE_CLOSER": "Your hand is too far away. Move it closer to the camera.",
    "MOVE_LEFT": "Move your hand a little to your left.",
    "MOVE_RIGHT": "Move your hand a little to your right.",
    "READY": "I can see your hand, but the pose is not clear. Hold it still.",
}
_GESTURE_MAX_S = float(_settings.get("gesture_max_seconds", 180.0))


def _gesture_label(name: str) -> str:
    """Spoken name for a gesture the camera reported."""
    return _GESTURE_ACTIONS.get(name) or name.replace("_", " ").lower()


def mode_gesture():
    """Mode 5: Gesture control — hold the pose, press Button 1 to take it."""
    logger.info("Mode 5: Gesture")

    if not _has("gesture"):
        _speak("Gesture control is not available.")
        return

    buttons = _morse_serial_singleton is not None
    if not buttons and not _STDIN_IS_TTY:
        _speak("Gesture mode needs the Pico buttons or an attached keyboard.")
        return
    if buttons:
        _speak("Gesture mode. Open palm for O C R, two fingers for voice, "
               "one finger for object detection, three fingers for G P S, "
               "or a closed fist to leave. "
               "Hold the shape, then press button 1 to use it. "
               "Press button 2 to hear what I can see. "
               "Press button 3 to leave gesture mode.", block=True)
        _drain_button_messages()
    else:
        _speak("Gesture mode. Open palm for O C R, two fingers for voice, "
               "one finger for object detection, three fingers for G P S, "
               "or a closed fist to leave. "
               "Press Enter to use the shape you are holding, "
               "type 2 to hear what I can see, or type 3 to leave.", block=True)

    active_action = None
    camera_failed = False
    startup_failure = None

    def on_gesture(name):
        """Reports from the detector. Returning False releases the camera."""
        nonlocal active_action, camera_failed, startup_failure

        if name == "CAMERA_UNAVAILABLE":
            camera_failed = True
            return False
        if name in {"MEDIAPIPE_UNAVAILABLE", "HAND_MODEL_MISSING",
                    "BACKEND_UNAVAILABLE",
                    "FRAME_READ_FAILED"}:
            startup_failure = name
            return False

        # Button 2 — say what is in view without acting on it. A sighted user
        # can see whether their hand is framed; this is the equivalent.
        if name.startswith("PREVIEW:"):
            seen = name.split(":", 1)[1]
            if seen == "NONE":
                _speak("I cannot see a hand. Move it in front of the camera, "
                       "about one arm away.")
            elif seen in _GESTURE_ACTIONS:
                _speak(f"I can see {_GESTURE_ACTIONS[seen]}. "
                       "Press button 1 to start it.")
            else:
                _speak(f"I can see {_gesture_label(seen)}, "
                       "which is not one of the modes.")
            return True

        if name.startswith("PREVIEW_GUIDANCE:"):
            guidance = name.split(":", 1)[1]
            _speak(_GESTURE_GUIDANCE.get(
                guidance, "I cannot read the hand position yet."
            ))
            return True

        if name.startswith("NO_GESTURE"):
            guidance = name.split(":", 1)[1] if ":" in name else "READY"
            _speak(_GESTURE_GUIDANCE.get(
                guidance,
                "I could not read the hand shape. Hold it still and try again."
            ))
            return True

        if name == "STOP":
            active_action = "STOP"
            return False

        if name in _GESTURE_ACTIONS:
            active_action = name
            return False          # leave the loop so the camera is released

        _speak("That pose is not assigned. Use one, two, three, or four open "
               "fingers, or a closed fist.")
        return True

    running = True
    while running:
        # A fresh stop signal per pass, and a watcher that exits before any
        # sub-mode starts — otherwise it would still be consuming the button
        # presses that the OCR or voice mode is waiting for.
        stop_gesture = threading.Event()
        watcher_finished = threading.Event()
        deadline = time.time() + _GESTURE_MAX_S
        requests = []
        requests_lock = threading.Lock()

        def _post(request, requests=requests, requests_lock=requests_lock):
            with requests_lock:
                requests.append(request)

        def _capture_check(requests=requests, requests_lock=requests_lock):
            """Polled once per frame by the detector; None means keep watching."""
            with requests_lock:
                return requests.pop(0) if requests else None

        def _watch_for_input(stop_gesture=stop_gesture,
                             watcher_finished=watcher_finished,
                             deadline=deadline):
            """Button 1 shutter, Button 2 preview, Button 3 leave."""
            while not stop_gesture.is_set() and not watcher_finished.is_set():
                if time.time() > deadline:
                    logger.info("Gesture mode hit its time limit.")
                    stop_gesture.set()
                    return

                if buttons:
                    try:
                        pressed = _morse_serial_singleton.wait_for_raw_button(timeout=0.5)
                    except Exception as e:
                        logger.debug(f"Button read error in gesture watcher: {e}")
                        continue
                    if pressed is None:
                        continue
                    if str(pressed) == "1":
                        _post("capture")
                    elif str(pressed) == "2":
                        _post("preview")
                    elif str(pressed) == "3":
                        stop_gesture.set()
                        return
                    continue

                if _STDIN_IS_TTY:
                    import select as _select
                    if _select.select([sys.stdin], [], [], 0.5)[0]:
                        typed = sys.stdin.readline().strip()
                        if typed == "3":
                            stop_gesture.set()
                            return
                        _post("preview" if typed == "2" else "capture")
                else:
                    time.sleep(0.5)

        watcher = threading.Thread(target=_watch_for_input, daemon=True)
        watcher.start()
        try:
            _modules["gesture"].detect_gesture(callback_fn=on_gesture,
                                               stop_event=stop_gesture,
                                               capture_check=_capture_check)
        except Exception as e:
            logger.error(f"Gesture detection failed: {e}")
            _speak("Gesture control stopped because of an error.")
            active_action = None
        finally:
            watcher_finished.set()
            watcher.join(timeout=1.5)

        if camera_failed:
            _speak("I cannot read the gesture camera. Check the USB camera or "
                   "Camera Module 3, then try again.")
            return

        if startup_failure:
            if startup_failure == "MEDIAPIPE_UNAVAILABLE":
                _speak("Gesture control needs MediaPipe, but it is not installed.")
            elif startup_failure == "HAND_MODEL_MISSING":
                _speak("The gesture hand model is not installed. Run the gesture "
                       "preparation script, then try again.")
            elif startup_failure == "BACKEND_UNAVAILABLE":
                _speak("No compatible MediaPipe hand model is available.")
            else:
                _speak("The gesture camera stopped returning images.")
            return

        if active_action == "STOP":
            _speak("Leaving gesture mode.")
            return

        if active_action in _GESTURE_ACTIONS:
            action = active_action
            active_action = None
            _speak(f"Starting {_GESTURE_ACTIONS[action]}.")
            {
                "MODE_SCAN": mode_ocr_scan,
                "MODE_VOICE": mode_voice_ask,
                "OBJECT_DETECT": mode_object_detection,
                "GPS_CHECK": mode_gps,
            }[action]()
            _speak("Back in gesture mode.")
        else:
            running = False

def _object_detection_stop_requested(pressed) -> bool:
    """Only the dedicated cancel button may stop a running detector."""
    return pressed is not None and str(pressed) == "3"


def mode_object_detection():
    """Mode 6: Object detection"""
    logger.info("Mode 6: Object Detection")

    if not _has("objdetect"):
        _speak("Object detection is not available.")
        return

    # EXIT PATH (FIX): the old version told the user to "press Q", but Q is only
    # read inside the OpenCV display window — and object_detection_display is
    # false on the headless production device, so the loop never checked for it.
    # With no max_frames and no stop signal, entering this mode trapped the
    # device until Ctrl+C killed the whole application. There is now a real
    # stop signal: Button 3 on the Pico W, Enter on a keyboard,
    # or an automatic time limit.
    _speak("Preparing object detection.", block=True)
    try:
        prepare = getattr(_modules["objdetect"], "prepare_model", None)
        if prepare is not None:
            prepare()
    except Exception as e:
        logger.error(f"Object detection setup error: {e}")
        code = getattr(e, "code", "")
        if code == "MODEL_MISSING":
            _speak("The object detection model is not installed. Please run the "
                   "object detection preparation script while online.", block=True)
        else:
            _speak("The object detection model could not be loaded.", block=True)
        return

    if _morse_serial_singleton is not None:
        _speak("Object detection is ready. Press button 3 to stop.", block=True)
        # Mode selection and model-loading traffic must never stop a newly
        # started detector. Only events arriving after this drain count.
        _drain_button_messages()
    else:
        _speak("Object detection is ready. Press Enter to stop.", block=True)

    import time

    last_speak_time = 0
    last_spoken_text = None
    stop_detection = threading.Event()
    max_seconds = float(_settings.get("object_detection_max_seconds", 120))
    started_at = time.time()

    def _watch_for_stop():
        """Background: Button 3/Enter stops; other buttons are ignored."""
        while not stop_detection.is_set():
            if time.time() - started_at > max_seconds:
                stop_detection.set()
                return
            if _morse_serial_singleton is not None:
                try:
                    pressed = _morse_serial_singleton.wait_for_raw_button(timeout=0.5)
                    if _object_detection_stop_requested(pressed):
                        stop_detection.set()
                        return
                    continue
                except Exception as e:
                    logger.debug(f"Button read error while stopping detection: {e}")
            if _STDIN_IS_TTY:
                import select as _select
                if _select.select([sys.stdin], [], [], 0.5)[0]:
                    sys.stdin.readline()
                    stop_detection.set()
                    return
            else:
                time.sleep(0.5)

    threading.Thread(target=_watch_for_stop, daemon=True).start()

    def detection_callback(text, detections):
        nonlocal last_speak_time, last_spoken_text
        now = time.time()
        # The detector has already applied multi-frame stability and selected a
        # short spatial description. Announce a changed scene immediately and
        # repeat an unchanged scene only occasionally.
        if detections and (text != last_spoken_text or now - last_speak_time > 5.0):
            _speak(text)
            last_speak_time = now
            last_spoken_text = text

        # Hard time limit so the mode can never run away, even if every
        # interactive stop path is unavailable.
        if now - started_at > max_seconds:
            stop_detection.set()

        return not stop_detection.is_set()

    try:
        _modules["objdetect"].run_detection(
            callback=detection_callback, stop_event=stop_detection
        )
    except Exception as e:
        logger.error(f"Object detection error: {e}")
        code = getattr(e, "code", "")
        if (code in {"CAMERA_READ_FAILED", "CAMERA_UNAVAILABLE"}
                or "camera" in str(e).lower()):
            _speak("I cannot read the object detection camera. Check the camera "
                   "connection and try again.")
        else:
            _speak("Object detection stopped because of an error.")
    finally:
        stop_detection.set()
        _speak("Object detection stopped.")

def mode_gps():
    """Mode 7: GPS Navigation with 4-way input (Voice, Gesture, Morse/Button, Keyboard)"""
    logger.info("Mode 7: GPS")
    if not _has("gps"):
        _speak("GPS navigation is not available.")
        return

    import time
    import select
    import sys
    import threading

    # Go through the safe-import registry rather than importing the modules
    # directly: a missing optional dependency (mediapipe, PyAudio) must not
    # take down GPS mode, which works fine on buttons and keyboard alone.
    voice = _modules.get("voice")
    gesture = _modules.get("gesture")

    def _watch_gesture_in_background():
        """Run one gesture-detection pass on a thread; report the first
        recognised menu gesture. Returns a dict the caller polls cheaply."""
        result = {"value": None}
        if gesture is None:
            return result

        stop_event = threading.Event()

        def _on_gesture(name):
            if name == "MODE_SCAN":          # Open palm  -> Option 1
                result["value"] = "1"
            elif name in ("MODE_VOICE", "GPS_CHECK"):   # V-sign / 3 fingers -> Option 2
                result["value"] = "2"
            else:
                return True                  # keep looking
            stop_event.set()
            return False                     # stop the detection loop

        def _worker():
            try:
                gesture.detect_gesture(callback_fn=_on_gesture, stop_event=stop_event)
            except Exception as e:
                logger.warning(f"Gesture detection unavailable in GPS mode: {e}")

        threading.Thread(target=_worker, daemon=True).start()
        result["stop"] = stop_event
        return result

    def _listen_in_background(lang="en-IN"):
        """
        voice.listen() blocks for up to ~18s (VAD timeout + phrase capture +
        Google API round-trip), so it must never be called inside a fast
        polling loop. Runs it on a daemon thread and hands back a small
        result dict the caller can poll cheaply.
        """
        result = {"text": None, "done": False}
        if voice is None:
            result["done"] = True
            return result

        def _worker():
            try:
                # _listen() drains the speech queue first, so this thread does
                # not open the microphone while the menu prompt is still
                # playing out of the speaker.
                spoken = _listen(lang)
                result["text"] = spoken.lower() if spoken else None
            except Exception as e:
                logger.warning(f"Voice read error in GPS mode: {e}")
            finally:
                result["done"] = True

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return result

    # 1. Speak the prompt out loud
    _speak("GPS Mode. Press 1 for Where am I, or 2 to Navigate.")
    print("\n1: Where am I?  2: Navigate to...")

    choice = None
    start_time = time.time()

    # Voice starts once in the background — never re-called inside the loop
    voice_result = _listen_in_background()
    gesture_choice = _watch_gesture_in_background()

    # 2. Listen for all 4 inputs simultaneously.
    # The window has to outlast a full voice attempt (speech-queue drain +
    # 6s to start speaking + 8s phrase + transcription), or the advertised
    # "say 'where am I'" option can never win the race — the old 10s budget
    # expired while the recogniser was still working. Buttons, keyboard and
    # gestures still break out immediately, so this only affects a user who
    # does nothing at all.
    while time.time() - start_time < 25:

        # A. Keyboard Input (Non-blocking).
        # Guarded on isatty(): under systemd stdin is not a terminal and
        # select() reports it readable immediately at EOF, which spun this
        # loop at 100% CPU for its whole duration.
        if _STDIN_IS_TTY and select.select([sys.stdin], [], [], 0.0)[0]:
            line = sys.stdin.readline().strip()
            if line in ('1', '2'):
                choice = line
                break

        # B. Morse / Pico W Button Input
        if _morse_serial_singleton is not None:
            try:
                btn = _morse_serial_singleton.wait_for_raw_button(timeout=0.1)
                if btn in (1, 2):
                    choice = str(btn)
                    break
            except Exception as e:
                logger.warning(f"Morse/button read error in GPS mode: {e}")

        # C. Voice Input — read the background thread's result, don't re-call it
        if voice_result["done"] and voice_result["text"]:
            v = voice_result["text"]
            if "where" in v or "one" in v:
                choice = '1'
                break
            elif "navigate" in v or "two" in v:
                choice = '2'
                break
            voice_result["text"] = None  # consumed; thread already finished

        # D. Gesture Input (Camera)
        # FIX: this used to call gesture.get_current_gesture(), which has never
        # existed in gesture_control.py. Every call raised AttributeError and
        # was swallowed by `except AttributeError: pass`, so the gesture input
        # advertised for this mode was silently dead. gesture_control exposes a
        # callback-driven loop, so it is now run once on a background thread
        # and this loop just polls the result.
        if gesture_choice["value"]:
            choice = gesture_choice["value"]
            break

        time.sleep(0.05)

    # Always release the gesture camera before continuing, whether a gesture
    # was used or not — otherwise the detection thread keeps the camera open
    # for the rest of the session.
    if gesture_choice.get("stop") is not None:
        gesture_choice["stop"].set()

    # 3. Execute the chosen option
    if choice == '1':
        _speak("Finding your live location...")
        try:
            loc = _modules["gps"].get_location()
            _speak(f"You are near {loc['address']}")
        except Exception as e:
            logger.error(f"GPS location error: {e}")
            _speak("I could not find your location right now.")

    elif choice == '2':
        _speak("Where do you want to navigate to? Please speak the destination, or type it.")

        destination = None
        dest_result = _listen_in_background()
        dest_start = time.time()

        # Poll keyboard non-blocking while the voice thread runs in the background,
        # so typing doesn't have to wait for the mic to finish/timeout.
        while time.time() - dest_start < 25:  # covers voice.listen()'s worst case
            if _STDIN_IS_TTY and select.select([sys.stdin], [], [], 0.0)[0]:
                typed = sys.stdin.readline().strip()
                if typed:
                    destination = typed
                    break
            if dest_result["done"]:
                if dest_result["text"]:
                    destination = dest_result["text"]
                break
            time.sleep(0.05)

        if destination:
            _speak(f"Calculating walking route to {destination}...")
            try:
                directions = _modules["gps"].get_directions(destination)
                _speak(directions)
            except Exception as e:
                logger.error(f"GPS directions error: {e}")
                _speak("I could not calculate directions right now.")
        else:
            _speak("I didn't hear a destination. Exiting GPS mode.")
    else:
        _speak("No valid input received. Returning to main menu.")

def mode_confidential_demo():
    """Mode 8: Stand-alone demo/test of Confidential Mode"""
    logger.info("Mode 8: Confidential Mode Demo")

    if not _has("privacy"):
        _speak("Confidential mode is not available.")
        return

    _speak("This is a demonstration of confidential mode.")
    result = _modules["privacy"].ask_privacy()
    if result == "PRIVATE":
        _speak("This message is being spoken privately, through the earphone only.")
    else:
        _speak("This message is being spoken normally, through the speaker.")
    _modules["privacy"].reset_to_normal()

# Math word problems typed in Morse use ordinary spelled-out words instead
# of symbols the buttons have no way to produce, e.g.:
#   "SQRT 16 PLUS 3 TIMES X SQUARE MINUS 5 EQUALS 0"
#   "MATRIX 2 BY 2 ROW1 1 2 ROW2 3 4 FIND DETERMINANT"
# The verified local solver parses these words and gives TTS-safe results.
# Only word problems outside its strict grammar use the clearly-labelled AI
# fallback. See CHANGES.md for why symbol input was not added to Morse.
MATH_SOLVER_PROMPT = (
    "Solve this word problem as a careful math tutor. Restate exactly how you "
    "interpreted the quantities before calculating. Check the final value "
    "against the question. This fallback is not verified by the local symbolic "
    "solver, so never claim that it was automatically verified. Write plain "
    "sentences for text to speech: no Markdown, LaTeX, tables, or unexplained "
    "symbols. Keep it to six short sentences and end with the final answer.\n\n"
    "Problem: "
)

def mode_math_solver():
    """Mode 9: verified local math, with a labelled AI word-problem fallback."""
    logger.info("Mode 9: Math Solver")

    if not _has("mathsolver"):
        _speak("The verified math solver is not available. Check that SymPy is installed.")
        return

    print("\n[Mode 9 — Math Solver]  input method")
    print("  Button 1 → Voice input")
    print("  Button 2 → Type problem on buttons")
    method = _select_with_buttons(
        ("1", "2"),
        "Math Solver. Press button 1 for voice input, or button 2 to type the problem on the buttons.",
    )

    problem = ""
    if method == '1':
        if _has("voice"):
            problem = _listen_until_stopped("en-IN", prompt="Speak your math problem now.")
            if not problem:
                _speak_voice_failure()
                return
        else:
            _speak("Voice module is not available.")
            return
    elif method == '2':
        problem = _morse_type_sentence(
            "Spell out the problem using words instead of symbols, for example S Q R T for square root, "
            "or S Q U A R E for squared."
        )
    else:
        _speak("No valid input method selected.")
        return

    if not problem or not problem.strip():
        _speak("No problem received.")
        return

    _speak("Solving...")
    result = _modules["mathsolver"].solve(problem)
    if result.ok:
        logger.info(f"Math solved locally: kind={result.kind}, exact={result.exact}")
        _speak(result.spoken)
        return

    if result.error_code != "UNSUPPORTED":
        logger.warning(f"Local math rejected input: {result.error_code}")
        _speak(result.spoken)
        return

    if not _has("ai_query"):
        _speak(result.spoken + " The A I fallback is not available.")
        return

    _speak("This word problem is outside the verified local solver. I will try "
           "the A I tutor, but its answer is not automatically verified.")
    answer = _modules["ai_query"].ask_ai(
        MATH_SOLVER_PROMPT + problem,
        # A non-empty, explicit context prevents unrelated textbook RAG
        # retrieval from being mixed into a standalone calculation.
        context="Standalone math problem. Use only the quantities in the problem.",
        speak_fn=_speak,
        flush_fn=_flush_speech,
    )
    answer = _modules["mathsolver"].sanitize_ai_answer(answer)
    _speak(answer or "The A I tutor did not return an answer.")

# ── MAIN LOOP ───────────────────────────────────────────────

BANNER = """
╔═══════════════════════════════════════════════════════════╗
║          🦯 BlindAssist — Accessible Terminal 🦯          ║
║     CSR / Infineon 2025 — Dhruv Vaghela & Dax Patel      ║
╠═══════════════════════════════════════════════════════════╣
║  Tap the DIGIT in Morse on the buttons (or type the       ║
║  number if testing on a laptop):                          ║
║                                                             ║
║  1 → OCR Scan       2 → Morse Type      3 → Voice Ask     ║
║  4 → Translate      5 → Gesture         6 → Object Det.   ║
║  7 → GPS            8 → Confidential    9 → Math Solver   ║
║  0 → Shutdown (asks you to double-press Button 3 first)   ║
╚═══════════════════════════════════════════════════════════╝
"""

MODE_MAP = {
    '1': mode_ocr_scan,
    '2': mode_morse_type,
    '3': mode_voice_ask,
    '4': mode_translate,
    '5': mode_gesture,
    '6': mode_object_detection,
    '7': mode_gps,
    '8': mode_confidential_demo,
    '9': mode_math_solver,
}

def _get_menu_choice():
    """
    FIX: reads a single hands-free Morse DIGIT from the Pico W when it's
    connected (1-9 select a mode, 0 starts the shutdown-confirmation
    flow). Digits are used instead of letters because every Morse digit
    is exactly 5 dots/dashes long while every letter is 1-4 symbols long
    — there is no possible confusion with letters typed inside a
    sentence in Mode 2 or Mode 9, even if that sentence contains the
    digit itself as content (e.g. typing "5" as part of a math problem
    happens inside a completely separate program state from the menu).
    Falls back to typing a number on a keyboard only when no Pico W is
    available (e.g. laptop development).
    """
    if _morse_serial_singleton is not None:
        _speak("Tap a digit 1 through 9 to choose a mode, or 0 to shut down.")
        return _morse_serial_singleton.read_menu_digit(timeout=120)
    return _keyboard_input("Mode (1-9, 0=quit): ", default=None)


def _has_any_input_device() -> bool:
    """True if the user can actually drive the menu at all."""
    return _morse_serial_singleton is not None or _STDIN_IS_TTY

def main():
    logger.info("=" * 50)
    logger.info("BlindAssist Starting...")
    logger.info("=" * 50)

    if _morse_serial_singleton is not None:
        _speak("Welcome to Blind Assist. Tap a digit on the buttons to select a mode.")
    else:
        _speak("Welcome to Blind Assist. Press a number to select a mode.")

    print(BANNER)

    # Refuse to spin: with no Pico W and no terminal there is no way for the
    # user to select anything, and the menu loop would otherwise busy-loop
    # forever announcing "no selection received".
    if not _has_any_input_device():
        msg = ("No input device available: no Pico W buttons detected and no "
               "terminal attached. Connect the buttons or run from a terminal.")
        logger.error(msg)
        _speak("No input device is connected. Please connect the buttons.")
        shutdown()
        return

    running = True
    consecutive_empty = 0
    while running:
        try:
            print("\n" + "─" * 50)
            choice = _get_menu_choice()

            if choice is None:
                consecutive_empty += 1
                # Back off instead of hammering the TTS queue if input dies
                # mid-session (e.g. the Pico W is unplugged).
                if consecutive_empty >= 3:
                    logger.error("No input received repeatedly — shutting down.")
                    _speak("Input device not responding. Shutting down.")
                    running = False
                    continue
                _speak("No selection received. Try again.")
                time.sleep(1)
                continue

            consecutive_empty = 0

            if choice == '0':
                # FIX: shutdown is destructive/irreversible, so it requires an
                # explicit double-press confirmation instead of executing on a
                # single tap. Modes 1-9 don't need this — you can always just
                # return to the menu from them.
                if _morse_serial_singleton is not None:
                    _speak("Shutdown requested. Double-press button 3 within 5 seconds to confirm, or wait to cancel.")
                    confirmed = _morse_serial_singleton.wait_for_confirm(timeout=5)
                else:
                    ans = (_keyboard_input("Confirm shutdown? (y/n): ", default="") or "").lower()
                    confirmed = (ans == 'y')

                if confirmed:
                    _speak("Confirmed. Shutting down.")
                    running = False
                else:
                    _speak("Shutdown cancelled. Returning to menu.")
                continue

            if choice in MODE_MAP:
                try:
                    MODE_MAP[choice]()
                except Exception as e:
                    logger.error(f"Mode {choice} error: {e}")
                    _speak("An error occurred. Returning to menu.")
            else:
                _speak("Invalid selection. Choose a digit 1 to 9, or 0 to shut down.")

        except (EOFError, KeyboardInterrupt):
            running = False

    shutdown()

def shutdown():
    global _shutdown_done
    if _shutdown_done:
        return
    _shutdown_done = True

    logger.info("Shutting down...")
    # FIX: this used to reference _modules["tts"].tts_manager, which does not
    # exist (the module's singleton is the private _tts_manager). The getattr
    # guard silently swallowed it, so the TTS worker and audio device were
    # never released — and the fixed 1-second sleep cut the goodbye off
    # mid-word. Now we wait for the queue to actually drain, then shut down
    # through the module's public API.
    _speak("Goodbye. Shutting down Blind Assist.")
    if _has("tts"):
        try:
            _modules["tts"].wait_until_idle(timeout=10)
            _modules["tts"].shutdown()
        except Exception as e:
            logger.warning(f"TTS shutdown error: {e}")
    if _morse_serial_singleton is not None:
        try:
            _morse_serial_singleton.close()
        except Exception:
            pass
    logger.info("Shutdown complete.")
    print("\n✅ BlindAssist closed.\n")

def _handle_signal(sig, frame):
    shutdown()
    raise SystemExit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    main()
