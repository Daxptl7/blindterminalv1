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

# Helper to check if module is available
def _has(mod_name):
    return _modules.get(mod_name) is not None

def _speak(text):
    """Safe TTS — works even if tts module failed to load."""
    if _has("tts"):
        _modules["tts"].speak(text)
    else:
        print(f"[TTS OFFLINE] {text}")

# ── MODE HANDLERS ───────────────────────────────────────────

def mode_ocr_scan():
    """Mode 1: OCR Scan → AI → TTS"""
    logger.info("Mode 1: OCR Scan")
    _speak("Starting OCR scan. Hold your document steady.")

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
        _speak("Scanning now.")
        # UPDATED FOR NEW ocr.py: scan_and_read() now only takes lang= —
        # it opens/captures/reads the camera internally, so cap/cam_type
        # are no longer passed in.
        text = _modules["ocr"].scan_and_read(lang='eng')

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
        # speak_with_privacy_check() does three things atomically:
        #   1. Asks "confidential or normal?" through bone conduction only
        #   2. Waits for Button 1 (private) or Button 2 (speaker) — with an
        #      8-second microphone fallback that accepts "yes"/"no"
        #   3. Routes the spoken text to ONLY the correct audio path
        # Nothing is read aloud before the user makes their choice.
        try:
            from modules import confidential_mode as _cm
            # Safely resolve the morse serial handle -- supports both variable
            # names used across different versions of this file.
            _ms_ref = _morse_serial_singleton  # always defined at module level
            _cm.speak_with_privacy_check(f"I found: {text[:150]}", "eng", _ms_ref)
        except Exception as e:
            print(f"\n[DEBUG] Privacy block failed: {e}\n")
            _speak(f"I found: {text[:150]}")

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
                    if _has("voice"):
                        spoken = _modules["voice"].listen("en-IN")
                        if spoken:
                            response = spoken.lower()
            else:
                # Keyboard fallback for laptop/development use
                try:
                    kb = input("[Y/N/1/2/Enter=Yes]: ").strip().lower()
                    if kb in ("n", "no", "2"):
                        response = "no"
                except EOFError:
                    response = "yes"

        except Exception as e:
            print(f"\n[DEBUG] Mic/Button error: {e}\n")

        if response not in ("no", "n", "2"):
            _speak("Analyzing...")
            if _has("ai_query"):
                answer = _modules["ai_query"].ask_ai(
                    "Explain this in simple terms for a visually impaired student:",
                    context=text
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
            text = input("Type: ").strip()
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
        answer = _modules["ai_query"].ask_ai(question)
        _speak(answer)
    else:
        _speak("AI module is not available.")

def mode_voice_ask():
    """Mode 3: Voice → AI → TTS"""
    logger.info("Mode 3: Voice Ask")
    
    if not _has("voice"):
        _speak("Voice recognition is not available.")
        return

    _speak("Voice mode. Ask your question now.")
    question = _modules["voice"].listen(lang='en-IN')

    if not question:
        _speak("I didn't catch that. Please try again.")
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
        answer = _modules["ai_query"].ask_ai(question)
        _speak(answer)
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


def _get_button_or_voice_choice(options_map: dict, timeout: float = 8.0) -> str | None:
    """
    Universal choice getter used by mode_translate and any other multi-choice mode.

    Priority order (same pattern as the OCR explanation prompt):
      1. Physical Pico W button press  — Button 1/2/3 within `timeout` seconds
      2. Microphone                    — spoken word matched against options_map values
      3. Keyboard fallback             — for laptop/development testing only
      4. Returns None                  — caller must handle gracefully

    options_map examples:
      {"1": "1", "2": "2", "3": "3"}          — numeric choices
      {"1": "hindi", "2": "gujarati", ...}     — spoken language names
    """
    # Build a flat set of all valid keys and spoken aliases for mic matching
    valid_keys  = set(options_map.keys())
    spoken_vals = {v.lower(): k for k, v in options_map.items()}

    # ── 1. Physical button (Pico W) ───────────────────────────────────────
    if _morse_serial_singleton is not None:
        try:
            btn = _morse_serial_singleton.wait_for_raw_button(timeout=timeout)
            if btn is not None:
                key = str(btn)
                if key in valid_keys:
                    return key
                # Button number not in map — still return it so caller can decide
                return key
        except Exception as e:
            logger.debug(f"Button read error in choice getter: {e}")

    # ── 2. Microphone fallback ────────────────────────────────────────────
    if _has("voice"):
        try:
            spoken = _modules["voice"].listen("en-IN")
            if spoken:
                spoken_lower = spoken.lower().strip()
                # Direct key match  ("1", "2" etc spoken as a word)
                if spoken_lower in valid_keys:
                    return spoken_lower
                # Spoken value match  ("hindi", "yes", "no", etc.)
                for alias, key in spoken_vals.items():
                    if alias in spoken_lower:
                        return key
                # Number words
                number_words = {
                    "one": "1", "two": "2", "three": "3", "four": "4",
                    "five": "5", "six": "6", "seven": "7", "eight": "8",
                }
                for word, digit in number_words.items():
                    if word in spoken_lower and digit in valid_keys:
                        return digit
        except Exception as e:
            logger.debug(f"Voice choice error: {e}")

    # ── 3. Keyboard fallback (headless Pi: skipped silently) ─────────────
    try:
        kb = input("Choice: ").strip()
        if kb in valid_keys:
            return kb
        # Accept number words typed on keyboard too
        kb_lower = kb.lower()
        if kb_lower in spoken_vals:
            return spoken_vals[kb_lower]
    except EOFError:
        pass

    return None


def mode_translate():
    """
    Mode 4: Translate text.

    Step 1 — ask HOW text will be entered  (1=type  2=Morse buttons  3=voice)
    Step 2 — get the text using that method
    Step 3 — ask WHICH direction to translate (EN→HI / EN→GU / HI→EN / GU→EN)
    Step 4 — translate and speak the result

    Every selection step works via:
      • Pico W physical buttons  (Button 1/2/3)
      • Microphone  (say "one", "hindi", "yes", etc.)
      • Keyboard  (laptop / development fallback)
    """
    logger.info("Mode 4: Translate")

    if not _has("translator"):
        _speak("Translation module is not available.")
        return

    # ── STEP 1: Choose input method ───────────────────────────────────────
    _speak(
        "Translation mode. "
        "Press 1 or say 'one' to type your text. "
        "Press 2 or say 'two' to use Morse code buttons. "
        "Press 3 or say 'three' to speak your text."
    )
    print("\n[Mode 4 — Translate]")
    print("  1 → Type text")
    print("  2 → Morse code buttons (Pico W)")
    print("  3 → Voice / microphone")

    input_choice = _get_button_or_voice_choice(
        {"1": "type", "2": "morse", "3": "voice"}, timeout=8.0
    )

    if input_choice not in ("1", "2", "3"):
        _speak("No valid selection received. Translation cancelled.")
        return

    # ── STEP 2: Get the source text ───────────────────────────────────────
    text = None

    if input_choice == "1":
        # ── Type text on keyboard ──
        _speak("Type your text and press Enter.")
        try:
            text = input("Text: ").strip()
        except EOFError:
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

        _speak(
            "Morse input mode. "
            "Tap Button 1 for dot and Button 2 for dash. "
            "Pause 1.5 seconds to finish a letter. "
            "Double-press Button 3 to confirm the full word and send it."
        )

        # Use MorseSerial if Pico W is connected, else fall back to keyboard Morse
        if _morse_serial_singleton is not None:
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
            try:
                raw_morse = input("Morse: ").strip()
            except EOFError:
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

        _speak(f"I decoded: {text}")
        logger.info(f"Translate — Morse decoded to: {text}")

    elif input_choice == "3":
        # ── Voice / microphone ──
        if not _has("voice"):
            _speak("Voice module is not available. Please use option 1 instead.")
            return
        _speak("Speak your text now.")
        try:
            text = _modules["voice"].listen("en-IN")
        except Exception as e:
            logger.warning(f"Voice listen error in translate: {e}")
            text = None
        if not text:
            _speak("I did not catch anything. Please try again.")
            return
        _speak(f"I heard: {text}")

    # Reaching here means text is valid
    if not text:
        _speak("No text to translate.")
        return

    # ── STEP 3: Choose translation direction ──────────────────────────────
    _speak(
        "Which translation do you want? "
        "Press 1 or say 'hindi'    for English to Hindi. "
        "Press 2 or say 'gujarati' for English to Gujarati. "
        "Press 3 or say 'english'  for Hindi to English. "
        "Press 4 or say 'gujarati english' for Gujarati to English."
    )
    print("\n  1 → English → Hindi")
    print("  2 → English → Gujarati")
    print("  3 → Hindi   → English")
    print("  4 → Gujarati → English")

    direction_choice = _get_button_or_voice_choice(
        {"1": "hindi", "2": "gujarati", "3": "english", "4": "gujarati english"},
        timeout=8.0
    )

    pairs = {
        "1": ("en", "hi"),
        "2": ("en", "gu"),
        "3": ("hi", "en"),
        "4": ("gu", "en"),
    }
    direction_labels = {
        "1": "English to Hindi",
        "2": "English to Gujarati",
        "3": "Hindi to English",
        "4": "Gujarati to English",
    }

    if direction_choice not in pairs:
        _speak("No valid direction selected. Translation cancelled.")
        return

    src, dest = pairs[direction_choice]
    label    = direction_labels[direction_choice]

    # ── STEP 4: Translate and speak ───────────────────────────────────────
    _speak(f"Translating. {label}.")
    try:
        result = _modules["translator"].translate(text, from_lang=src, to_lang=dest)
    except Exception as e:
        logger.error(f"Translation error: {e}")
        _speak("Translation failed. Please try again.")
        return

    if not result:
        _speak("Translation returned an empty result.")
        return

    _speak(f"Translation: {result}")
    logger.info(f"Translated [{src}→{dest}]: {text[:80]} → {result[:80]}")


def mode_gesture():
    """Mode 5: Gesture control"""
    logger.info("Mode 5: Gesture")
    
    if not _has("gesture"):
        _speak("Gesture control is not available.")
        return

    _speak("Gesture mode activated. Show your hand to the camera.")

    active_action = None
    stop_gesture = threading.Event()

    def on_gesture(name):
        nonlocal active_action
        _speak(f"Gesture: {name}")
        if name == "STOP":
            _speak("Gesture control stopped.")
            return False
        if name in {"MODE_SCAN", "MODE_VOICE"}:
            active_action = name
            return False  # Release camera/mic resources by exiting loop
        if name == "CONFIRM":
            _speak("Confirmed.")
        elif name == "REPEAT":
            _speak("Repeat requested.")
        return True

    running = True
    while running:
        _modules["gesture"].detect_gesture(callback_fn=on_gesture, stop_event=stop_gesture)
        
        if active_action == "MODE_SCAN":
            active_action = None
            mode_ocr_scan()
            _speak("Resuming gesture control.")
        elif active_action == "MODE_VOICE":
            active_action = None
            mode_voice_ask()
            _speak("Resuming gesture control.")
        else:
            running = False

def mode_object_detection():
    """Mode 6: Object detection"""
    logger.info("Mode 6: Object Detection")
    
    if not _has("objdetect"):
        _speak("Object detection is not available.")
        return

    _speak("Starting object detection. Point camera at objects. Press Q to stop.")
    
    from collections import Counter
    import time
    
    last_speak_time = 0
    last_detected_classes = set()

    def detection_callback(text, detections):
        nonlocal last_speak_time, last_detected_classes
        now = time.time()
        current_classes = {d['name'] for d in detections}
        
        # Detect if any new object types entered the camera view
        new_objects = current_classes - last_detected_classes
        
        # Speak only if 3.5s passed OR a new object type is detected
        if (now - last_speak_time > 3.5 and current_classes) or new_objects:
            if current_classes:
                counts = Counter([d['name'] for d in detections])
                items = [f"{count} {name}" + ("s" if count > 1 else "") for name, count in counts.items()]
                _speak(f"I see {', '.join(items)}")
            last_speak_time = now
            last_detected_classes = current_classes

    try:
        _modules["objdetect"].run_detection(callback=detection_callback)
    except Exception as e:
        logger.error(f"Object detection error: {e}")
        _speak("Object detection error.")

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
    import modules.voice as voice
    import modules.gesture_control as gesture

    def _listen_in_background(lang="en-IN"):
        """
        voice.listen() blocks for up to ~18s (VAD timeout + phrase capture +
        Google API round-trip), so it must never be called inside a fast
        polling loop. Runs it on a daemon thread and hands back a small
        result dict the caller can poll cheaply.
        """
        result = {"text": None, "done": False}

        def _worker():
            try:
                spoken = voice.listen(lang)
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

    # 2. Listen for all 4 inputs simultaneously for 10 seconds
    while time.time() - start_time < 10:

        # A. Keyboard Input (Non-blocking)
        if select.select([sys.stdin], [], [], 0.0)[0]:
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
        try:
            gesture_cmd = gesture.get_current_gesture()
            if gesture_cmd in ("MODE_SCAN",):             # High five / Scan -> Option 1
                choice = '1'
                break
            elif gesture_cmd in ("MODE_VOICE", "PEACE"):   # Peace sign -> Option 2
                choice = '2'
                break
        except AttributeError:
            pass
        except Exception as e:
            logger.warning(f"Gesture read error in GPS mode: {e}")

        time.sleep(0.05)

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
        while time.time() - dest_start < 18:  # covers voice.listen()'s worst case
            if select.select([sys.stdin], [], [], 0.0)[0]:
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
        _speak("This message is being spoken privately, through bone conduction only.")
    else:
        _speak("This message is being spoken normally, through the speaker.")
    _modules["privacy"].reset_to_normal()

# Math word problems typed in Morse use ordinary spelled-out words instead
# of symbols the buttons have no way to produce, e.g.:
#   "SQRT 16 PLUS 3 TIMES X SQUARE MINUS 5 EQUALS 0"
#   "MATRIX 2 BY 2 ROW1 1 2 ROW2 3 4 FIND DETERMINANT"
# The AI is asked to read these the same way a teacher reading a problem
# aloud would, and to answer the same way — in full spoken sentences,
# never using symbols like "^", "√", or LaTeX, since that would be
# unreadable/unspeakable for a blind student. See CHANGES.md for why
# this approach was chosen over trying to invent new symbol input.
MATH_SOLVER_PROMPT = (
    "You are a patient math tutor speaking to a blind student through a "
    "text-to-speech system. The student's question may use spelled-out "
    "words instead of symbols (for example 'SQRT' means square root, "
    "'SQUARE' or 'POWER 2' means squared, 'MATRIX ROW1 1 2 ROW2 3 4' "
    "describes a matrix by rows). Solve the problem step by step. "
    "In your answer, speak every step in plain spoken sentences — never "
    "use mathematical symbols, exponents written with ^, square root "
    "signs, or LaTeX, since none of that can be read aloud. Say things "
    "like 'x squared' or 'the square root of sixteen' instead. Keep each "
    "step short and clear. End with the final answer stated plainly.\n\n"
    "Student's problem: "
)

def mode_math_solver():
    """Mode 9: Math Solver — voice or Morse-typed word problems, spoken step-by-step (NEW)"""
    logger.info("Mode 9: Math Solver")

    if not _has("ai_query"):
        _speak("AI module is not available.")
        return

    _speak("Math Solver. Press 1 for voice input, or 2 to type the problem on the buttons.")
    method = None
    if _morse_serial_singleton is not None:
        digit = _morse_serial_singleton.read_menu_digit(timeout=20)
        method = digit
    else:
        method = input("1=voice, 2=type: ").strip()

    problem = ""
    if method == '1':
        if _has("voice"):
            _speak("Speak your math problem now.")
            problem = _modules["voice"].listen()
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
    answer = _modules["ai_query"].ask_ai(MATH_SOLVER_PROMPT + problem)
    _speak(answer)

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
    else:
        return input("Mode (1-9, 0=quit): ").strip()

def main():
    logger.info("=" * 50)
    logger.info("BlindAssist Starting...")
    logger.info("=" * 50)

    if _morse_serial_singleton is not None:
        _speak("Welcome to Blind Assist. Tap a digit on the buttons to select a mode.")
    else:
        _speak("Welcome to Blind Assist. Press a number to select a mode.")

    print(BANNER)

    running = True
    while running:
        try:
            print("\n" + "─" * 50)
            choice = _get_menu_choice()

            if choice is None:
                _speak("No selection received. Try again.")
                continue

            if choice == '0':
                # FIX: shutdown is destructive/irreversible, so it requires an
                # explicit double-press confirmation instead of executing on a
                # single tap. Modes 1-9 don't need this — you can always just
                # return to the menu from them.
                if _morse_serial_singleton is not None:
                    _speak("Shutdown requested. Double-press button 3 within 5 seconds to confirm, or wait to cancel.")
                    confirmed = _morse_serial_singleton.wait_for_confirm(timeout=5)
                else:
                    ans = input("Confirm shutdown? (y/n): ").strip().lower()
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
    _speak("Goodbye. Shutting down Blind Assist.")
    time.sleep(1)
    if _has("tts") and getattr(_modules["tts"], "tts_manager", None) is not None:
        _modules["tts"].tts_manager.shutdown()
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
