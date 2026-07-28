"""
translator.py — BlindAssist Project
=====================================
Project  : Accessible Educational Terminal for Visually Impaired
Team     : Dhruv Vaghela & Dax Patel  |  CSR / Infineon 2025
Module   : Multilingual Translation

Translates text between English, Hindi, and Gujarati using
Google Translate's public web endpoint. Falls back gracefully
if internet is unavailable.
"""

import sys
import signal
import logging
import requests

from pathlib import Path
from typing import Optional

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

# ──────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "translator.log"
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────
logger = logging.getLogger("TranslatorModule")

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

TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"


def _request_translation(text: str, src: str, dest: str):
    params = {
        "client": "gtx",
        "sl": src,
        "tl": dest,
        "dt": "t",
        "q": text,
    }
    response = requests.get(TRANSLATE_URL, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


# ──────────────────────────────────────────────────────────────
# PUBLIC API — translate()
# ──────────────────────────────────────────────────────────────
def translate(text: str, from_lang: str = 'en', to_lang: str = 'hi') -> str:
    """
    Translate text between supported languages.

    Args:
        text: The text to translate.
        from_lang: Source language code ('en', 'hi', 'gu', 'eng', 'hin', 'guj').
        to_lang: Target language code ('en', 'hi', 'gu', 'eng', 'hin', 'guj').

    Returns:
        Translated text, or original text with error message if translation fails.
    """
    if not text or not text.strip():
        return "No text to translate."

    # Normalize language codes
    src = SETTINGS_TO_GOOGLE.get(from_lang, from_lang)
    dest = SETTINGS_TO_GOOGLE.get(to_lang, to_lang)

    if src not in SUPPORTED_LANGS:
        logger.warning(f"Unsupported source language: {from_lang}")
        return f"Unsupported source language: {from_lang}"

    if dest not in SUPPORTED_LANGS:
        logger.warning(f"Unsupported target language: {to_lang}")
        return f"Unsupported target language: {to_lang}"

    if src == dest:
        logger.info("Source and target language are the same.")
        return text

    try:
        logger.info(
            f"Translating from {SUPPORTED_LANGS[src]} to "
            f"{SUPPORTED_LANGS[dest]}: \"{text[:80]}...\""
        )

        result = _request_translation(text, src, dest)
        translated = "".join(
            part[0] for part in result[0]
            if isinstance(part, list) and part and part[0]
        ).strip()

        if translated:
            logger.info(f"Translation result: \"{translated[:80]}...\"")
            return translated

        logger.warning("Translation returned empty result.")
        return text

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return f"Translation failed. Original text: {text}"


# ──────────────────────────────────────────────────────────────
# PUBLIC API — detect_language()
# ──────────────────────────────────────────────────────────────
def detect_language(text: str) -> Optional[str]:
    """
    Detect the language of given text.

    Returns:
        Language code ('en', 'hi', 'gu') or None.
    """
    try:
        result = _request_translation(text, "auto", "en")
        lang = result[2] if len(result) > 2 else None
        if lang:
            logger.info(f"Detected language: {lang}")
            return lang
        return None
    except Exception as e:
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
            src = input("From language (1=EN, 2=HI, 3=GU): ").strip()
            dest = input("To language   (1=EN, 2=HI, 3=GU): ").strip()

            src_code = lang_map.get(src, 'en')
            dest_code = lang_map.get(dest, 'hi')

            # --- RUN TRANSLATION ---
            result = translate(text, src_code, dest_code)
            print(f"\n>>> Translation Result: {result}\n")

        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nExiting Translator Test...")
            break

    print("Translator Module Closed.")
