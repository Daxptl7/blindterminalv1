"""
AET -- Confidential Mode Module (Dual USB Sound Card Version)

Controls dual audio routing between the normal speaker and the wired
3.5mm earphones (plugged into the second USB sound card) using
software switching inside tts.py. No GPIO / bone conduction hardware
involved anymore.

Privacy selection uses physical buttons only. No microphone is opened.
Missing buttons or an unanswered prompt cancel document playback.
"""

import logging
import re
import time

from modules.config_loader import load_settings
from modules import tts

logger = logging.getLogger("ConfidentialMode")

# MorseSerial pulls in pyserial, which is absent on any machine without the
# Pico W toolchain installed. Importing it at module scope made THIS ENTIRE
# MODULE fail to import ("No module named 'serial'"), so main.py's safe-import
# marked privacy as unavailable and Confidential Mode — the feature that keeps
# a blind user's private documents from being read aloud in public — silently
# did nothing. It is only needed as a type hint and for the standalone test,
# so the import is now optional.
try:
    from modules.morse_serial import MorseSerial
except Exception as e:  # pragma: no cover - depends on host hardware/deps
    MorseSerial = None
    logger.info(f"Pico W buttons unavailable in confidential mode ({e}); button selection unavailable.")

settings = load_settings()

# Safely load the timeout even if the settings file is missing the privacy block
try:
    PROMPT_TIMEOUT = max(15, min(120, float(settings.get("privacy", {}).get(
        "confidential_prompt_timeout_seconds", 15))))
except (TypeError, ValueError):
    PROMPT_TIMEOUT = 15


def enable_earphone():
    """Routes audio strictly to the wired earphones (Private)."""
    tts.use_earphone_output()
    print(" 🔒 [PRIVATE MODE] Audio routed to wired earphones.")


# The old name, from when this hardware was going to be a bone-conduction pad.
enable_bone_conduction = enable_earphone


def enable_speaker():
    """Routes audio strictly to the main 3W speaker (Normal)."""
    tts.use_speaker_output()
    print(" 🔊 [NORMAL MODE] Audio routed to main speaker.")


# main.py calls _modules["privacy"].reset_to_normal() after every OCR scan
# and after the confidential demo — keep that name available as an alias.
reset_to_normal = enable_speaker


class PrivateAudio:
    """Context manager that routes audio to the private earphones for the
    duration of the `with` block, then always routes back to the speaker
    on the way out — even if an exception happens inside the block."""

    def __enter__(self):
        enable_earphone()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        enable_speaker()
        return False


def ask_confidentiality(morse_serial=None):
    """Button 1 selects earphones, Button 2 selects speaker; None cancels."""
    with PrivateAudio():
        if morse_serial is None:
            tts.speak("Buttons are not connected. Playback cancelled.", "en", block=True)
            return None
        try:
            # Clear old presses before the prompt, never after it. A press
            # made during speech stays queued for the selection loop.
            while morse_serial.get_message(timeout=0.05) is not None:
                pass
            tts.speak("Press button 1 for private earphones. Press button 2 for speaker.",
                      "en", block=True)
            deadline = time.monotonic() + PROMPT_TIMEOUT
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                button = morse_serial.wait_for_raw_button(timeout=remaining)
                if button == 1:
                    return "PRIVATE"
                if button == 2:
                    return "NORMAL"
                if button is None:
                    break
                # Ignore unrelated buttons; do not consume the choice window.
        except Exception:
            logger.exception("Privacy button input failed")
            tts.speak("Button input failed. Playback cancelled.", "en", block=True)
            return None
        tts.speak("No privacy option selected. Playback cancelled.", "en", block=True)
    return None


# Alias — main.py's standalone demo mode (Mode 8) calls ask_privacy().
def ask_privacy(morse_serial=None):
    return ask_confidentiality(morse_serial)


def speak_with_privacy_check(text, lang_code, morse_serial=None):
    """Asks confidential/public, then speaks `text` through the correct
    audio path only — nothing is read aloud before the choice is made."""
    mode = ask_confidentiality(morse_serial)
    if mode not in ("PRIVATE", "NORMAL"):
        return None
    if mode == "PRIVATE":
        with PrivateAudio():
            tts.speak(text, lang_code, block=True)
    else:
        enable_speaker()
        tts.speak(text, lang_code, block=True)
    return mode


def split_text_for_speech(text, max_chars=600):
    """Split a document into complete, sentence-aware TTS chunks.

    Long OCR results should not be sent to a speech engine as one enormous
    utterance: synthesis can time out and the user may hear nothing.  This
    splitter prefers paragraph and sentence boundaries, then word boundaries,
    while retaining every word in the recognized document.
    """
    try:
        limit = max(80, int(max_chars))
    except (TypeError, ValueError):
        limit = 600

    normalized = re.sub(r"[ \t]+", " ", str(text or "")).strip()
    if not normalized:
        return []

    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", normalized)
        if paragraph.strip()
    ]
    chunks = []

    def append_piece(piece):
        piece = piece.strip()
        while len(piece) > limit:
            cut = piece.rfind(" ", 0, limit + 1)
            if cut <= 0:
                cut = limit
            chunks.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            chunks.append(piece)

    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?\u0964])\s+", paragraph)
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{current} {sentence}".strip()
            if len(candidate) <= limit:
                current = candidate
                continue
            if current:
                chunks.append(current)
                current = ""
            if len(sentence) > limit:
                append_piece(sentence)
            else:
                current = sentence
        if current:
            chunks.append(current)

    return chunks


def speak_document_with_privacy_check(
    text,
    lang_code,
    morse_serial=None,
    chunk_chars=600,
):
    """Read a complete OCR document on one privacy route.

    The privacy question is asked once.  Every chunk is then spoken on the
    selected output before that route is released.
    """
    chunks = split_text_for_speech(text, max_chars=chunk_chars)
    if not chunks:
        return None

    mode = ask_confidentiality(morse_serial)
    if mode not in ("PRIVATE", "NORMAL"):
        return None
    if mode == "PRIVATE":
        with PrivateAudio():
            for index, chunk in enumerate(chunks):
                prefix = "I found. " if index == 0 else ""
                tts.speak(prefix + chunk, lang_code, block=True)
    else:
        enable_speaker()
        for index, chunk in enumerate(chunks):
            prefix = "I found. " if index == 0 else ""
            tts.speak(prefix + chunk, lang_code, block=True)
    return mode


# ---- Standalone test ----
if __name__ == "__main__":
    print("==================================================")
    print(" BlindAssist Confidential Mode — Dual USB Test")
    print("==================================================")

    ms = MorseSerial() if MorseSerial is not None else None
    if ms is None:
        print("(No Pico W detected — playback requires button selection.)")
    try:
        while True:
            print("\nPress Enter for privacy prompt (or type QUIT): ", end="")
            inp = input()
            if inp.strip().upper() == "QUIT":
                break

            print("\n─────────────────────────────────────────────")
            print(" 🔒 Is this confidential?")
            print(" Button 1 / key 1 -> PRIVATE (earphones only)")
            print(" Button 2 / key 2 -> NORMAL (speaker)")
            print(f" ({PROMPT_TIMEOUT}s timeout -> playback cancelled)")
            print("─────────────────────────────────────────────")

            speak_with_privacy_check(
                "This is a test of the confidential audio routing system.", "en", ms
            )
    except KeyboardInterrupt:
        pass
    finally:
        if ms is not None:
            ms.close()
        print("\nConfidential Mode Closed.")
