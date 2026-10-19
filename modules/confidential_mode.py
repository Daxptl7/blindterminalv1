"""
AET -- Confidential Mode Module (Dual USB Sound Card Version)

Controls dual audio routing between the normal speaker and the wired
3.5mm earphones (plugged into the second USB sound card) using
software switching inside tts.py. No GPIO / bone conduction hardware
involved anymore.

Output selection happens after processing and applies only to that result.
No microphone is opened by the selector. An unanswered prompt cancels playback.
"""

import logging
import re
import time
import sys
import select

from modules.config_loader import load_settings
from modules import tts
from modules.privacy_policy import PRIVATE, processing_session

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
    tts.require_private_output()
    tts.use_earphone_output()
    print(" 🎧 [EARPHONE OUTPUT] Audio routed to wired earphones.")


# The old name, from when this hardware was going to be a bone-conduction pad.
enable_bone_conduction = enable_earphone


def enable_speaker():
    """Routes audio strictly to the main 3W speaker (Normal)."""
    tts.use_speaker_output()
    print(" 🔊 [SPEAKER OUTPUT] Prompts use the main speaker.")


# main.py calls _modules["privacy"].reset_to_normal() after every OCR scan
# and after the confidential demo — keep that name available as an alias.
reset_to_normal = enable_speaker


class PrivateAudio:
    """Context manager that routes audio to the private earphones for the
    duration of the `with` block, then always routes back to the speaker
    on the way out — even if an exception happens inside the block."""

    def __enter__(self):
        enable_earphone()
        self._session = processing_session(PRIVATE)
        self._session.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._session.__exit__(exc_type, exc_value, traceback)
        enable_speaker()
        return False


def ask_confidentiality(morse_serial=None):
    """Choose this result's audio output; this never chooses processing policy.

    PRIVATE/NORMAL are legacy return values for earphones/speaker only.
    """
    # The choice prompt must be audible from the device speaker. It is spoken
    # before the user has selected a destination, so routing it through the
    # private output made the later speaker/earphone choice sound reversed.
    enable_speaker()
    if morse_serial is None:
        if sys.stdin is not None and sys.stdin.isatty():
            tts.speak("Your result is ready. Type 1 for earphones, or 2 for the speaker.",
                      "en", block=True)
            deadline = time.monotonic() + PROMPT_TIMEOUT
            while time.monotonic() < deadline:
                if select.select([sys.stdin], [], [], .1)[0]:
                    choice = sys.stdin.readline().strip()
                    if choice in ("1", "2"):
                        return "PRIVATE" if choice == "1" else "NORMAL"
                    if not choice:
                        break
        tts.speak("No audio output selected. Result playback cancelled.", "en", block=True)
        return None
    try:
        # Clear old presses before the prompt, never after it. A press made
        # during speech stays queued for the selection loop.
        while morse_serial.get_message(timeout=0.05) is not None:
            pass
        tts.speak("Your result is ready. Press button 1 to hear it in the earphones, or button 2 to hear it on the speaker.",
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
    tts.speak("No audio output selected. Playback cancelled.", "en", block=True)
    return None


# Alias — main.py's standalone demo mode (Mode 8) calls ask_privacy().
def ask_privacy(morse_serial=None):
    return ask_confidentiality(morse_serial)


def speak_with_privacy_check(text, lang_code, morse_serial=None, **kwargs):
    """Choose an output after processing and route only this result there."""
    return speak_document_with_privacy_check(text, lang_code, morse_serial, **kwargs)


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
    *,
    speak_fn=None,
    choose_output=None,
    stop_check=None,
    before_play=None,
):
    """Read one completed result using per-utterance audio routing.

    Every chunk of a long result uses the chosen output. The global output
    and processing policy are never changed by the result's route.
    """
    chunks = split_text_for_speech(text, max_chars=chunk_chars)
    if not chunks:
        return None

    mode = choose_output() if choose_output else ask_confidentiality(morse_serial)
    if mode not in ("PRIVATE", "NORMAL"):
        return None
    device = tts.require_private_output() if mode == "PRIVATE" else tts.SPEAKER_DEVICE
    if before_play:
        before_play()
    for chunk in chunks:
        if speak_fn:
            speak_fn(chunk, lang=lang_code, block=True, device=device, stop_check=stop_check)
        else:
            tts.speak_on_device(chunk, device, lang=lang_code, block=True, stop_check=stop_check)
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
            print(" Where should this result be played?")
            print(" Button 1 / key 1 -> Earphones")
            print(" Button 2 / key 2 -> Speaker")
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
