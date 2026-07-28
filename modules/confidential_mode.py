"""
AET -- Confidential Mode Module (Dual USB Sound Card Version)

Controls dual audio routing between the normal speaker and the wired
3.5mm earphones (plugged into the second USB sound card) using
software switching inside tts.py. No GPIO / bone conduction hardware
involved anymore.

FIXES IN THIS VERSION
----------------------
1. PrivateAudio.enter/exit were missing their double underscores, so
   `with PrivateAudio():` could never actually work as a context
   manager — Python requires __enter__/__exit__. That raised an
   AttributeError every single time, which main.py's try/except was
   silently swallowing. This is why the confidential prompt never
   played: the code crashed before it ever got to ask the question,
   and fell straight through to the "just speak it normally" fallback.
2. Added a microphone fallback: if nobody presses a button within
   PROMPT_TIMEOUT seconds, we listen on the mic for a spoken
   "private"/"yes" or "public"/"speaker"/"no" instead of just
   defaulting silently.
3. Added reset_to_normal() and ask_privacy() so main.py (which was
   written against slightly different function names) works without
   further changes.
"""

import time

from modules.config_loader import load_settings
from modules.morse_serial import MorseSerial
from modules import tts

settings = load_settings()

# Safely load the timeout even if the settings file is missing the privacy block
PROMPT_TIMEOUT = settings.get("privacy", {}).get("confidential_prompt_timeout_seconds", 8)


def enable_bone_conduction():
    """Routes audio strictly to the wired earphones (Private)."""
    tts.use_bone_conduction_output()
    print(" 🔒 [PRIVATE MODE] Audio routed to wired earphones.")


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
        enable_bone_conduction()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        enable_speaker()
        return False


def _interpret_privacy_voice(spoken):
    """Map a spoken phrase to 'PRIVATE', 'NORMAL', or None (unrecognized)."""
    if not spoken:
        return None
    s = spoken.strip().lower()
    private_words = ("private", "privacy", "confidential", "yes", "one", "1")
    normal_words = ("public", "normal", "speaker", "no", "two", "2")
    if any(w in s for w in private_words):
        return "PRIVATE"
    if any(w in s for w in normal_words):
        return "NORMAL"
    return None


def ask_confidentiality(morse_serial: MorseSerial = None):
    """
    Asks the privacy question via earphones ONLY, then waits for Button 1
    (private) / Button 2 (speaker). If nobody presses a button in time,
    falls back to listening on the mic for a spoken "private"/"yes" or
    "public"/"no". Defaults to NORMAL only if neither answers.
    """
    # Flush out any old, accidental button presses from the queue!
    if morse_serial is not None:
        while morse_serial.get_message(timeout=0.05) is not None:
            pass

    with PrivateAudio():
        tts.speak(
            "Is this confidential? Press 1 for Private. Press 2 for Speaker. "
            "You can also just say private, or public.",
            "en",
            block=True,
        )

    button = None
    if morse_serial is not None:
        button = morse_serial.wait_for_raw_button(timeout=PROMPT_TIMEOUT)

    if button == 1:
        return "PRIVATE"
    if button == 2:
        return "NORMAL"

    # No button pressed in time (or no Pico W connected) — try the mic.
    try:
        from modules import voice as _voice

        spoken = _voice.listen("en-IN")
        decision = _interpret_privacy_voice(spoken)
        if decision:
            return decision
    except Exception as e:
        print(f"[DEBUG] Voice fallback for privacy prompt failed: {e}")

    # Nobody answered by button or voice — safe default is NORMAL.
    return "NORMAL"


# Alias — main.py's standalone demo mode (Mode 8) calls ask_privacy().
def ask_privacy(morse_serial: MorseSerial = None):
    return ask_confidentiality(morse_serial)


def speak_with_privacy_check(text, lang_code, morse_serial: MorseSerial = None):
    """Asks confidential/public, then speaks `text` through the correct
    audio path only — nothing is read aloud before the choice is made."""
    mode = ask_confidentiality(morse_serial)
    if mode == "PRIVATE":
        with PrivateAudio():
            tts.speak(text, lang_code, block=True)
    else:
        enable_speaker()
        tts.speak(text, lang_code, block=True)
    return mode


# ---- Standalone test ----
if __name__ == "__main__":
    print("==================================================")
    print(" BlindAssist Confidential Mode — Dual USB Test")
    print("==================================================")

    ms = MorseSerial()
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
            print(f" ({PROMPT_TIMEOUT}s timeout -> falls back to mic, then NORMAL)")
            print("─────────────────────────────────────────────")

            speak_with_privacy_check(
                "This is a test of the confidential audio routing system.", "en", ms
            )
    except KeyboardInterrupt:
        pass
    finally:
        ms.close()
        print("\nConfidential Mode Closed.")
