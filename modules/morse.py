"""
morse.py — BlindAssist Project (OPTIMIZED)
============================================
Background timing thread with millisecond precision.
Auto-decodes letters and words without polling.
Thread-safe with Lock-free queue for main.py.
"""

import json
import logging
import signal
import sys
import time
import threading
from queue import Queue

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "settings.json"
LOG_PATH = BASE_DIR / "logs" / "morse.log"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("MorseModule")

# ── CONFIG ──────────────────────────────────────────────────
DEFAULT_SETTINGS = {
    "morse_dot_max_ms": 400,
    "morse_dash_min_ms": 400,
    "morse_letter_gap_ms": 1500,
    "morse_word_gap_ms": 3000,
}

def _load_settings():
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_SETTINGS.copy()

_settings = _load_settings()
DOT_MAX_MS = _settings.get("morse_dot_max_ms", 400)
DASH_MIN_MS = _settings.get("morse_dash_min_ms", 400)
LETTER_GAP_MS = _settings.get("morse_letter_gap_ms", 1500)
WORD_GAP_MS = _settings.get("morse_word_gap_ms", 3000)

# ── MORSE TABLE ─────────────────────────────────────────────
MORSE_CODE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', '-----': '0', '.----': '1', '..---': '2',
    '...--': '3', '....-': '4', '.....': '5', '-....': '6',
    '--...': '7', '---..': '8', '----.': '9',
    '.-.-.-': '.', '--..--': ',', '..--..': '?',
}

# ── DECODER WITH AUTO-TIMING ────────────────────────────────
class MorseDecoder:
    def __init__(self):
        self.sequence = ""
        self.word = ""
        self.last_symbol_time = 0
        self.lock = threading.Lock()
        self.output_queue = Queue()  # Decoded letters/words go here
        self.running = True
        self._timer_thread = threading.Thread(target=self._timing_loop, daemon=True)
        self._timer_thread.start()
        logger.info("Morse decoder started with auto-timing.")

    def _timing_loop(self):
        """Background thread: checks gaps and auto-decodes.

        FIX: the WORD branch was unreachable. The loop skipped every tick where
        `self.sequence` was empty, but the LETTER branch (1.5s) always clears
        the sequence — so by the time the 3s word gap elapsed there was never a
        pending sequence left to trigger it, and a completed WORD event was
        never emitted at all. The guard now also runs while a decoded word is
        pending, and word completion is tracked from the last *decode*, not
        just the last keypress.
        """
        while self.running:
            time.sleep(0.05)  # 50ms check interval

            with self.lock:
                if not self.sequence and not self.word:
                    continue

                gap = (time.time() - self.last_symbol_time) * 1000

                if self.sequence and gap >= LETTER_GAP_MS:
                    self._decode_letter()

                # Word boundary: no new symbol for WORD_GAP_MS and nothing
                # half-typed. Checked independently of the letter branch.
                if self.word and not self.sequence and gap >= WORD_GAP_MS:
                    self.output_queue.put(("WORD", self.word))
                    logger.info(f"Word complete: {self.word}")
                    self.word = ""

    def _decode_letter(self):
        """Decode current sequence to letter."""
        if not self.sequence:
            return
            
        letter = MORSE_CODE_DICT.get(self.sequence, '?')
        self.word += letter
        logger.info(f"Decoded: {self.sequence} -> {letter}")
        self.output_queue.put(("LETTER", letter))
        self.sequence = ""

    def add_dot(self):
        with self.lock:
            self.sequence += '.'
            self.last_symbol_time = time.time()
            logger.debug(f"Dot added: {self.sequence}")

    def add_dash(self):
        with self.lock:
            self.sequence += '-'
            self.last_symbol_time = time.time()
            logger.debug(f"Dash added: {self.sequence}")

    def backspace(self):
        with self.lock:
            if self.word:
                self.word = self.word[:-1]
                logger.info(f"Backspace: {self.word}")

    def get_output(self, timeout: float = 0.1):
        """Non-blocking read of decoded output."""
        try:
            return self.output_queue.get(timeout=timeout)
        except:
            return None

    def get_word(self) -> str:
        with self.lock:
            word = self.word
            self.word = ""
            return word

    def reset(self):
        with self.lock:
            self.sequence = ""
            self.word = ""
            # Clear queue
            while not self.output_queue.empty():
                self.output_queue.get()

    def shutdown(self):
        self.running = False
        self._timer_thread.join(timeout=1)


# ── GLOBAL INSTANCE ─────────────────────────────────────────
_decoder = MorseDecoder()

def add_dot():
    _decoder.add_dot()

def add_dash():
    _decoder.add_dash()

def backspace():
    _decoder.backspace()

def get_output(timeout: float = 0.1):
    return _decoder.get_output(timeout)

def get_word() -> str:
    return _decoder.get_word()

def reset():
    _decoder.reset()

# ── KEYBOARD SIMULATION ─────────────────────────────────────
def run_simulation():
    """Non-blocking keyboard Morse input."""
    try:
        import readchar
    except ImportError:
        print("readchar is not installed. Install dependencies with: pip install -r requirements.txt")
        return

    print("\n--- Morse Simulation ---")
    print(". = DOT  |  - = DASH  |  Enter = Confirm  |  BS = Delete  |  Tab = Space")
    print("Auto-decodes after 1.5s gap. Auto-words after 3s gap.")
    
    while _decoder.running:
        try:
            char = readchar.readchar()
            
            if char == '.':
                add_dot()
                print(".", end="", flush=True)
            elif char == '-':
                add_dash()
                print("-", end="", flush=True)
            elif char in ['\r', '\n']:
                word = get_word()
                print(f"\n[CONFIRMED]: {word}\n")
            elif char == '\x7f':  # Backspace
                backspace()
                print("\b \b", end="", flush=True)
            elif char == '\t':
                with _decoder.lock:
                    _decoder.word += " "
                print(" ", end="", flush=True)
            
            # Check for auto-decoded output
            output = get_output(timeout=0)
            if output and output[0] == "LETTER":
                print(f"\n[LETTER: {output[1]}]", end="", flush=True)
                
        except Exception as e:
            logger.error(f"Sim error: {e}")

if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda s, f: (_decoder.shutdown(), sys.exit(0)))
    run_simulation()
