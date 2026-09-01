"""
morse_serial.py — BlindAssist Project (BRIDGE MODULE)
========================================================
This file is NEW — it did not exist in the teammate's zip.

Your physical hardware has 3 tactile buttons wired to a Raspberry Pi
Pico W (per the hardware wiring guide), not directly to the Pi 5's
own GPIO pins. The Pico W runs its own firmware (pico_firmware/main.py)
that decodes button timing and sends short text messages to the Pi 5
over a USB-serial link (e.g. /dev/ttyACM0).

The teammate's modules/morse.py only supports typing dots and dashes
on a keyboard — it never reads the Pico W at all. This module bridges
that gap: it opens the serial port, reads incoming messages in a
background thread, and exposes simple functions that main.py and
confidential_mode.py can call.

Falls back gracefully (raises a clear error that main.py already
catches) if no Pico W is connected — e.g. while you are coding on a
laptop without the hardware attached.

RECONNECT BEHAVIOUR
-------------------
A USB-serial link is not permanent. If the Pico W resets, the cable is
nudged, or another process grabs the port, pyserial raises
"device reports readiness to read but returned no data". The old code
caught that, slept, and retried forever on the SAME dead handle — so the
buttons went permanently deaf while the app carried on as if nothing had
happened. For a blind user that is the worst possible failure: silence
is indistinguishable from "I did not press hard enough".

The reader thread now closes the dead handle and re-opens the port,
backing off between attempts. It re-globs the device path every time,
because a Pico that reboots often comes back as a DIFFERENT node
(/dev/ttyACM0 -> /dev/ttyACM1). Repeated identical failures collapse
into one log line plus a periodic summary instead of spamming the log.
Callers can poll is_connected() — or pass on_state_change — so the user
can actually be TOLD the buttons stopped working.
"""

import glob
import json
import logging
import queue
import threading
import time
from pathlib import Path

import serial

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "morse_serial.log"
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("MorseSerialModule")

_settings = {}
try:
    with open(CONFIG_PATH, 'r') as f:
        _settings = json.load(f)
except Exception:
    pass

# Configured port wins over auto-detection; "" or absent means auto-detect.
MORSE_PORT = str(_settings.get("morse_port", "") or "").strip()

# Reconnect pacing. Starts fast (a Pico reset re-enumerates in ~1-2s) and
# backs off to RECONNECT_MAX so an unplugged cable does not burn CPU or
# flood the log for hours.
RECONNECT_BACKOFF_START = 0.5
RECONNECT_BACKOFF_MAX = 5.0
# While the link is down, emit one "still down" line every N seconds
# rather than one per failed attempt.
DOWN_LOG_INTERVAL = 30.0


def find_pico_port():
    """
    Resolve the Pico W's serial device path.

    Honours the "morse_port" setting when present so a Pi with several
    USB-serial gadgets attached can pin the right one; otherwise falls
    back to the lowest-numbered /dev/ttyACM* node.
    """
    if MORSE_PORT:
        return MORSE_PORT
    candidates = sorted(glob.glob("/dev/ttyACM*"))
    if not candidates:
        raise RuntimeError("No Pico W detected. Check the USB cable connection.")
    return candidates[0]


class MorseSerial:
    def __init__(self, baudrate: int = 115200, on_state_change=None):
        """
        on_state_change(connected: bool, detail: str) — optional callback
        fired when the link drops or comes back, so main.py can speak a
        warning. Never called from the caller's thread: it runs on the
        reader thread, so keep it short and non-blocking (tts.speak() is
        already queue-based, so it is safe).
        """
        self.baudrate = baudrate
        self.on_state_change = on_state_change
        self.port = find_pico_port()
        self.ser = serial.Serial(self.port, baudrate, timeout=1)
        self.message_queue = queue.Queue()
        self.running = True
        self.current_word = ""
        self._connected = True
        self._last_error = None
        self._last_down_log = 0.0
        self._failed_attempts = 0
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info(f"Connected to Pico W on {self.port}")
        print(f"[morse_serial] Connected to Pico W on {self.port}")

    # ── link state ───────────────────────────────────────────
    def is_connected(self) -> bool:
        """True while the reader thread is successfully talking to the Pico."""
        return self._connected

    def _set_connected(self, connected: bool, detail: str = ""):
        if connected == self._connected:
            return
        self._connected = connected
        if self.on_state_change:
            try:
                self.on_state_change(connected, detail)
            except Exception as e:
                logger.warning(f"on_state_change callback failed: {e}")

    def _mark_down(self, error):
        """Record a read failure, collapsing repeated identical errors."""
        text = str(error)
        self._failed_attempts += 1
        now = time.time()
        first_time = text != self._last_error
        if first_time:
            self._last_error = text
            self._last_down_log = now
            logger.error(f"Read error: {text}")
            print(f"[morse_serial] Read error: {text}")
            print("[morse_serial] Buttons are unresponsive — attempting to reconnect...")
        elif now - self._last_down_log >= DOWN_LOG_INTERVAL:
            self._last_down_log = now
            msg = f"Still disconnected after {self._failed_attempts} attempts: {text}"
            logger.error(msg)
            print(f"[morse_serial] {msg}")
        self._set_connected(False, text)

    def _mark_up(self):
        """Called after a clean read — the link is healthy."""
        if self._failed_attempts:
            logger.info(f"Reconnected after {self._failed_attempts} failed attempts.")
            print("[morse_serial] Reconnected — buttons are live again.")
        self._failed_attempts = 0
        self._last_error = None
        self._set_connected(True, "")

    def _reconnect(self) -> bool:
        """
        Drop the dead handle and re-open the port. Re-globs the path first:
        a Pico that reboots can come back on a different /dev/ttyACM node,
        so reusing the cached path would fail forever.
        """
        try:
            self.ser.close()
        except Exception:
            pass
        try:
            self.port = find_pico_port()
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            return True
        except Exception:
            # Port not back yet — caller backs off and retries.
            return False

    def _read_loop(self):
        backoff = RECONNECT_BACKOFF_START
        while self.running:
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    self.message_queue.put(line)
                    self._mark_up()
                    backoff = RECONNECT_BACKOFF_START
            except Exception as e:
                # close() flips self.running first, so a shutdown race that
                # trips readline() must not be reported as a fault.
                if not self.running:
                    break
                self._mark_down(e)
                # Always pause BEFORE retrying, and only ever reset the
                # backoff on a genuinely successful read (see _mark_up
                # above). Re-opening the port is not proof the link works:
                # when ModemManager is holding /dev/ttyACM0, or the cable is
                # half-dead, open() succeeds and every read still fails —
                # treating that as recovery spins this thread at 100% CPU.
                time.sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX)
                self._reconnect()

    def get_message(self, timeout=None):
        """Blocking or non-blocking read of the next raw message, e.g. 'LETTER:A'."""
        try:
            return self.message_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def wait_for_raw_button(self, timeout=8):
        """
        Used by Confidential Mode: waits up to `timeout` seconds for a
        RAW:1, RAW:2, or RAW:3 message and returns the button number
        (int) or None if nothing was pressed in time.
        Button 1 = DOT button   -> used as "PRIVATE"
        Button 2 = DASH button  -> used as "NORMAL"
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = deadline - time.time()
            msg = self.get_message(timeout=max(0.1, remaining))
            if msg and msg.startswith("RAW:"):
                digits = "".join(ch for ch in msg.split(":", 1)[1] if ch.isdigit())
                if digits:
                    return int(digits[0])
                logger.debug(f"Ignoring malformed raw button message: {msg!r}")
        return None

    def type_word(self, timeout=30, on_update=None):
        """
        Collects LETTER / WORD_SPACE / BACKSPACE messages into a growing
        string until a CONFIRM message arrives (double-press of Button 3),
        or the timeout expires. Returns the final typed string.

        on_update(event, letter_or_none, current_text) is called after
        every change so the caller can speak live feedback — e.g.
        main.py uses this to say each letter and each finished word out
        loud, since a blind user gets no other confirmation of what was
        registered. event is one of: "letter", "space", "backspace".
        """
        self.current_word = ""
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = deadline - time.time()
            msg = self.get_message(timeout=max(0.1, remaining))
            if msg is None:
                continue
            if msg.startswith("LETTER:"):
                letter = msg.split(":")[1]
                self.current_word += letter
                deadline = time.time() + timeout  # reset idle timeout on activity
                if on_update:
                    on_update("letter", letter, self.current_word)
            elif msg == "WORD_SPACE":
                self.current_word += " "
                deadline = time.time() + timeout
                if on_update:
                    on_update("space", None, self.current_word)
            elif msg == "BACKSPACE":
                self.current_word = self.current_word[:-1]
                deadline = time.time() + timeout
                if on_update:
                    on_update("backspace", None, self.current_word)
            elif msg == "CONFIRM":
                return self.current_word
        return self.current_word

    def read_single_letter(self, timeout=None):
        """
        Waits for one decoded character (letter OR digit) and returns it
        as a string, ignoring RAW/WORD_SPACE messages along the way.
        Returns None on timeout. Kept for backward compatibility — menu
        selection itself now uses read_menu_digit() below, which is safer.
        """
        deadline = time.time() + timeout if timeout else None
        while True:
            if deadline is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                msg = self.get_message(timeout=remaining)
            else:
                msg = self.get_message(timeout=1)
            if msg is None:
                if deadline is None:
                    continue
                return None
            if msg.startswith("LETTER:"):
                letter = msg.split(":")[1]
                if letter and letter != "?":
                    return letter

    def read_menu_digit(self, timeout=None):
        """
        MENU SELECTION (hands-free): waits for one Morse DIGIT (0-9) and
        returns it as a single-character string, or None on timeout.

        Why digits and not letters: every Morse digit is exactly 5
        dots/dashes long; every Morse letter is 1-4 symbols long. That
        length difference means a digit can never be mistaken for a
        letter (or vice versa) — there is no possible ambiguity between
        "pick a menu item" and "type a word", even in principle. If the
        user accidentally taps out a letter while at the menu (e.g. a
        stray press), it is simply ignored and we keep waiting, instead
        of guessing at an unintended selection.
        """
        deadline = time.time() + timeout if timeout else None
        while True:
            if deadline is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                msg = self.get_message(timeout=remaining)
            else:
                msg = self.get_message(timeout=1)
            if msg is None:
                if deadline is None:
                    continue
                return None
            if msg.startswith("LETTER:"):
                char = msg.split(":")[1]
                if char.isdigit():
                    return char
                # a real letter was tapped at the menu — ignore it and
                # keep listening instead of treating it as a selection.

    def wait_for_confirm(self, timeout=5):
        """
        SAFETY GATE for destructive actions (shutdown): waits for a
        CONFIRM message (double-press of Button 3, the same gesture used
        to send a typed sentence) within `timeout` seconds. Returns True
        if confirmed, False if the window expired (i.e. treat as
        cancelled — nothing destructive happens on a timeout or a
        single accidental tap).
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = deadline - time.time()
            msg = self.get_message(timeout=max(0.1, remaining))
            if msg == "CONFIRM":
                return True
        return False

    def close(self):
        self.running = False
        try:
            self.ser.close()
        except Exception:
            pass


# ── Standalone test ─────────────────────────────────────────
if __name__ == "__main__":
    ms = MorseSerial()
    print("Type something using Morse code on the buttons, then double-press Button 3 to confirm...")
    result = ms.type_word()
    print(f"You typed: {result}")
