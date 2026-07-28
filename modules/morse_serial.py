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
"""

import glob
import queue
import threading
import time

import serial


def find_pico_port():
    """Auto-detect the Pico W's serial device path."""
    candidates = sorted(glob.glob("/dev/ttyACM*"))
    if not candidates:
        raise RuntimeError("No Pico W detected. Check the USB cable connection.")
    return candidates[0]


class MorseSerial:
    def __init__(self, baudrate: int = 115200):
        port = find_pico_port()
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.message_queue = queue.Queue()
        self.running = True
        self.current_word = ""
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print(f"[morse_serial] Connected to Pico W on {port}")

    def _read_loop(self):
        while self.running:
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    self.message_queue.put(line)
            except Exception as e:
                print(f"[morse_serial] Read error: {e}")
                time.sleep(0.5)

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
                return int(msg.split(":")[1])
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
