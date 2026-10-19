"""One serial consumer for Mode 4, including its three-click exit gesture."""

import queue
import threading
import time


class TranslationExited(Exception):
    """The user requested the application's main menu."""


class TranslationButtons:
    def __init__(self, serial, window=5.0):
        self.serial = serial
        self.window = window
        self.exited = threading.Event()
        self._closed = threading.Event()
        self._messages = queue.Queue()
        self._lock = threading.Lock()
        self._pending = []
        self._presses = 0
        self._deadline = None
        self._thread = None

    def __enter__(self):
        if self.serial is not None:
            self._thread = threading.Thread(target=self._read, daemon=True,
                                            name="translation_buttons")
            self._thread.start()
        return self

    def __exit__(self, *args):
        self._closed.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

    def check_exit(self):
        if self.exited.is_set():
            raise TranslationExited()

    def _release_pending(self):
        for message in self._pending:
            self._messages.put(message)
        self._pending.clear()
        self._presses = 0
        self._deadline = None

    def _accept(self, message, now):
        """Delay Button 3 actions until a possible exit gesture is resolved."""
        with self._lock:
            if self._deadline is not None and now >= self._deadline:
                self._release_pending()
            if message is None:
                return
            if message in ("RAW:1", "RAW:2"):
                # Another physical button breaks a consecutive triple click.
                self._release_pending()
            if message == "RAW:3":
                self._presses += 1
                if self._deadline is None:
                    self._deadline = now + self.window
                if self._presses == 3:
                    self._pending.clear()
                    self.exited.set()
                    return
            if self._deadline is not None:
                self._pending.append(message)
            else:
                self._messages.put(message)

    def _read(self):
        while not self._closed.is_set() and not self.exited.is_set():
            try:
                message = self.serial.get_message(timeout=0.05)
            except Exception:
                # An unplugged controller must not trap the user in a menu.
                self.exited.set()
                return
            self._accept(message, time.monotonic())

    def clear(self):
        """Discard previous-stage events before speaking the next prompt."""
        self.check_exit()
        with self._lock:
            self._pending.clear()
            self._presses = 0
            self._deadline = None
            while True:
                try:
                    self._messages.get_nowait()
                except queue.Empty:
                    break

    def get_message(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            self.check_exit()
            remaining = 0.05 if deadline is None else min(0.05, deadline - time.monotonic())
            if remaining <= 0:
                return None
            try:
                message = self._messages.get(timeout=remaining)
                self.check_exit()
                return message
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    return None

    def wait_for_raw_button(self, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            message = self.get_message(timeout=deadline - time.monotonic())
            if message in ("RAW:1", "RAW:2", "RAW:3"):
                return int(message[-1])
        return None

    def recording_stopped(self):
        """One resolved Button 3 press finishes speech; three leave Mode 4."""
        if self.exited.is_set():
            return True
        while True:
            try:
                message = self._messages.get_nowait()
            except queue.Empty:
                return False
            if message == "RAW:3":
                return True

    def type_word(self, timeout=60):
        text = ""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            message = self.get_message(timeout=deadline - time.monotonic())
            if message is None:
                break
            if message.startswith("LETTER:"):
                text += message.split(":", 1)[1]
            elif message == "WORD_SPACE":
                text += " "
            elif message == "BACKSPACE":
                text = text[:-1]
            elif message == "CONFIRM":
                return text
            else:
                continue
            deadline = time.monotonic() + timeout
        return text
