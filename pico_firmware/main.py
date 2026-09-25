# BlindAssist Pico / Pico W firmware. Save as main.py ON THE PICO.
# Buttons: GP0, GP1, GP2 to 3.3V when pressed (internal pull-down).
# Set ACTIVE_LOW=True only for buttons wired between GPIO and GND.
import time

ACTIVE_LOW = False
LETTER_GAP_MS = 1500
LONG_PRESS_MS = 400
DOUBLE_PRESS_MS = 500
DEBOUNCE_MS = 30

MORSE_TABLE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
}


class ButtonController:
    """Debounced edge polling: no button or double-click wait blocks another."""
    def __init__(self, send, ticks_diff=None):
        self.send = send
        self.diff = ticks_diff or time.ticks_diff
        self.raw = [False] * 3
        self.stable = [False] * 3
        self.changed = [0] * 3
        self.pressed = [0] * 3
        self.long_sent = False
        self.pending_confirm = None
        self.morse_buffer = ""
        self.last_symbol = 0

    def flush_letter(self):
        if self.morse_buffer:
            self.send("LETTER:" + MORSE_TABLE.get(self.morse_buffer, "?"))
            self.morse_buffer = ""

    def update(self, values, now):
        for index, value in enumerate(values):
            value = bool(value)
            if value != self.raw[index]:
                self.raw[index] = value
                self.changed[index] = now
            if value != self.stable[index] and self.diff(now, self.changed[index]) >= DEBOUNCE_MS:
                self.stable[index] = value
                if value:
                    self.pressed[index] = now
                    self.send("RAW:" + str(index + 1))
                    if index < 2:
                        self.morse_buffer += "." if index == 0 else "-"
                        self.last_symbol = now
                    else:
                        self.long_sent = False
                elif index == 2:
                    # Also classify here if a slow polling iteration skipped
                    # the long-press threshold while the button was down.
                    if not self.long_sent and self.diff(now, self.pressed[2]) >= LONG_PRESS_MS:
                        self.backspace()
                    if not self.long_sent:
                        if self.pending_confirm is not None and self.diff(now, self.pending_confirm) <= DOUBLE_PRESS_MS:
                            self.flush_letter()
                            self.send("CONFIRM")
                            self.pending_confirm = None
                        else:
                            self.pending_confirm = now
        if self.stable[2] and not self.long_sent and self.diff(now, self.pressed[2]) >= LONG_PRESS_MS:
            self.backspace()
        # A second press in progress is allowed to finish before deciding
        # whether it was a double click or a hold.
        if self.pending_confirm is not None and not self.raw[2] and not self.stable[2] and self.diff(now, self.pending_confirm) > DOUBLE_PRESS_MS:
            self.flush_letter()
            self.send("WORD_SPACE")
            self.pending_confirm = None
        if self.morse_buffer and not any(self.stable[:2]) and self.diff(now, self.last_symbol) >= LETTER_GAP_MS:
            self.flush_letter()

    def backspace(self):
        self.send("BACKSPACE")
        self.morse_buffer = ""
        self.pending_confirm = None
        self.long_sent = True


def run():
    import machine
    pull = machine.Pin.PULL_UP if ACTIVE_LOW else machine.Pin.PULL_DOWN
    pins = [machine.Pin(number, machine.Pin.IN, pull) for number in (0, 1, 2)]
    try:
        led = machine.Pin("LED", machine.Pin.OUT)
    except (TypeError, ValueError):
        led = machine.Pin(25, machine.Pin.OUT)  # original Pico
    led_until = [None]

    def send(message):
        print(message)
        led.value(1)
        led_until[0] = time.ticks_add(time.ticks_ms(), 20)

    controller = ButtonController(send)
    send("READY")
    while True:
        now = time.ticks_ms()
        controller.update([not pin.value() if ACTIVE_LOW else pin.value() for pin in pins], now)
        if led_until[0] is not None and time.ticks_diff(now, led_until[0]) >= 0:
            led.value(0)
            led_until[0] = None
        time.sleep_ms(5)


if __name__ == "__main__":
    run()
