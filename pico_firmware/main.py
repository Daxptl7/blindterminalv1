# ============================================================
# AET / BlindAssist — Pico W Morse Code Firmware
# Reads 3 tactile buttons, decodes Morse code, sends results
# to the Raspberry Pi 5 over USB serial (115200 baud).
#
# UPLOAD THIS FILE TO THE PICO W ITSELF (using Thonny), NOT to
# the Raspberry Pi. Save it as main.py on the Pico so it runs
# automatically every time the Pico is powered on.
# ============================================================
import machine
import time

# ---- Pin setup ----
DOT_PIN = machine.Pin(0, machine.Pin.IN, machine.Pin.PULL_DOWN)      # Button 1
DASH_PIN = machine.Pin(1, machine.Pin.IN, machine.Pin.PULL_DOWN)     # Button 2
CONFIRM_PIN = machine.Pin(2, machine.Pin.IN, machine.Pin.PULL_DOWN)  # Button 3
led = machine.Pin("LED", machine.Pin.OUT)

# ---- Timing thresholds (milliseconds) ----
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

morse_buffer = ""
last_symbol_time = time.ticks_ms()
last_confirm_press_time = 0
confirm_press_count = 0


def send(msg):
    """Send one line to the Pi 5 over USB serial and blink the LED."""
    print(msg)
    led.value(1)
    time.sleep_ms(20)
    led.value(0)


def wait_for_release(pin):
    """Block until the given pin returns to 0, with debounce."""
    time.sleep_ms(DEBOUNCE_MS)
    while pin.value() == 1:
        time.sleep_ms(5)


def handle_dot_dash():
    global morse_buffer, last_symbol_time
    if DOT_PIN.value() == 1:
        wait_for_release(DOT_PIN)
        morse_buffer += "."
        last_symbol_time = time.ticks_ms()
        send("RAW:1")
    elif DASH_PIN.value() == 1:
        wait_for_release(DASH_PIN)
        morse_buffer += "-"
        last_symbol_time = time.ticks_ms()
        send("RAW:2")


def handle_confirm_button():
    global confirm_press_count, last_confirm_press_time, morse_buffer
    if CONFIRM_PIN.value() == 1:
        press_start = time.ticks_ms()
        while CONFIRM_PIN.value() == 1:
            time.sleep_ms(5)
            if time.ticks_diff(time.ticks_ms(), press_start) > LONG_PRESS_MS:
                wait_for_release(CONFIRM_PIN)
                send("BACKSPACE")
                morse_buffer = ""
                return
        send("RAW:3")
        now = time.ticks_ms()
        if time.ticks_diff(now, last_confirm_press_time) < DOUBLE_PRESS_MS:
            confirm_press_count += 1
        else:
            confirm_press_count = 1
        last_confirm_press_time = now
        if confirm_press_count >= 2:
            send("CONFIRM")
            confirm_press_count = 0
            morse_buffer = ""
        else:
            time.sleep_ms(DOUBLE_PRESS_MS)
            if confirm_press_count == 1:
                send("WORD_SPACE")
                confirm_press_count = 0


def check_letter_timeout():
    global morse_buffer, last_symbol_time
    if morse_buffer and time.ticks_diff(time.ticks_ms(), last_symbol_time) > LETTER_GAP_MS:
        letter = MORSE_TABLE.get(morse_buffer, None)
        send("LETTER:" + letter if letter else "LETTER:?")
        morse_buffer = ""


send("READY")

while True:
    handle_dot_dash()
    handle_confirm_button()
    check_letter_timeout()
    time.sleep_ms(10)
