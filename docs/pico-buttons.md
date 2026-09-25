# Pico buttons and mode selection

At the main menu, Button 1 once and Button 2 four times is `.----`, digit 1,
which starts OCR. Fully release each button between presses. The host now counts
five RAW dot/dash events and tolerates up to five seconds between them. It ignores
interleaved decoded letters, so the firmware's 1.5-second letter timeout cannot
split a slowly entered menu digit. A longer pause resets the partial digit.
`morse_menu_symbol_gap_seconds` in host settings changes this menu-only limit.
Ordinary Morse text entry retains the 1.5-second letter pause.

The replacement `pico_firmware/main.py` debounces all three buttons without
blocking waits. Short Button 3 clicks send RAW:3, a single click sends WORD_SPACE
after the double-click window, a double click sends CONFIRM, and a hold sends
BACKSPACE once. The last pending letter is flushed before space/confirm.
RAW:3 is emitted at press time, including a press subsequently held down.

## Apply and verify

1. Copy the updated host files to the computer running BlindAssist and restart
   the app. Install its requirements in the same Python environment; `pyserial`
   is required. This workspace's `.venv` now has it installed.
2. In Thonny, select the MicroPython interpreter for the actual Pico board and
   save `pico_firmware/main.py` onto that board as `main.py`. This is separate
   from the application's `main.py` on the host computer. Restart the Pico.
3. In Thonny's serial shell, observe RAW:1 for Button 1 and RAW:2 for Button 2.
   A quickly entered dot and four dashes, followed by a pause, produces LETTER:1.
   READY appears at firmware startup, so opening a terminal later may miss it.
4. Close/disconnect Thonny's serial connection before running BlindAssist.
   Confirm the host reports a connected Pico port, then enter the five presses.

The default firmware expects GP0/GP1/GP2 buttons connected to 3.3V on press.
For buttons wired to GND, set ACTIVE_LOW=True in the firmware before uploading;
this enables internal pull-ups. GPIO numbers are not physical header numbers.
The pin and timer APIs follow the [MicroPython RP2 reference](https://docs.micropython.org/en/latest/rp2/quickref.html).

An LED blink proves firmware activity, not that the host received serial data.
If the app reports no Pico, check its Python environment, a USB data cable, and
the configured `morse_port`. `/dev/ttyACM0` is a Linux path, not a Mac path.
Set `morse_port` to the actual device, or an empty string to enable the existing
auto-detection (avoid auto-detection when multiple serial gadgets are attached).
The current workspace has no visible USB modem device, so these hardware checks
and uploading the replacement firmware still need to be performed on the device.
