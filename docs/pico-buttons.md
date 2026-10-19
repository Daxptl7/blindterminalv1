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

## Voice Ask follow-up questions

After entering a mode using its Morse digit, processing starts automatically.
AI, speech recognition, OCR, and phrase translation try online services before
their installed offline fallbacks. There is no online/offline selection prompt.
Local fallback requires the appropriate models to be provisioned.

When a completed result is ready, Button 1 selects earphones and Button 2 selects
the speaker. Only that result (including all chunks of a long answer) uses the
selected output. Menus, recording cues, and follow-up prompts stay on the speaker.
Every new answer or explanation gets a fresh output choice after processing.
An unanswered output prompt cancels result playback. Earphone selection never
changes the processing policy, and an unavailable earphone route never causes
the result to be read on the speaker. Live translation remains continuous on its
configured live output and does not ask after every sentence.

While Mode 3 is reading an answer, Button 1 once interrupts the current speech
and its remaining sections. The device goes directly to the follow-up recording
prompt, keeping the generated answer as recent conversation context and noting
that playback was interrupted. The ordinary yes/no prompt is skipped in this
case. Wait for the recording cue, then speak; finish with three Button 3 presses.

After Mode 3 finishes reading its answer, it asks whether to continue:

- Button 1 once: record a follow-up question. Finish recording with three
  Button 3 presses, as for the first question.
- Button 2 once: end Voice Ask and return to the main menu.
- No answer within the configured button-choice timeout: return to the menu.

The question is repeated after every answer, so multiple follow-ups can stay in
one conversation. Recent question/answer context is kept in memory for this
mode only. Each follow-up uses automatic processing and a new output choice.
Old button presses are cleared before the choice prompt,
while a choice made during the prompt is preserved.

## Translation input selection

After entering Mode 4, press Button 1 once for a spoken phrase or Button 2 once
for Morse input. Press Button 3 once and wait five seconds for live conversation.
These are button choices, not five-press Morse digits.

After each translation attempt, Button 1 starts another translation with the
same input method. Button 2 returns to the audio/Morse/live input menu without
leaving Translate. A live session that ends normally also offers this choice.
For phrase translations, the earphone/speaker question comes after processing
and before this repeat question. Output selection is independent of the input
method and does not change the next menu's speaker output.

Three consecutive Button 3 presses within five seconds leave Translate and
return to the application's main menu. This works during menus, voice capture,
Morse entry, processing, speech playback, and live conversation. Fully release
the button between clicks. Another physical button interrupts the sequence.
The firmware's decoded CONFIRM event does not count as an additional press.

Because Button 3 also selects options, its normal actions wait until five
seconds after the first click. This prevents a triple click from starting Live,
submitting Morse text, or selecting Gujarati before it can exit:

- Voice capture: press Button 3 once, then wait five seconds to finish recording.
  A triple click cancels capture and leaves Translate without transcribing it.
- Morse: a single Button 3 click inserts a space; double-click and wait five
  seconds to submit the text. A triple click leaves Translate instead.
- Source/target menus: Button 1 selects English, Button 2 Hindi, and a single
  Button 3 click followed by the five-second wait selects Gujarati.
- Live: choose a language pair as described in
  [live translation setup](LIVE_TRANSLATION_SETUP.md). Triple-click Button 3
  to leave Translate. On a laptop, Enter ends the live session and offers repeat.

Without Pico buttons, type each menu number and press Enter; type `333` at any
translation menu to return to the main menu. Voice recording finishes with Enter.
Other modes retain their existing controls. The host handles this change; the
existing firmware that emits a RAW event for every press does not need updating.

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
