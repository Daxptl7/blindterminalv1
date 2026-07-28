# AET / BlindAssist — Changes vs. the Original Zip (Dax's Code)

This is the full project, ready to unzip directly into `~/blindterminal`
on a fresh Raspberry Pi OS install. Everything below has already been
fixed and tested for valid Python syntax — see
`AET_Master_Software_Guide.md` for full step-by-step setup instructions.

## Setup process (not code) — environment fix
- The guide's Section 6.1 now sets up the Python environment with
  **Miniforge/conda** (Python 3.11), not a plain `venv`. A plain venv
  always inherits the Pi's system Python (3.13 on current images), and
  `mediapipe` (Mode 5 - Gesture) does not yet support 3.13, so the venv
  route fails on `pip install mediapipe`. This is a setup-process fix
  only — no application code changed because of it.
- If you previously created `python3 -m venv aet-env`, delete it
  (`rm -rf ~/aet-env`) before following the guide's Section 6.1 — a
  venv and a conda environment can both be named `aet-env` at the same
  time in different folders, which is easy to activate by mistake and
  will point systemd's `ExecStart` (Section 9) at the wrong Python.
- Under Python 3.13 specifically, `import openai` can fail with
  `ModuleNotFoundError: No module named 'cgi'` (the `cgi` module was
  removed in 3.13, per PEP 594, and part of `openai`'s dependency chain
  still imports it). Using the Python 3.11 conda environment from
  Section 6.1 avoids this; `pip install --upgrade openai` or
  `pip install legacy-cgi` are documented fallbacks in the guide's
  Troubleshooting section either way.

## New files
- `modules/morse_serial.py` — reads the 3 physical buttons via the Pico W
  over USB serial (the original zip only supported keyboard-typed Morse).
- `pico_firmware/main.py` — upload this to the Pico W itself (via Thonny),
  not to the Pi. Reads the buttons and talks to morse_serial.py.

## Modified files
- `modules/ocr.py` — **Camera capture no longer depends on the `picamera2`
  Python module.** If your Raspberry Pi OS's default Python is newer than
  what `mediapipe` supports (mediapipe currently caps around Python 3.12),
  you likely created a separate virtual environment with an older Python
  just for mediapipe. Apt-installed `python3-picamera2` / `python3-libcamera`
  are C extensions compiled for ONE specific Python version and cannot be
  imported from a differently-versioned venv — this is a confirmed,
  documented limitation (Raspberry Pi's own engineers: "the libcamera
  library does not have sufficient ABI stability... for different
  versions of Python"), not a bug in this project's code. The fix: capture
  now shells out to the `rpicam-still` command-line tool (ships on every
  Raspberry Pi OS image) and reads back the resulting JPEG with
  `cv2.imread()` — a subprocess call and a file read have no Python-version
  dependency at all, so this works identically from any venv.
  Plain `cv2.VideoCapture()` was deliberately NOT used as the fix, even
  though it seems like the obvious bypass: OpenCV's VideoCapture only
  supports simple V4L2 devices and does not support the libcamera stack
  that Camera Module 3 requires — there is no legacy V4L2 driver for that
  sensor at all, so `cv2.VideoCapture(0)` against the CSI camera will
  simply fail to open (confirmed via Raspberry Pi's own forums and an
  open, still-unresolved OpenCV GitHub issue, #21653). USB webcams are
  unaffected either way, since those already work through V4L2 directly.
- `requirements-pi.txt` — removed `picamera2` (no longer a dependency for
  the reason above). If your own code still needs it directly, install it
  via apt as `python3-picamera2` into a venv created with
  `--system-site-packages` that matches your system Python version exactly.
- `main.py` —
  - **Menu selection now uses DIGITS (1-9), not letters.** Every Morse
    digit is exactly 5 dots/dashes long; every letter is 1-4 symbols —
    so a digit can never be confused with a letter typed as content
    inside a sentence (Mode 2 or Mode 9), even in principle.
  - **Shutdown (0) now requires confirmation.** A single tap no longer
    turns the device off. It asks for a double-press of Button 3 within
    5 seconds; anything else cancels and returns to the menu. Modes 1-9
    still launch on a single tap since they're safe/reversible.
  - **Live spoken feedback while typing.** Previously a blind user got
    no feedback at all until an entire sentence was finished. Now each
    letter is spoken as it's typed, and each finished word is spoken
    when you press Button 3 once (space). Double-press still means
    "done, send the whole sentence" — it never auto-sends after one word.
  - **New Mode 9 — Math Solver**, for square roots, exponents, variables,
    and matrices. Since the buttons have no way to produce symbols like
    "√" or "^", problems are either spoken (Mode 3-style voice input) or
    typed by spelling operations out as words, e.g.
    `SQRT 16 PLUS 3 TIMES X SQUARE MINUS 5 EQUALS 0` or
    `MATRIX 2 BY 2 ROW1 1 2 ROW2 3 4 FIND DETERMINANT`. The AI is
    instructed to answer the same way — full spoken sentences, never
    symbols or LaTeX, since a screen reader/TTS can't read those aloud.
  - Mode 2 (Morse Type) now reads real button presses via morse_serial.py.
  - Mode 1 (OCR) now asks the Confidential Mode privacy question before
    reading scanned text aloud.
  - Added Mode 8 (Confidential Mode demo).
- `modules/morse_serial.py` — added `read_menu_digit()` (digit-only menu
  reading), `wait_for_confirm()` (shutdown safety gate), and an
  `on_update` callback in `type_word()` so main.py can speak live
  feedback per letter/word.
- `modules/confidential_mode.py` — was never called from anywhere in the
  original main.py. Now: reads `use_gpio` from settings.json (was
  hardcoded False), and reads real button presses from the Pico W for the
  PRIVATE/NORMAL choice (was a stub that always returned NORMAL).
- `modules/object_detection.py` — `cv2.imshow`/`cv2.waitKey` were called
  unconditionally, which crashes or hangs on a headless Pi (no monitor —
  exactly how this device runs). Now gated behind a settings flag
  (`object_detection_display`, default false), matching the safe pattern
  already used in `gesture_control.py`. Also now respects
  `yolo_confidence` / `yolo_model_path` from settings.json instead of
  hardcoding them.
- `modules/tts.py` — `bone_device` / `speaker_device` were defined in
  settings.json but never used; audio always played through whatever
  ALSA considered "default". Added `switch_output_device()` /
  `use_bone_conduction_output()` / `use_speaker_output()` so Confidential
  Mode can actually move the audio stream, not just mute a control.
- `config/settings.json` —
  - `gps_port` changed from `/dev/ttyS0` to `/dev/serial0` to match the
    hardware wiring guide's UART setup.
  - `use_gpio` set to `true` (was `false`, correct for laptop dev only).
  - `gesture_camera_index` set to `1` to avoid clashing with the Camera
    Module 3 on the OCR module's default camera index.
  - Added `object_detection_display: false`.

## Confirmed correct, unchanged
- `modules/ai_query.py` — the model name `gemini-3.5-flash` is current
  and valid; do not change it to `gemini-1.5-flash` (deprecated/shut down).
  The multi-provider racing (Groq + OpenAI, falling back to Gemini, then
  offline TinyLlama) is genuinely good design — kept as-is.
- `modules/gps_navigator.py`, `modules/gesture_control.py`,
  `modules/voice.py`, `modules/translator.py`, `modules/ocr.py`,
  `services/` (RAG pipeline) — reviewed, no bugs found.

## One thing to verify on your hardware
Run `ls /dev/ttyACM*` on the Pi. If it shows a device, your buttons are
wired to a Pico W and everything above applies as-is. If it shows nothing,
your buttons are wired directly to the Pi 5's own GPIO pins instead —
`modules/morse_serial.py` would need different code (direct `RPi.GPIO`
edge detection) — let your mentor/teammate know.
