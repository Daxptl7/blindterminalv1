# AET / BlindAssist — Changes vs. the Original Zip (Dax's Code)

This is the full project, ready to unzip directly into `~/blindterminal`
on a fresh Raspberry Pi OS install. Everything below has already been
fixed and tested for valid Python syntax — see
`AET_Master_Software_Guide.md` for full step-by-step setup instructions.

---

# Voice: record until you say stop, and record from the right microphone

Two separate problems, fixed together.

## 1. Every recording stopped after 8 seconds

`voice_record_max_seconds` (8 s) capped *every* capture, and 700 ms of silence
ended the phrase even inside that window — so pausing to think ended the
question. Anything longer than a short command was truncated and the fragment
was what got transcribed.

The modes where you compose a sentence — **Mode 3 (Voice Ask)**, **Mode 7
(Translate → voice input)** and **Mode 9 (Math Solver → voice)** — now record
open-ended: no time limit, no silence cut-off.

**Press Button 3 three times to stop the recording.** The device says
"Take as long as you need. Press button 3 three times when you are finished."
Three presses rather than one so a knock against the button cannot cut a
question short; they must come within 4 s of each other. On a laptop with no
Pico W attached, ENTER stops it instead. A 3-minute backstop
(`voice_manual_max_seconds`) exists only so a stuck button cannot record
forever.

Short one-word answers ("yes", "hindi", a menu choice) still use the old timed
capture — waiting for three button presses to answer "yes" would be worse.

New settings, all in `config/settings.json`:

| key | default | meaning |
|---|---|---|
| `voice_manual_max_seconds` | 180 | backstop for an open-ended recording |
| `voice_stop_press_count` | 3 | presses of Button 3 that end it |
| `voice_stop_press_window_seconds` | 4.0 | max gap between those presses |
| `voice_open_transient_ms` | 250 | audio discarded when a stream opens |

## 2. It was recording from a card with no microphone on it

This Pi has two identical "USB Audio Device" dongles (cards 2 and 3). Only one
has a microphone capsule; the other's capture input floats and picks up 50/100
Hz mains hum at about **-42 dBFS**. That is *louder* than the -60 dBFS silence
gate the old code used, so the hum was accepted as a valid recording, sent to
Vosk and Google, and came back as "I didn't catch that" — for every question,
because ALSA's `default` and PortAudio's default both point at that same card.
The logs show it exactly: `Captured 8.0s at -42 dBFS`, then two engines failing.

- Hum is now told apart from a live input without needing speech in the clip:
  it is stationary (loud and quiet 100 ms frames measure the same) and has the
  crest factor of a sine wave, where even a silent live microphone has noise
  peaking 12+ dB above its own RMS.
- A card caught doing that is tried **last** from then on, so the next attempt
  records from the other dongle. No up-front probing — that would have cost
  several seconds of silence before every question.
- The two capture backends (arecord and PyAudio/VAD) used to pick devices
  independently, so which microphone you got depended on which backend ran.
  They now share one preference order.
- webrtcvad was also triggering on the click a USB stream makes when it opens
  (it armed 90 ms after opening and "ended" the phrase 700 ms later, before
  anyone had spoken). That opening transient is discarded now.

**Run `python3 mic_check.py` once.** It records from each card while you speak
and reports which one hears you, then offers to save it as `mic_device` in
`settings.json` — both backends honour that setting. Until it is set, the
device figures it out by itself but wastes the first attempt doing so.

---

# Voice overhaul + production audit

Run `python3 selftest.py` after any install or hardware change. It reports,
in one place, what works right now and what to do about anything that does
not. `--mic` additionally records for three seconds and reports the captured
level and transcription.

## ⚠️ Action required: rotate the Groq API key

`config/settings.json` is gitignored *now*, but it was committed earlier
(commits `7342980` and `04ea3b9`), so the live `gsk_…` Groq key and the Gemini
key are recoverable from git history by anyone with the repository. Gitignoring
a file does not remove it from history.

1. Revoke and reissue the key at <https://console.groq.com/keys> (and the
   Gemini key at <https://aistudio.google.com/apikey>).
2. Put the new keys only in `config/settings.json` on the device.
3. `config/settings.example.json` is the tracked template — it has the same
   keys with empty values. Keep secrets out of it.

Purging history (`git filter-repo`) is worth doing too, but rotating the key
is the part that actually stops the leak.

## Voice input — why it was not working

Five independent faults, each sufficient on its own to make Modes 3/4/7/9 fail.

1. **`listen()` killed its own recording.** The arecord path started the
   recorder, ran a wait loop *only* when a TTY was attached, then called
   `process.terminate()` unconditionally. The production device is headless, so
   stdin is not a TTY, the loop was skipped and the recorder was terminated
   milliseconds after starting. Every capture produced an essentially empty WAV
   and every transcription failed. Recording now always runs its full window; a
   keypress only shortens it, and `stop()` can abort it from another thread.

2. **The device recorded its own voice.** `tts.speak()` is a non-blocking
   queue, so "Ask your question now" was still coming out of the speaker when
   the microphone opened. The VAD triggered on the device's own prompt and the
   AI was asked to answer the question it had just spoken. `main._listen()` now
   drains the speech queue before opening the microphone.

3. **There was no offline speech recognition at all.** `vosk` and
   `pocketsphinx` were listed in `STT_ENGINES` but neither was installed and
   `models/vosk/` did not exist, so the advertised three-engine chain was
   really Google-only — dead without internet, and silent about why. vosk plus
   the Indian-English model are now installed under `models_local/vosk/`
   (deliberately on the SD card, not the removable USB that `models/` points
   at, because speech is the device's primary input). Engine availability is
   probed once at import and logged as one clear line.

4. **One timeout covered two different things.** The 8-second budget bounded
   "waiting for the user to start speaking" *and* "how long they may speak"
   together, so a user who paused to think lost most of their speaking window.
   These are now separate (`voice_start_timeout_seconds` vs
   `voice_record_max_seconds`), and the phrase clock starts only once speech is
   actually detected.

5. **No level normalization.** A USB capsule on a Pi captures around −35 dBFS
   and the 300 Hz high-pass made it quieter still; every engine degrades badly
   on input that quiet. Audio is now DC-corrected and gain-normalized to
   −20 dBFS before recognition, with the gain capped so a silent room is not
   amplified into noise.

Also in `modules/voice.py`:

- `pyaudio.open()` sat outside the try/finally, so a busy or missing microphone
  raised straight out of the module and took the whole mode down.
- The capture device is resolved against real hardware (`arecord -l`) instead
  of trusting `mic_device: "default"`, silent cards are skipped, and the
  working device is cached and re-probed if it disappears.
- Short frames from a buffer overrun no longer crash `webrtcvad`.
- Sample rates are negotiated (16/48/32/8 kHz) and converted, so USB dongles
  that only offer 44.1/48 kHz work.
- A cough no longer costs a full recognition round-trip (`voice_min_speech_ms`).
- The recordings directory falls back to a writable location instead of
  aborting the capture when the USB stick is not mounted.
- Failures are distinguishable: `get_last_error()` returns a sentence written
  to be read aloud, so "the mic is muted", "no internet", "nothing was said"
  and "I misheard you" no longer all say *"I didn't catch that"*. Hardware
  faults are reported immediately instead of being retried.
- A short beep marks the start of recording, so a blind user can tell the
  listening pause from a crash.

## Audio output — Confidential Mode was not private

`switch_output_device()` called `pygame.mixer.init(devicename="plughw:2,0")`.
SDL enumerates outputs by friendly name (`USB Audio Device Analog Stereo`), not
by ALSA name, so that call could never match a device: every switch threw, the
mixer was re-initialised on the default output, and "private" speech played out
of whichever card ALSA happened to default to. For this product that is a
privacy failure, not a cosmetic one.

Routing is now applied per utterance by a player that can actually target an
ALSA card (`aplay -D`, `mpg123 -a`, or `ffmpeg -f alsa`), with gTTS MP3 decoded
to WAV in memory via `soundfile` because `aplay` speaks WAV only. If no player
can drive the requested card, that is logged as an error rather than silently
played on the wrong one.

`speaker_device` and `bone_device` are also validated against `aplay -l` at
startup — `bone_device` was set to `plughw:4,0`, and this machine has no card 4.

## Configuration corrected against the actual hardware

| Setting | Was | Now | Why |
|---|---|---|---|
| `bone_device` | `plughw:4,0` | `plughw:2,0` | there is no card 4; cards are 2 and 3 |
| `yolo_model_path` | `/mnt/aet_usb/models/yolov8m.pt` | `models/yolov8n.pt` | that path does not exist, and only `yolov8n/s` are on the stick |
| `yolo_confidence` | `0.75` | `0.5` | at 0.75 most real objects were never announced |
| `gemini_model_name` | `gemini-3.5-flash-lite` | `gemini-2.5-flash` | not a real model id, so Gemini always failed |

## Other modules

- **`modules/object_detection.py`** — `_resolve_model_path()` searched only for
  `yolov8m.pt`, missed, and then returned the bad configured path anyway;
  ultralytics cannot auto-download to an arbitrary absolute path, so Mode 6
  failed on first use. It now accepts any available yolov8 weight, preferring
  nano (the only size with a usable frame rate on a Pi 5 CPU), and tolerates an
  unmounted `models/` symlink.
- **`modules/config_loader.py`** — was a bare `open()`/`json.load()`. A missing
  or malformed `settings.json` made every importer fail, and `main.py` then
  reported *Confidential Mode* as unavailable rather than the config as broken.
  It now degrades to `{}` and says which it was.
- **`modules/tts.py`** — fixed a latent `NameError` in `_find_player()`, where
  the ffplay branch's arg-builder closed over `binary` from a loop that had not
  run on that path.
- **`main.py`** — all voice call sites go through `_listen()`; the GPS mode's
  input race was extended from 10s to 25s, because it expired while the
  recogniser was still working and the advertised voice option could never win.

## New files

- `selftest.py` — pre-flight check (exit code 0 = ready, so it can gate a
  systemd unit).
- `config/settings.example.json` — tracked, secret-free config template.
- `models_local/vosk/` — offline speech model (gitignored; 55MB).

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
