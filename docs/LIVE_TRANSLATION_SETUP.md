# Live speech translation on Raspberry Pi

Mode 4 → press Button 3 once and wait five seconds for live conversation → select:

1. English ↔ Gujarati
2. English ↔ Hindi
3. Hindi ↔ Gujarati

Wait for the spoken **Live translation ready** message. Speak one person at a
time, pausing at the end of a sentence. Press Button 3 three times within five
seconds to leave Translate and return to the main menu. A single Button 3 press
also waits five seconds in language-pair menus so it cannot consume an exit
gesture. The interpreter translates questions rather than answering them.

On a laptop, Enter ends the current live session. When a session ends without
the exit gesture, Button 1 repeats using live input and Button 2 returns to the
audio/Morse/live selector inside Translate. See [translation controls](pico-buttons.md)
for the full repeat and exit behavior.

## Install on a Pi

Use 64-bit Raspberry Pi OS with a Pi-native Python environment. Live speech
runs in Gemini over the internet; no large local speech model is loaded.
It requires a working Gemini API key with access to `gemini-3.8-live`.

```bash
sudo apt install portaudio19-dev libportaudio2 alsa-utils
./install.sh pi
.venv/bin/python scripts/check_live_translation.py --list-devices
```

For an existing environment, install the updated requirements. Live audio now
requires `sounddevice`, included in both the core and Pi dependency sets.
Python 3.13 uses `audioop-lts` for streaming speaker sample-rate conversion.
Do not copy a Mac virtual environment to the Pi.

Example settings (merge into `config/settings.json`):

```json
{
  "live_translation_model": "gemini-3.8-live",
  "live_input_device": null,
  "live_output_device": null,
  "live_mic_gain": 1.0,
  "live_echo_guard": true,
  "live_echo_tail_ms": 300,
  "live_vad_silence_ms": 700,
  "live_connect_timeout_s": 15,
  "live_send_timeout_s": 5,
  "live_response_timeout_s": 25,
  "live_max_reconnects": 2,
  "live_show_transcripts": false,
  "live_translation_glossary": {}
}
```

On Linux, null live devices inherit `mic_device` and `speaker_device`. Numeric
ALSA routes such as `plughw:4,0` are matched to PortAudio's `hw:4,0` device.
Alternatively set a PortAudio device index or unique name from `--list-devices`.
Index **0** is valid. On a laptop, null uses the operating system defaults.
An explicit missing device produces an error rather than selecting another card.

USB microphones and speakers often support only 44.1 or 48 kHz. The interpreter
negotiates mono or stereo at a supported rate. Microphone PCM is sent with its
actual rate, which Gemini supports; stereo capture is downmixed. Gemini's 24 kHz
output is converted continuously to the speaker rate without changing pitch.

## Speaker echo and usable conversation

With `live_echo_guard: true`, microphone packets become silence while translated
speech plays, plus a short tail to prevent speaker echo becoming another turn.
This is **half duplex**, not acoustic echo cancellation: wait for the translation
to finish before the other person speaks. The physical stop button remains active.

For an isolated headset, set `live_echo_guard: false` to permit speech interruption.
Do not enable this with an open speaker near the microphone. Keep gain at 1.0
unless microphone-level measurements show it is needed; high gain clips speech.

A glossary can improve consistent classroom vocabulary, for example:

```json
{
  "live_translation_glossary": {
    "photosynthesis": {
      "hi": "प्रकाश संश्लेषण",
      "gu": "પ્રકાશસંશ્લેષણ"
    }
  }
}
```

The instructions require complete translations, preserve negation and quantities,
and request repetition for unclear speech. These are model instructions, not a
proof of semantic accuracy. Audio is streamed before a semantic review can occur.

## Preflight and fault recovery

Default checks do not record or contact the API:

```bash
.venv/bin/python scripts/check_live_translation.py --list-devices
```

These probes are explicit hardware/network operations:

```bash
# One second of microphone capture in memory, followed by a quiet speaker tone.
.venv/bin/python scripts/check_live_translation.py --probe-audio
# API key/model/configuration plus synthetic silence; no microphone input.
.venv/bin/python scripts/check_live_translation.py --probe-api
```

Input and output queues are bounded. Lost microphone audio, a stopped capture
stream, playback failure, a silent response, and network send timeouts produce
errors rather than indefinite listening. Temporary network failures get at most
two reconnect attempts. Repeat the last sentence after reconnecting: partial
turns are discarded and never replayed automatically. Authentication/model/setup
errors require configuration repair. Long server sessions may also reconnect.

The main application speaks readiness and errors. Transcripts are hidden by
default and this module does not write conversation audio or text to disk.
`live_show_transcripts` explicitly prints transcripts to the terminal. Private
sessions reject cloud live interpretation before opening audio or a connection.

## Measure accuracy on the actual device

Automated tests:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_live_translator.py' -v
```

For an API benchmark, create a JSONL manifest with one utterance per WAV. WAVs
must contain PCM16 mono/stereo, use 16/24/44.1/48 kHz, and last at most 30 seconds.
Paths are relative to the manifest. This command sends those clips to Gemini and
saves translated WAVs/transcripts, so use evaluation material intended for it.

```json
{"id":"en-hi-001","src":"en","dest":"hi","audio":"en-hi-001.wav","source_text":"Do not open the door.","references":["दरवाज़ा मत खोलिए।"]}
```

```bash
.venv/bin/python scripts/benchmark_live_translation.py path/to/live-eval.jsonl \
  --output-dir runtime/live-evaluation
```

Results include translated audio, source/output transcripts, first-audio and total
latency, reference chrF, and deterministic script/number checks. A native reviewer
must listen to the saved audio and score meaning, completeness, pronunciation,
negation, names, quantities, and correct direction. To bind a human score to an
exact output, copy `output_sha256` into `reviewed_output_sha256` and set the
manifest's boolean `human_adequate`; changed output invalidates the old review.

Use the six-direction, 500 reviewed examples per direction and 95% adequacy
criteria in [translation validation](TRANSLATION_VALIDATION.md), with live speech
recorded by native speakers. Include Pi microphone recordings, room noise,
accents, mixed-language speech, and quiet speech. Report p50/p95 end-of-utterance
to audible translation latency separately; the automated first-audio metric is
measured from the beginning of the supplied clip and excludes physical playback.

Before release, perform at least a 30-minute Pi conversation trial, repeated mode
switches, stop during connection/playback, Wi-Fi loss, microphone/speaker unplug,
and CPU-load tests. Synthetic speech smoke checks and laptop tests establish
neither field accuracy nor Raspberry Pi audio reliability.

API/audio references:

- [Gemini Live audio formats and activity detection](https://ai.google.dev/gemini-api/docs/live-api/capabilities)
- [Live session limits](https://ai.google.dev/gemini-api/docs/live-api/session-management)
- [PortAudio raw streams](https://python-sounddevice.readthedocs.io/en/latest/api/raw-streams.html)
