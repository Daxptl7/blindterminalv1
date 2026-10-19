# Local-language translation setup

BlindAssist now exposes all six English, Hindi, and Gujarati directions. The
provider order tries cloud services first, then installed local engines and the
offline phrasebook. There is no online/offline question at mode entry. Earphones
or speaker are chosen only after the result exists and only for that result.
Explicit private API requests still exclude cloud services and persistent cache;
the user-facing earphone choice does not create such a processing session.

For real-time speech interpretation, follow
[live translation setup](LIVE_TRANSLATION_SETUP.md). It has its own Gemini
streaming connection, audio-device preflight, and recorded-speech benchmark.

## Translation buttons on the Pi

Inside Mode 4, press one physical button once:

- Button 1: speak a phrase. Choose source and target languages before recording.
- Button 2: enter English text using the Morse buttons.
- Button 3 once, then wait five seconds: choose a live conversation language pair.

There is no typed-text option or fourth button. Voice/Morse language menus use
1 for English, 2 for Hindi and 3 for Gujarati. Finish a phrase recording with
one Button 3 press followed by five seconds. Three consecutive Button 3 presses
within five seconds leave Translate, including during live conversation.
After a phrase translation is processed, select earphones or speaker, then answer
the repeat question: Button 1 repeats the same input method; Button 2 returns to
the audio/Morse/live selector. See [button controls](pico-buttons.md).
Each new menu lets the previous press settle before listening, then preserves
choices made during the spoken prompt. Delayed events from the previous choice
are cleared between menus. The log reports the input method actually selected.

## Laptop phase

Use Python 3.10 or newer in a separate translation environment. The current
IndicTransToolkit release provides macOS ARM wheels through Python 3.13. From
the project root:

```bash
python3 -m venv .venv-translation
source .venv-translation/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-translation.txt
```

Download the three AI4Bharat IndicTrans2 distilled model snapshots only after
accepting their Hugging Face terms. Keep them out of Git:

```text
models_local/indictrans2/en-indic
models_local/indictrans2/indic-en
models_local/indictrans2/indic-indic
```

The corresponding model IDs are:

- `ai4bharat/indictrans2-en-indic-dist-200M`
- `ai4bharat/indictrans2-indic-en-dist-200M`
- `ai4bharat/indictrans2-indic-indic-dist-320M`

After accepting access to all three model pages and authenticating, download
them with the Hugging Face CLI:

```bash
hf auth login
hf download ai4bharat/indictrans2-en-indic-dist-200M --local-dir models_local/indictrans2/en-indic
hf download ai4bharat/indictrans2-indic-en-dist-200M --local-dir models_local/indictrans2/indic-en
hf download ai4bharat/indictrans2-indic-indic-dist-320M --local-dir models_local/indictrans2/indic-indic
```

Check `df -h .` first and keep several gigabytes free. The current model files
alone are larger than a minimal laptop or Pi installation.

Downloads are disabled in runtime settings by default. This prevents a private
request from unexpectedly accessing the network or filling a Pi SD card.

Run `python selftest.py`. It reports each missing bundle without making any
network request. Then test one paragraph in every direction and run the
benchmark described in `docs/TRANSLATION_VALIDATION.md`.

## Speech input and output

Translation quality and speech-recognition quality are separate. Install Vosk
models in these language-specific directories:

```text
models_local/vosk/en
models_local/vosk/hi
models_local/vosk/gu
```

The code will not feed Hindi or Gujarati audio into an English offline model.
The online recognizer is tried first; local models are the automatic fallback.
If neither is available, the app reports that recognition failed. Install `espeak-ng` on Raspberry
Pi OS so offline Hindi/Gujarati translation can also be spoken offline.

## Raspberry Pi phase

Do not copy the laptop virtual environment. Pull the same source commit, create
a Pi-native environment, and install packages there. Measure RAM, temperature,
and p95 latency with the real three model bundles. If the Transformers runtime
does not meet the target, convert and validate a CTranslate2 INT8 deployment as
a separate optimization phase; never assume quantization preserves quality.
