# Local-language translation setup

BlindAssist now exposes all six English, Hindi, and Gujarati directions. The
provider order is offline-first: IndicTrans2, Argos, trusted/local Libre, then
the cloud providers. A private request never uses a cloud service or the
persistent text cache.

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
If a requested local model is absent it uses the configured online recognizer
or reports that the language is unavailable. Install `espeak-ng` on Raspberry
Pi OS so offline Hindi/Gujarati translation can also be spoken offline.

## Raspberry Pi phase

Do not copy the laptop virtual environment. Pull the same source commit, create
a Pi-native environment, and install packages there. Measure RAM, temperature,
and p95 latency with the real three model bundles. If the Transformers runtime
does not meet the target, convert and validate a CTranslate2 INT8 deployment as
a separate optimization phase; never assume quantization preserves quality.
