# Device installation and release gates

The code is prepared for device validation. Passing unit tests is not a claim
that camera accuracy, earphone isolation or mobility behavior was validated.
The first release profile excludes GPS navigation. Mode 6 announces recognized
objects and horizontal position; it does not measure distance or certify paths.

## Install

Use a clean checkout on 64-bit Raspberry Pi OS. Python 3.11 is the reference
Pi target; desktop development supports Python 3.10–3.13 subject to available
dependency wheels. Do not reuse a virtual environment copied from another OS.

Install system packages on the Pi:

```bash
sudo apt update
sudo apt install python3-venv python3-dev portaudio19-dev libsdl2-dev \
  alsa-utils mpg123 espeak-ng tesseract-ocr tesseract-ocr-eng \
  tesseract-ocr-hin tesseract-ocr-guj rpicam-apps
./install.sh pi
```

On macOS, install Tesseract and PortAudio through your package manager and run
`./install.sh desktop`. Run `./install.sh --check` for a read-only layout check.
The installer installs BlindAssist in `.venv`, validates dependency consistency,
and writes an exact `deployment.lock.txt` after a successful install. Keep the
lock with the OS/Python version and release image; subsequent installs can use
it. A lock generated on macOS is not a Raspberry Pi lock. Resolve dependencies
on the target platform and validate the resulting image before distributing it.

The installer does not overwrite existing configuration. Edit
`config/settings.json` using `config/settings.example.json` as the canonical
template. The legacy example filename is kept identical by the layout check.
Store secrets only in the ignored device configuration or environment. The
installer restricts a newly created configuration to its owner.

Persistent application data uses `runtime/`; it does not require the legacy
`data` or `models` USB symlinks. Configure another `data_directory` explicitly
if necessary. Back up that directory before updating or reingesting books.

## Provision models and outputs

Follow [translation setup](TRANSLATION_SETUP.md),
[object detection setup](OBJECT_DETECTION_SETUP.md), and
[gesture setup](GESTURE_CONTROL_SETUP.md). Install local speech-recognition
models for each advertised voice language. A language without a working model
must not be advertised as supporting private/offline voice input.

Local sentence-transformer embeddings load installed model assets only; runtime
will not download missing weights. Provision the configured model into the
SentenceTransformers cache while preparing the image. If cloud embeddings are
selected, a failed request does not silently change model spaces. Reingest saved
source documents into a separate local index before promising offline retrieval.

Configure two distinct sound cards using `aplay -l`. Private playback requires a
verified named ALSA output different from the speaker. It cannot fall back to
the default device. Private output on macOS is intentionally unavailable until
a platform-specific route can be verified; use explicit public mode for desktop
development. Test disconnecting each card before and during playback on the Pi.

Privacy is selected before content capture. A private session prohibits cloud
OCR/STT/AI/translation/TTS and persistent indexing. Temporary private recordings
are removed after transcription or session cleanup. The operating system's disk
and temporary-storage encryption/retention are separate device-image concerns.
An unanswered privacy prompt cancels content playback. Explanation and replay
inherit the selected privacy policy. Private AI requires a provisioned local
model; otherwise the app reports unavailable without sending the question online.

## Validate and start

```bash
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python selftest.py --profile device --vision --mic
```

The hardware command accesses the microphone and camera. Run it during device
setup with test material. Configure `enabled_modes` and `supported_languages`
to match the exact advertised product. Set `release_profile` to `device` only
after provisioning; the app then requires the device startup preflight to pass.
Development preflight remains available for laptops and labels missing features.

Run the OCR and translation benchmarks using their documented frozen datasets:

- [OCR gate](OCR_VALIDATION.md): representative camera images with exact human
  transcription, ordered numeric fidelity and required numeric examples.
- [Translation gate](TRANSLATION_VALIDATION.md): each released direction has
  sufficient reviewed examples; reviewer scores must identify the exact output.

Record p95 latency on the Pi and test camera/mic/button unplugging, cancellation,
lost network, repeated mode switches, private explanations, disk-full conditions,
and interrupted index publication. Complete task-based trials with blind users.
Do not label these checks complete from laptop unit tests.

Start with `.venv/bin/python main.py`. For unattended boot, adapt
`deploy/blindassist.service.example` to the installed path and service user. The
user needs access to serial, video and audio devices. Avoid running as root.

## Storage migration and recovery

New vector stores are namespaced by provider/model/dimension. Each snapshot
contains text, tags and vectors together. Writes are serialized across processes
and published atomically; a previous valid snapshot is retained. Source documents
are archived independently under `runtime/vector_store/sources/` so a model change
does not delete the library. Failed or incomplete embedding batches are rejected.
Repeated ingestion of identical source/tag combinations is idempotent.

Old USB stores are left untouched. Reingest original books or saved source JSON
into the new store using `scripts/ingest_textbooks.py`; do not mix model spaces.
Legacy pickle metadata is never loaded. A corrupt or incompatible index produces
an error instead of silently replacing the original files with an empty library.

## Update and rollback

Prepare an update in a separate checkout/virtual environment with a copy of the
device configuration and data. Run the checks above before changing the running
service path. Keep the previous application, lock, models, configuration and data
backup together. Stop the service before switching releases; restart and check
the spoken startup result. Roll back the service path and matching data backup
if validation fails. Do not downgrade an index format without restoring its
matching backup. The installer does not modify system services automatically.

Keep a release record containing code revision, OS/Python versions, dependency
lock, model hashes, applicable licenses/attribution, configuration without secrets,
accuracy reports, hardware results and unresolved limitations. `pip check` must
pass on the released image. An existing development environment with unrelated
package conflicts is not a validated deployment image.
