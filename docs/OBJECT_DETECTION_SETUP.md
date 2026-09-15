# BlindAssist Object Detection

Mode 6 is an object **announcer**. It identifies known COCO objects and reports
their horizontal position. It is not a replacement for a cane, guide dog, depth
camera, or mobility training, and it must not claim that a route is safe.

## One-time laptop setup

Install and validate the small local model while online:

```bash
python3 scripts/prepare_object_detection.py
```

The command installs `models_local/yolo/yolov8n.pt`, records its SHA-256 digest,
and performs a real inference. `models_local/` is intentionally ignored by Git,
so production never accidentally commits a large model or depends on a broken
USB-drive symlink.

Run the non-camera checks:

```bash
python3 -m unittest discover -s tests -p 'test_object_detection_product.py' -v
```

Run the live model and camera preflight:

```bash
python3 selftest.py --vision
```

On macOS, Terminal (or the application launching Python) must be allowed under
System Settings → Privacy & Security → Camera.

## Raspberry Pi deployment

After pulling the code, either copy the laptop model directory to the Pi:

```text
models_local/yolo/yolov8n.pt
models_local/yolo/yolov8n.pt.sha256
```

or run the preparation script once on the Pi while it is online. For better Pi
CPU performance, create the NCNN export:

```bash
python3 scripts/prepare_object_detection.py --export-ncnn
python3 selftest.py --vision
```

On ARM devices, the resolver prefers `yolov8n_ncnn_model/` over the PyTorch
weights when both exist.

## Camera backends

`object_detection_camera_backend` accepts:

- `auto`: try OpenCV/USB first, then Camera Module 3 through `rpicam-vid`.
- `usb` or `opencv`: use only an OpenCV camera.
- `rpicam`: use only Camera Module 3.

The Camera Module 3 adapter uses an MJPEG stream from `rpicam-vid`; it does not
require importing Picamera2 into the virtual environment.

## Recommended starting configuration

```json
{
  "yolo_model_path": "models_local/yolo/yolov8n.pt",
  "yolo_allow_download": false,
  "yolo_confidence": 0.5,
  "yolo_imgsz": 416,
  "yolo_device": "cpu",
  "object_detection_camera_backend": "auto",
  "object_detection_camera_index": 0,
  "object_detection_rpicam_camera": 0,
  "object_detection_camera_fps": 15,
  "object_detection_interval_s": 0.4,
  "object_detection_stability_window": 5,
  "object_detection_stability_min_hits": 3,
  "object_detection_max_announced": 2,
  "object_detection_max_seconds": 120,
  "object_detection_display": false
}
```

The current device-specific `settings.json` can override these values. Tune the
confidence and input size only against saved camera footage; do not choose them
from a single demonstration.

## Behaviour

- The model loads before the stop listener starts.
- Old Pico W messages are drained before detection begins.
- Only Button 3 stops Mode 6; Buttons 1 and 2 are ignored.
- A detection must appear in at least three of five inference frames before it
  reaches speech.
- Speech contains at most two prioritized spatial groups, such as “a bus ahead,
  and a person on your left.”
- An unchanged scene is repeated at most once every five seconds.
- Model and camera failures produce distinct actionable messages.

## Product validation

Do not report one overall “accuracy.” For each important class, record:

- precision, recall and F1;
- false announcements per minute;
- missed important objects;
- median and p95 inference latency;
- camera startup success rate;
- performance in daylight, indoor light and low light.

Use recordings split by session/location. Adjacent frames from one video must
not be placed in both training/tuning and test sets, because that exaggerates
accuracy.
