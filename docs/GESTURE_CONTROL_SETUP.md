# BlindAssist Gesture Control

Mode 5 is an optional shortcut for starting existing BlindAssist modes. It is
not a mobility or emergency control. Every action requires a physical Button 1
press while the user holds a pose; merely moving a hand cannot launch a mode.

## Gesture vocabulary

| Pose | Action |
| --- | --- |
| One finger | Object detection |
| Two fingers | Voice question |
| Three fingers | GPS |
| Four fingers/open palm | OCR scan |
| Closed fist | Leave gesture mode |

Finger state is calculated from the 3D PIP and DIP joint angles, so the same
pose works when the hand is rotated in the camera image. A thumbs-up is not
treated as a fist.

## One-time laptop setup

Install and runtime-test Google's official Hand Landmarker bundle:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/prepare_gesture_control.py
```

This creates these gitignored files:

```text
models_local/mediapipe/hand_landmarker.task
models_local/mediapipe/hand_landmarker.task.sha256
```

Run the deterministic tests:

```bash
python3 -m unittest discover -s tests -p 'test_gesture_product.py' -v
```

Run model and camera preflight:

```bash
python3 selftest.py --vision
```

macOS must grant Camera permission to Terminal, iTerm, or the application that
starts Python: System Settings → Privacy & Security → Camera. A model test can
pass while the live camera check fails if this permission is missing.

To benchmark one saved hand image through the actual MediaPipe model:

```bash
python3 scripts/benchmark_gesture.py --image /path/to/hand.jpg
```

To exercise the complete live camera and detector loop without launching a
mode:

```bash
python3 scripts/benchmark_gesture.py --camera --frames 100
```

## Accessible operation

- Hold the complete hand near the centre of the image, about one arm away.
- Button 2 speaks framing or the currently recognized action without starting
  it.
- Button 1 takes the gesture reading. The detector requires at least five
  agreeing frames and 60% support in the last second.
- Button 3 always leaves Mode 5 without requiring a visible gesture.
- Camera, model, MediaPipe, and frame-read failures produce distinct spoken
  messages instead of silently returning to the menu.

## Raspberry Pi deployment

After pulling the code, copy `models_local/mediapipe/` from the laptop to the
same project path on the Pi, or run the preparation script once while the Pi is
online. Then run:

```bash
source aet-env/bin/activate
python3 selftest.py --vision
python3 main.py
```

`gesture_camera_backend` supports `auto`, `usb`/`opencv`, and `rpicam`. `auto`
tries a real USB video node first, then Camera Module 3 through `rpicam-vid`.
The scan avoids Pi ISP and metadata nodes that can open but never return an
image.

## Recommended settings

```json
{
  "gesture_confidence": 0.60,
  "gesture_backend": "auto",
  "gesture_allow_unsafe_tasks": true,
  "gesture_hand_model_path": "models_local/mediapipe/hand_landmarker.task",
  "gesture_camera_backend": "auto",
  "gesture_camera_index": 0,
  "gesture_rpicam_camera": 0,
  "gesture_camera_fps": 15,
  "gesture_capture_window_ms": 1000,
  "gesture_capture_fresh_ms": 300,
  "gesture_capture_min_frames": 5,
  "gesture_capture_min_support": 0.60,
  "gesture_straight_pip_degrees": 150,
  "gesture_straight_dip_degrees": 145,
  "gesture_enable_swipes": false
}
```

## Accuracy gate

Do not claim 95% accuracy from synthetic landmarks or one demonstration.
Record at least 30 attempts per pose per participant, including `no gesture`,
with multiple users, left/right hands, rotations, skin tones, sleeves,
backgrounds, camera distances, daylight, indoor light, and low light. Split the
test data by participant/session rather than by adjacent video frames.

Measure:

- action precision and recall for each pose;
- false action rate, especially when Button 1 is pressed with no valid pose;
- successful framing rate after spoken Button 2 guidance;
- median and p95 time from Button 1 to action;
- camera/backend startup success across 20 cold starts.

Release only when every safety-relevant action meets its target, not when the
average hides a weak gesture. Retune the confidence, joint-angle thresholds,
and vote support against a training/validation set, then evaluate once on a
held-out test set.
