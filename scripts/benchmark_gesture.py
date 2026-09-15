#!/usr/bin/env python3
"""Run the real gesture landmark/classification pipeline on one image."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from modules import gesture_control as gesture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument(
        "--camera", action="store_true",
        help="exercise the complete live detector without triggering an action",
    )
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()

    if args.camera:
        events = []
        started = time.perf_counter()
        gesture.detect_gesture(
            callback_fn=events.append,
            capture_check=lambda: None,
            max_frames=max(1, args.frames),
            display=False,
        )
        elapsed = time.perf_counter() - started
        failure_events = {
            "MEDIAPIPE_UNAVAILABLE", "HAND_MODEL_MISSING",
            "BACKEND_UNAVAILABLE", "CAMERA_UNAVAILABLE", "FRAME_READ_FAILED",
        }
        report = {
            "source": "camera",
            "frames": max(1, args.frames),
            "elapsed_s": round(elapsed, 2),
            "effective_fps": round(max(1, args.frames) / elapsed, 2),
            "events": events,
            "ready": not any(event in failure_events for event in events),
        }
        print(json.dumps(report, indent=2))
        return 0 if report["ready"] else 1

    frame = cv2.imread(str(args.image.expanduser()))
    if frame is None:
        parser.error(f"could not read image: {args.image}")
    if not gesture.TASKS_AVAILABLE:
        parser.error("MediaPipe Tasks is unavailable")
    model_path = gesture._resolve_hand_model_path()
    if not model_path.is_file():
        parser.error("model missing; run scripts/prepare_gesture_control.py")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = gesture.mp.Image(
        image_format=gesture.mp.ImageFormat.SRGB,
        data=rgb,
    )
    detector = gesture._create_task_detector(model_path)
    timings = []
    detected_counts = []
    classifications = []
    guidance = []
    try:
        for timestamp_ms in range(1, max(1, args.frames) + 1):
            started = time.perf_counter()
            result = detector.detect_for_video(mp_image, timestamp_ms)
            timings.append((time.perf_counter() - started) * 1000)
            hands = result.hand_landmarks or []
            detected_counts.append(len(hands))
            if hands:
                guidance.append(gesture.framing_guidance(hands[0]))
                classifications.append(gesture.classify_gesture(hands[0]))
    finally:
        detector.close()

    warm = timings[1:] or timings
    recognized = [value for value in classifications if value]
    report = {
        "image": str(args.image.expanduser().resolve()),
        "image_size": [frame.shape[1], frame.shape[0]],
        "model": str(model_path),
        "frames": len(timings),
        "frames_with_hand": sum(count > 0 for count in detected_counts),
        "dominant_guidance": Counter(guidance).most_common(1)[0][0] if guidance else None,
        "dominant_classification": (
            Counter(recognized).most_common(1)[0][0] if recognized else None
        ),
        "cold_ms": round(timings[0], 2),
        "warm_median_ms": round(statistics.median(warm), 2),
        "warm_p95_ms": round(sorted(warm)[max(0, int(len(warm) * 0.95) - 1)], 2),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["frames_with_hand"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
