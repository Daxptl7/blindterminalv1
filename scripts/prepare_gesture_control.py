#!/usr/bin/env python3
"""Download and validate the official MediaPipe Hand Landmarker model."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = BASE_DIR / "models_local" / "mediapipe" / "hand_landmarker.task"
DEFAULT_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MIN_MODEL_BYTES = 1_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model(path: Path, load: bool = True) -> None:
    if not path.is_file():
        raise RuntimeError(f"model does not exist: {path}")
    if path.stat().st_size < MIN_MODEL_BYTES:
        raise RuntimeError(
            f"download is only {path.stat().st_size} bytes; expected a model bundle"
        )
    if not zipfile.is_zipfile(path):
        raise RuntimeError("download is not a valid MediaPipe .task model bundle")
    if not load:
        return

    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    from modules import gesture_control

    if not gesture_control.TASKS_AVAILABLE:
        raise RuntimeError(
            "MediaPipe Tasks is unavailable; install a compatible mediapipe package"
        )
    detector = gesture_control._create_task_detector(path)
    detector.close()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "BlindAssist-setup/1"})
    temporary = None
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            with tempfile.NamedTemporaryFile(
                prefix="hand_landmarker-", suffix=".task", dir=target.parent,
                delete=False,
            ) as output:
                temporary = Path(output.name)
                shutil.copyfileobj(response, output)
        validate_model(temporary)
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--force", action="store_true", help="download again")
    parser.add_argument(
        "--no-load-test", action="store_true",
        help="validate the bundle without loading it in MediaPipe",
    )
    args = parser.parse_args()
    target = args.target.expanduser().resolve()

    if target.exists() and not args.force:
        print(f"Using existing model: {target}")
    else:
        print(f"Downloading official MediaPipe model to {target}")
        download(args.url, target)

    validate_model(target, load=not args.no_load_test)
    checksum = sha256(target)
    checksum_path = target.with_suffix(target.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {target.name}\n", encoding="utf-8")
    print(f"Gesture model ready: {target}")
    print(f"Size: {target.stat().st_size / 1_000_000:.1f} MB")
    print(f"SHA-256: {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
