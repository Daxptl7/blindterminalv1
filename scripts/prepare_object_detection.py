#!/usr/bin/env python3
"""Provision and validate the local object-detection model.

Run this once while the development laptop (or Pi) has internet access:

    python3 scripts/prepare_object_detection.py

For the optimized Raspberry Pi format:

    python3 scripts/prepare_object_detection.py --export-ncnn
"""

import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "models_local" / "yolo"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(model_path: Path, image_size: int):
    import numpy as np
    from ultralytics import YOLO

    print(f"Loading {model_path} ...")
    model = YOLO(str(model_path))
    sample = np.zeros((480, 640, 3), dtype=np.uint8)
    results = model.predict(
        source=sample,
        imgsz=image_size,
        conf=0.5,
        device="cpu",
        verbose=False,
    )
    if not results:
        raise RuntimeError("The model loaded but returned no inference result.")
    print("Model validation inference passed.")
    return model


def download_model(model_name: str, target: Path) -> Path:
    from ultralytics import YOLO

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="blindassist-yolo-") as temp_dir:
        previous = Path.cwd()
        try:
            os.chdir(temp_dir)
            model = YOLO(model_name)
            downloaded = Path(getattr(model, "ckpt_path", model_name))
            if not downloaded.is_absolute():
                downloaded = Path(temp_dir) / downloaded
            if not downloaded.exists():
                matches = list(Path(temp_dir).rglob(Path(model_name).name))
                if not matches:
                    raise RuntimeError("Ultralytics did not produce a model file.")
                downloaded = matches[0]
            shutil.copy2(downloaded, target)
        finally:
            os.chdir(previous)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install and validate BlindAssist object-detection weights"
    )
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Ultralytics model name or an existing .pt path")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--force", action="store_true",
                        help="replace an existing local model")
    parser.add_argument("--export-ncnn", action="store_true",
                        help="also export the model to NCNN for Raspberry Pi")
    args = parser.parse_args()

    source = Path(args.model).expanduser()
    target = args.target_dir.expanduser().resolve() / source.name
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target:
            if args.force or not target.exists():
                shutil.copy2(source, target)
    elif args.force or not target.exists():
        print(f"Downloading {args.model} into {target.parent} ...")
        download_model(args.model, target)

    if not target.exists():
        raise RuntimeError(f"Model was not installed at {target}")

    digest = sha256(target)
    target.with_suffix(target.suffix + ".sha256").write_text(
        f"{digest}  {target.name}\n", encoding="utf-8"
    )
    print(f"SHA-256: {digest}")
    model = validate(target, args.imgsz)

    if args.export_ncnn:
        print("Exporting NCNN model for Raspberry Pi ...")
        exported = Path(model.export(format="ncnn", imgsz=args.imgsz))
        validate(exported, args.imgsz)
        print(f"NCNN model ready: {exported}")

    print(f"Object detection model ready: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
