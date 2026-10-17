#!/usr/bin/env bash
# Install BlindAssist into this checkout without changing global Python packages.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-desktop}"
if [[ "$TARGET" != "desktop" && "$TARGET" != "pi" && "$TARGET" != "--check" ]]; then
  echo "Usage: ./install.sh [desktop|pi|--check]" >&2
  exit 2
fi
PYTHON_BIN="${BLINDASSIST_PYTHON:-python3}"
"$PYTHON_BIN" -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,14), "Use Python 3.10–3.13; the Pi reference image uses 3.11."'
if [[ "$TARGET" == "--check" ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/scripts/check_install_layout.py"
  exit 0
fi
if [[ "$TARGET" == "pi" ]]; then
  command -v apt-get >/dev/null || { echo "The Pi profile requires Raspberry Pi OS." >&2; exit 1; }
  for binary in arecord aplay espeak-ng tesseract rpicam-still; do
    if ! command -v "$binary" >/dev/null; then
      echo "Missing $binary. Install system packages from docs/DEPLOYMENT.md first." >&2
      exit 1
    fi
  done
fi
"$PYTHON_BIN" -m venv "$ROOT_DIR/.venv"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip
if [[ -f "$ROOT_DIR/deployment.lock.txt" ]]; then
  "$VENV_PYTHON" -m pip install -r "$ROOT_DIR/deployment.lock.txt"
else
  "$VENV_PYTHON" -m pip install -r "$ROOT_DIR/requirements.txt"
  if [[ "$TARGET" == "pi" ]]; then
    "$VENV_PYTHON" -m pip install -r "$ROOT_DIR/requirements-pi.txt"
  fi
fi
"$VENV_PYTHON" -m pip check
if [[ ! -f "$ROOT_DIR/config/settings.json" ]]; then
  cp "$ROOT_DIR/config/settings.example.json" "$ROOT_DIR/config/settings.json"
  chmod 600 "$ROOT_DIR/config/settings.json"
fi
mkdir -p "$ROOT_DIR/runtime" "$ROOT_DIR/logs"
"$VENV_PYTHON" -m pip freeze > "$ROOT_DIR/deployment.lock.txt"
echo "BlindAssist dependencies installed. Provision models using docs/DEPLOYMENT.md."
"$VENV_PYTHON" "$ROOT_DIR/selftest.py"
echo "Start with: $VENV_PYTHON $ROOT_DIR/main.py"
