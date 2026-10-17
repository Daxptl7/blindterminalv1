#!/usr/bin/env bash
# Launch from any working directory after pulling the repository on the Pi.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${BLINDASSIST_PYTHON:-}"
if [[ -z "$PYTHON_BIN" && -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  PYTHON_BIN="$VIRTUAL_ENV/bin/python"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/aet-env/bin/python"; do
    if [[ -x "$candidate" ]]; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "No project virtual environment found. Run ./install.sh pi first." >&2
  exit 1
fi
if [[ ! -f "$ROOT_DIR/config/settings.json" ]]; then
  echo "Create config/settings.json from config/settings.example.json and configure your Pi devices first." >&2
  exit 1
fi
cd "$ROOT_DIR"
exec "$PYTHON_BIN" "$ROOT_DIR/main.py" "$@"
