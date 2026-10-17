"""Read-only validation of the installer/config layout; no downloads or capture."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.config_loader import validate_settings

for name in ("main.py", "selftest.py", "requirements.txt", "requirements-pi.txt",
             "docs/DEPLOYMENT.md", "config/settings.example.json"):
    if not (ROOT / name).is_file():
        raise SystemExit(f"Missing installation file: {name}")
validate_settings(json.loads((ROOT / "config/settings.example.json").read_text()))
if (ROOT / "config/settings.json.example").read_bytes() != (ROOT / "config/settings.example.json").read_bytes():
    raise SystemExit("Legacy configuration example differs from the canonical template.")
print("BlindAssist installation layout and example configuration are valid.")
