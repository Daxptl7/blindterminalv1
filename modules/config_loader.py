"""AET -- shared settings loader used by every module."""
import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.json")

def load_settings():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
