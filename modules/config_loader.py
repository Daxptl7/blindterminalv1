"""AET -- shared settings loader used by every module.

This used to be a bare open()/json.load() with no error handling. Any module
importing it (confidential_mode, and anything added later) failed to import
outright if config/settings.json was missing, unreadable, or contained a
trailing comma — and main.py's _safe_import then reported the *feature* as
unavailable rather than the config as broken. A missing settings file should
degrade to defaults, not silently remove Confidential Mode.
"""
import json
import logging
import os
import math
from pathlib import Path

logger = logging.getLogger("ConfigLoader")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.json")


class SettingsError(ValueError):
    pass


def validate_settings(data):
    if not isinstance(data, dict):
        raise SettingsError("settings.json must contain a JSON object")
    if data.get("app_mode", "device") not in ("device", "simulation"):
        raise SettingsError("app_mode must be device or simulation")
    if "privacy" in data and not isinstance(data["privacy"], dict):
        raise SettingsError("privacy must be an object")
    for key, value in data.get("privacy", {}).items():
        if key.endswith("_seconds") and (isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value) or not 1 <= value <= 120):
            raise SettingsError(f"privacy.{key} must be between 1 and 120 seconds")
    if data.get("release_profile", "development") not in ("development", "device"):
        raise SettingsError("release_profile must be development or device")
    if "data_directory" in data and (not isinstance(data["data_directory"], str) or not data["data_directory"].strip()):
        raise SettingsError("data_directory must be a non-empty path")
    for key, value in data.items():
        if key.endswith(("_seconds", "_timeout_s", "_interval_s", "_budget_s", "_ms")):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise SettingsError(f"{key} must be a positive finite number")
        if key.startswith(("use_", "allow_")) or key.endswith(("_enabled", "_preload", "_first")):
            if not isinstance(value, bool):
                raise SettingsError(f"{key} must be true or false")
    for key in ("tts_rate", "voice_keep_recordings", "voice_stop_press_count"):
        if key in data and (isinstance(data[key], bool) or not isinstance(data[key], int)):
            raise SettingsError(f"{key} must be an integer")
    for key in ("live_echo_guard", "live_show_transcripts"):
        if key in data and not isinstance(data[key], bool):
            raise SettingsError(f"{key} must be true or false")
    for key in ("live_input_device", "live_output_device"):
        value = data.get(key)
        if value is not None and (isinstance(value, bool)
                or not isinstance(value, (str, int)) or isinstance(value, int) and value < 0):
            raise SettingsError(f"{key} must be a device name, non-negative index, or null")
    for key, low, high in (("live_mic_gain", 0.1, 4),
                           ("live_vad_silence_ms", 250, 2000),
                           ("live_echo_tail_ms", 1, 2000)):
        if key in data and (isinstance(data[key], bool)
                or not isinstance(data[key], (int, float))
                or not math.isfinite(data[key]) or not low <= data[key] <= high):
            raise SettingsError(f"{key} must be between {low} and {high}")
    if "live_max_reconnects" in data:
        value = data["live_max_reconnects"]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 5:
            raise SettingsError("live_max_reconnects must be an integer between 0 and 5")
    if "live_translation_glossary" in data:
        glossary = data["live_translation_glossary"]
        if (not isinstance(glossary, dict) or len(glossary) > 100
                or len(json.dumps(glossary, ensure_ascii=False)) > 4000):
            raise SettingsError("live_translation_glossary must be an object of at most 100 terms and 4000 characters")
    if "enabled_modes" in data and (not isinstance(data["enabled_modes"], list)
            or any(value not in list("123456789") for value in data["enabled_modes"])):
        raise SettingsError("enabled_modes must be a list of mode digits")
    return data


def data_directory():
    """Persistent state never depends on the legacy removable-drive symlink."""
    base = Path(__file__).resolve().parent.parent
    path = Path(load_settings().get("data_directory", "runtime")).expanduser()
    if not path.is_absolute():
        path = base / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_settings(strict=False) -> dict:
    """Return settings.json as a dict, or {} if it cannot be read.

    Missing optional configuration returns defaults. Invalid supplied settings
    raise SettingsError so callers cannot silently activate an unintended mode.
    """
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return validate_settings(data)
    except FileNotFoundError:
        if strict:
            raise SettingsError("Copy config/settings.example.json to config/settings.json before starting.")
        logger.warning(f"No settings file at {CONFIG_PATH}; using built-in defaults.")
    except json.JSONDecodeError as e:
        if strict:
            raise SettingsError(f"settings.json is invalid JSON at line {e.lineno}") from e
        # Worth shouting about: the device will run with defaults and the user
        # will wonder why none of their configuration applies.
        logger.error(f"settings.json is not valid JSON ({e}); using built-in defaults.")
    except OSError as e:
        if strict:
            raise SettingsError("Could not read settings.json") from e
        logger.error(f"Could not read settings.json ({e}); using built-in defaults.")
    return {}
