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

logger = logging.getLogger("ConfigLoader")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.json")


def load_settings() -> dict:
    """Return settings.json as a dict, or {} if it cannot be read.

    Never raises: callers use .get() with their own defaults.
    """
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.error(f"{CONFIG_PATH} is not a JSON object; ignoring it.")
            return {}
        return data
    except FileNotFoundError:
        logger.warning(f"No settings file at {CONFIG_PATH}; using built-in defaults.")
    except json.JSONDecodeError as e:
        # Worth shouting about: the device will run with defaults and the user
        # will wonder why none of their configuration applies.
        logger.error(f"settings.json is not valid JSON ({e}); using built-in defaults.")
    except OSError as e:
        logger.error(f"Could not read settings.json ({e}); using built-in defaults.")
    return {}
