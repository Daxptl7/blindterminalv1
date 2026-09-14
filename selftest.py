#!/usr/bin/env python3
"""
selftest.py — BlindAssist pre-flight check
==========================================
Run this after any install, reflash, or hardware change:

    python3 selftest.py            # report only
    python3 selftest.py --speak    # also read the summary aloud
    python3 selftest.py --mic      # additionally record 3s and report the level

Why this exists: almost every failure on this device is silent. A microphone
plugged into the wrong USB sound card, an unmounted model stick, a missing
speech engine and an expired API key all present identically to a blind user —
the device simply does not answer. This prints, in one place, what actually
works right now and what to do about anything that does not.

Exit code is 0 when every REQUIRED check passes, 1 otherwise, so it can gate a
systemd unit or a deployment script.
"""

import argparse
import json
import shutil
import sys
import wave
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = YELLOW = DIM = RESET = ""

_results = []


def check(name, ok, detail="", fix="", required=True):
    """Record one check. `ok` may be True, False, or None for 'degraded'."""
    _results.append({"name": name, "ok": ok, "detail": detail,
                     "fix": fix, "required": required})
    mark = f"{GREEN}PASS{RESET}" if ok else (
        f"{RED}FAIL{RESET}" if ok is False else f"{YELLOW}WARN{RESET}")
    print(f"  [{mark}] {name}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
    if not ok and fix:
        print(f"         {YELLOW}fix:{RESET} {fix}")


def section(title):
    print(f"\n{title}\n" + "─" * 60)


# ── CONFIG ──────────────────────────────────────────────────
def check_config():
    section("Configuration")
    path = BASE_DIR / "config" / "settings.json"
    if not path.exists():
        check("settings.json present", False, str(path),
              "copy config/settings.example.json to config/settings.json")
        return {}
    try:
        settings = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        check("settings.json is valid JSON", False, str(e),
              "fix the syntax error; the device is running on defaults until you do")
        return {}
    check("settings.json is valid JSON", True, f"{len(settings)} keys")

    keys = {k: settings.get(k) for k in
            ("groq_api_key", "openai_api_key", "gemini_api_key")}
    have = [k.replace("_api_key", "") for k, v in keys.items() if v]
    check("At least one AI API key set", bool(have), ", ".join(have) or "none",
          "add groq_api_key (free tier, fastest) to config/settings.json")
    return settings


# ── AUDIO OUT ───────────────────────────────────────────────
def check_audio_output(settings):
    section("Audio output (speech)")
    try:
        from modules import tts
    except Exception as e:
        check("tts module imports", False, str(e), required=True)
        return
    check("tts module imports", True)

    available = tts._AVAILABLE_OUTPUTS
    on_pi = Path("/proc/device-tree/model").exists()
    check("ALSA playback devices found", bool(available) or None,
          ", ".join(available) or "none (normal on a laptop without ALSA)",
          "check `aplay -l`; is the USB sound card plugged in?", required=on_pi)

    for role, configured in (("speaker_device", settings.get("speaker_device")),
                             ("earphone_device",
                              settings.get("earphone_device")
                              or settings.get("bone_device"))):
        if not configured:
            continue
        ok = (not available) or configured in available
        check(f"{role} exists ({configured})", ok or None,
              "" if ok else f"not present; using {tts.SPEAKER_DEVICE}",
              f"set {role} to one of: {', '.join(available)}",
              required=False)

    engine = settings.get("tts_engine", "pyttsx3")
    if engine == "gtts":
        can_route = bool(shutil.which("mpg123") or shutil.which("mpg321"))
        try:
            import soundfile  # noqa: F401
            can_route = True
        except Exception:
            pass
        check("gTTS audio can be routed to a chosen card", can_route or None,
              "needed for Confidential Mode's private/speaker split",
              "sudo apt install mpg123   (or pip install soundfile)",
              required=False)


# ── AUDIO IN / SPEECH RECOGNITION ───────────────────────────
def check_voice(record_seconds=0):
    section("Voice input (speech recognition)")
    try:
        from modules import voice
    except Exception as e:
        check("voice module imports", False, str(e))
        return
    check("voice module imports", True)

    diag = voice.diagnostics()

    on_pi = Path("/proc/device-tree/model").exists()
    check("arecord capture program", diag["arecord"] or None,
          "required on Raspberry Pi" if not diag["arecord"] else "available",
          "sudo apt install alsa-utils", required=on_pi)
    check("Microphone route configured", bool(diag["mic_device"]),
          str(diag["mic_device"] or "none"),
          "set mic_device in config/settings.json")

    active_by_language = diag["engines_active"]
    for code, name in (("en", "English"), ("hi", "Hindi"), ("gu", "Gujarati")):
        active = active_by_language.get(code, [])
        check(f"{name} speech recognition", bool(active),
              " -> ".join(active) or "NONE",
              f"install a {name} Vosk model at models_local/vosk/{code}")
        # PocketSphinx is only an emergency English fallback and is not close
        # enough to the product accuracy target to count as offline-ready.
        offline = "vosk" in active
        check(f"{name} works without internet", offline or None,
              diag.get("vosk_models", {}).get(code, "no language model"),
              f"install a {name} Vosk model at models_local/vosk/{code}",
              required=False)

    check("Recordings directory writable", bool(diag["recording_dir"]),
          diag["recording_dir"] or "none",
          "check that data/ is mounted, or set voice_recording_dir", required=False)

    if record_seconds:
        print(f"\n  Recording up to {record_seconds}s from the microphone — please speak…")
        path = voice.record(max_seconds=record_seconds, speak_fn=print)
        if not path:
            check("Live microphone capture", False, voice.get_last_error() or "no audio",
                  "check arecord -l and set mic_device in settings.json",
                  required=on_pi)
        else:
            try:
                with wave.open(path, "rb") as recorded:
                    pcm = recorded.readframes(recorded.getnframes())
                level = voice._dbfs(voice._to_array(pcm))
                check("Live microphone capture", level > -45 or None,
                      f"{level:.0f} dBFS on {diag['mic_device']}",
                      "raise the capture level with alsamixer")
                text = voice.transcribe_file(path, "en-IN")
                check("Live transcription", bool(text), f"heard: {text!r}",
                      "speak closer or install/check the configured STT engine",
                      required=False)
            except Exception as e:
                check("Live microphone capture", False, str(e), required=on_pi)


# ── HARDWARE ────────────────────────────────────────────────
def check_hardware(settings):
    section("Hardware")
    pico = sorted(Path("/dev").glob("ttyACM*"))
    check("Pico W Morse buttons", bool(pico), str(pico[0]) if pico else "not connected",
          "plug in the Pico W; without it the menu needs a keyboard", required=False)

    cams = sorted(Path("/dev").glob("video*"))
    check("Camera device nodes", bool(cams), f"{len(cams)} nodes",
          "check the ribbon cable / USB webcam", required=False)
    check("rpicam-still (Camera Module 3)", bool(shutil.which("rpicam-still")), "",
          "sudo apt install rpicam-apps", required=False)
    check("tesseract OCR", bool(shutil.which("tesseract")), "",
          "sudo apt install tesseract-ocr", required=False)

    model = settings.get("yolo_model_path")
    try:
        from modules import object_detection as od
        resolved = od.MODEL_PATH
        exists = Path(resolved).exists()
        check("YOLO weights present", exists or None,
              f"{resolved}" + ("" if exists else " (will download on first use)"),
              f"yolo_model_path={model!r} does not exist", required=False)
    except Exception as e:
        check("Object detection module", None, str(e), required=False)


# ── LOCAL LANGUAGE TRANSLATION ─────────────────────────────
def check_translation():
    section("Translation (English, Hindi, Gujarati)")
    try:
        from modules import translator
    except Exception as e:
        check("translator module imports", False, str(e))
        return
    check("translator module imports", True)

    try:
        info = translator.diagnostics()
    except Exception as e:
        check("translation diagnostics", False, str(e))
        return

    pairs = info.get("supported_pairs", [])
    check("All six translation directions exposed", len(pairs) == 6,
          ", ".join(pairs), "restore en/hi/gu source and target selection")

    bundles = info.get("indictrans2", {}).get("bundles", {})
    for bundle, label in (("en-indic", "English to Hindi/Gujarati"),
                          ("indic-en", "Hindi/Gujarati to English"),
                          ("indic-indic", "Hindi to/from Gujarati")):
        state = bundles.get(bundle, {})
        check(f"Offline model: {label}", state.get("available") or None,
              state.get("configured", "not configured"),
              "install the local IndicTrans2 bundle; see docs/TRANSLATION_SETUP.md",
              required=False)

    if info.get("libre_url"):
        check("LibreTranslate privacy classification",
              info.get("libre_trusted_local") or None,
              str(info["libre_url"]),
              "set libretranslate_trusted_local=true only for a trusted local server",
              required=False)

    check("Versioned validated translation cache", info.get("cache_schema") == 2,
          f"schema={info.get('cache_schema')}, entries={info.get('cache_entries')}")


# ── SUMMARY ─────────────────────────────────────────────────
def summarize(speak: bool):
    section("Summary")
    failed = [r for r in _results if r["ok"] is False and r["required"]]
    degraded = [r for r in _results if r["ok"] is None or
                (r["ok"] is False and not r["required"])]

    for r in failed:
        print(f"  {RED}BROKEN{RESET}   {r['name']}")
    for r in degraded:
        print(f"  {YELLOW}DEGRADED{RESET} {r['name']}")

    if not failed and not degraded:
        message = "All checks passed. The device is ready."
    elif not failed:
        message = (f"The device will run, with {len(degraded)} feature "
                   f"{'limitation' if len(degraded) == 1 else 'limitations'}.")
    else:
        message = (f"{len(failed)} required "
                   f"{'check has' if len(failed) == 1 else 'checks have'} failed. "
                   "The device will not work correctly.")

    print(f"\n  {message}\n")

    if speak:
        try:
            from modules import tts
            tts.speak(message, block=True)
            tts.shutdown()
        except Exception as e:
            print(f"  (could not speak the summary: {e})")

    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description="BlindAssist pre-flight check")
    parser.add_argument("--speak", action="store_true", help="read the summary aloud")
    parser.add_argument("--mic", action="store_true",
                        help="record 3 seconds and report the level")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  BlindAssist — Pre-flight Check")
    print("=" * 60)

    settings = check_config()
    check_audio_output(settings)
    check_voice(record_seconds=3 if args.mic else 0)
    check_translation()
    check_hardware(settings)
    return summarize(args.speak)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
