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
import subprocess
import sys
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
    check("ALSA playback devices found", bool(available), ", ".join(available) or "none",
          "check `aplay -l`; is the USB sound card plugged in?")

    for role, configured in (("speaker_device", settings.get("speaker_device")),
                             ("bone_device", settings.get("bone_device"))):
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

    check("arecord available", diag["arecord"], "",
          "sudo apt install alsa-utils")
    check("Microphone hardware detected", bool(diag["capture_devices"]),
          ", ".join(diag["capture_devices"]) or "none",
          "check `arecord -l`; is the USB microphone plugged in?")

    active = diag["engines_active"]
    check("Speech recognition engine available", bool(active),
          " -> ".join(active) or "NONE",
          "pip install vosk  and download a model (see README)")

    check("Works without internet", diag["offline_capable"] or None,
          f"vosk={diag['vosk']} sphinx={diag['sphinx']}",
          "pip install vosk, then extract a model to models_local/vosk/",
          required=False)

    check("VAD (stops when you stop talking)", diag["vad"] and diag["pyaudio"] or None,
          f"webrtcvad={diag['vad']} pyaudio={diag['pyaudio']}",
          "pip install webrtcvad-wheels PyAudio", required=False)

    check("Recordings directory writable", bool(diag["recording_dir"]),
          diag["recording_dir"] or "none",
          "check that data/ is mounted, or set voice_recording_dir", required=False)

    if record_seconds:
        print(f"\n  Recording {record_seconds}s from the microphone — please speak…")
        pcm, error = voice._capture_arecord(record_seconds)
        if not pcm:
            hint = "try a different card: set mic_device in settings.json"
            if "hum" in (error or "").lower():
                # Every capture card returned stationary mains hum, i.e. no
                # capsule is connected to any of them (or nobody spoke).
                hint = "run `python3 mic_check.py` to find which card the microphone is on"
            check("Live microphone capture", False, error or "no audio", hint)
        else:
            level = voice._dbfs(voice._to_array(pcm))
            # Below about -45 dBFS is too quiet for any recogniser to work with.
            check("Live microphone capture", level > -45 or None,
                  f"{level:.0f} dBFS on {voice._capture_device_cache}",
                  "raise the capture level: alsamixer -c <card>, then F4 and raise Mic")
            text = voice._transcribe(pcm, "en-IN")
            check("Live transcription", bool(text), f"heard: {text!r}",
                  "speak louder/closer, or check the internet if only Google is active",
                  required=False)


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
    check_hardware(settings)
    return summarize(args.speak)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
