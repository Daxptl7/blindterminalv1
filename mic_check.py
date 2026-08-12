"""
mic_check.py — which USB sound card is the microphone actually on?
==================================================================
This device has two identical "USB Audio Device" dongles (a bone-conduction
unit and a speaker unit). Both advertise a capture channel, but only one of
them has a live microphone capsule on it — and ALSA's "default" is neither.
That is why voice input recorded 8 seconds of mains hum and then said "I didn't
catch that": the recorder was listening to the wrong card.

Run this, speak when told, and it reports the level each card heard. The card
that jumps when you speak is the microphone.

    python3 mic_check.py            # test every capture card, then offer to save
    python3 mic_check.py --list     # just list devices, no recording

The chosen card is written to config/settings.json as "mic_device", which both
voice.py capture backends (arecord and PyAudio) then use.
"""

import json
import math
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

RECORD_SECONDS = 5
SPEECH_MARGIN_DB = 8.0      # how far above its own noise floor a card must jump


def list_devices():
    print("ALSA capture cards (arecord -l):")
    try:
        print(subprocess.run(["arecord", "-l"], capture_output=True, text=True,
                             timeout=5).stdout.strip() or "  none found")
    except Exception as e:
        print(f"  could not run arecord: {e}")

    print("\nPortAudio input devices (PyAudio):")
    try:
        import pyaudio
    except Exception as e:
        print(f"  PyAudio unavailable: {e}")
        return
    pa = pyaudio.PyAudio()
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                print(f"  [{i}] {info['name']}  rate={info['defaultSampleRate']:.0f}")
        try:
            print(f"  default input: {pa.get_default_input_device_info()['name']}")
        except Exception:
            print("  default input: none")
    finally:
        pa.terminate()


def capture_cards():
    """['plughw:2,0', 'plughw:3,0'] — every real capture device, in order."""
    devices = []
    try:
        out = subprocess.run(["arecord", "-l"], capture_output=True, text=True,
                             timeout=5).stdout
    except Exception:
        return devices
    for line in out.splitlines():
        if not line.startswith("card "):
            continue
        try:
            card = line.split("card ")[1].split(":")[0].strip()
            dev = line.split("device ")[1].split(":")[0].strip()
            candidate = f"plughw:{card},{dev}"
            if candidate not in devices:
                devices.append(candidate)
        except (IndexError, ValueError):
            continue
    return devices


def _dbfs(samples):
    if samples.size == 0:
        return -120.0
    rms = math.sqrt(float(np.mean(np.square(samples, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-9) / 32768.0)


def record(device, seconds=RECORD_SECONDS):
    """Record `seconds` from `device` and return the samples (or None)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    cmd = ["arecord", "-D", device, "-f", "S16_LE", "-r", "16000", "-c", "1",
           "-d", str(seconds), "-t", "wav", path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=seconds + 10)
        if result.returncode != 0:
            detail = (result.stderr or "").strip().splitlines()
            print(f"    ✗ could not open {device}: {detail[-1] if detail else 'unknown error'}")
            return None
        with wave.open(path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
        # Skip the click a USB capture device makes when it is opened.
        return samples[4000:] if samples.size > 8000 else samples
    except Exception as e:
        print(f"    ✗ {device} failed: {e}")
        return None
    finally:
        Path(path).unlink(missing_ok=True)


def test_device(device):
    """Measure the quiet floor, then the level while the user speaks."""
    print(f"\n── {device} ──")
    print("    Stay quiet…")
    quiet = record(device, 3)
    if quiet is None:
        return None
    floor = _dbfs(quiet)
    print(f"    noise floor: {floor:.0f} dBFS")

    input(f"    Press ENTER, then SPEAK for {RECORD_SECONDS} seconds > ")
    loud = record(device, RECORD_SECONDS)
    if loud is None:
        return None
    level = _dbfs(loud)
    peak = int(np.max(np.abs(loud))) if loud.size else 0
    jump = level - floor
    print(f"    while speaking: {level:.0f} dBFS (peak {peak}), "
          f"{jump:+.0f} dB above its own floor")

    if jump >= SPEECH_MARGIN_DB:
        print("    ✓ this card HEARS YOU")
    else:
        print("    ✗ no reaction — nothing is plugged into this card's mic input")
    return {"device": device, "floor": floor, "level": level,
            "peak": peak, "jump": jump, "hears": jump >= SPEECH_MARGIN_DB}


def save_choice(device):
    try:
        settings = json.loads(CONFIG_PATH.read_text())
    except Exception as e:
        print(f"Could not read {CONFIG_PATH}: {e}")
        return
    settings["mic_device"] = device
    try:
        CONFIG_PATH.write_text(json.dumps(settings, indent=2) + "\n")
        print(f"Saved mic_device = {device} to {CONFIG_PATH}")
    except Exception as e:
        print(f"Could not write {CONFIG_PATH}: {e}")


def main():
    if "--list" in sys.argv:
        list_devices()
        return

    list_devices()
    devices = capture_cards()
    if not devices:
        print("\nNo capture cards found at all. Check the USB microphone is plugged in.")
        return

    print(f"\nTesting {len(devices)} capture card(s). "
          "Speak normally, about a hand's width from the microphone.")
    results = [r for r in (test_device(d) for d in devices) if r]

    print("\n── RESULT ──")
    for r in results:
        mark = "✓" if r["hears"] else "✗"
        print(f"  {mark} {r['device']:14} floor {r['floor']:>5.0f} dBFS  "
              f"speech {r['level']:>5.0f} dBFS  ({r['jump']:+.0f} dB)")

    hearing = [r for r in results if r["hears"]]
    if not hearing:
        print("\nNo card reacted to your voice. Things to check, in order:")
        print("  1. Is the microphone plugged into the USB dongle's PINK socket?")
        print("  2. Raise the capture level:  amixer -c <card> sset Mic 100%")
        print("  3. Unmute capture:           amixer -c <card> sset Mic cap")
        print("  4. Try the headset on the other dongle.")
        return

    best = max(hearing, key=lambda r: r["jump"])
    print(f"\nMicrophone is on {best['device']}.")
    if best["level"] < -30:
        print(f"  It is quiet ({best['level']:.0f} dBFS). Turn its capture up with:")
        card = best["device"].split(":")[1].split(",")[0]
        print(f"    amixer -c {card} sset Mic 100%")
    answer = input(f"Save mic_device = {best['device']} to settings.json? [Y/n] ").strip().lower()
    if answer in ("", "y", "yes"):
        save_choice(best["device"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
