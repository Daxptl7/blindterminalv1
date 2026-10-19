"""PortAudio device negotiation and streaming PCM conversion for live speech.

Network input keeps the microphone's supported rate; Gemini resamples it.
Output is converted from Gemini's 24 kHz to the speaker's supported rate.
"""
import re
import sys
from dataclasses import dataclass

import numpy as np

try:
    import audioop
except ImportError:
    audioop = None


class LiveAudioError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioFormat:
    device: object
    name: str
    rate: int
    channels: int


def configured_device(settings, kind):
    """Keep index zero; inherit verified Pi routes only on Linux."""
    value = settings.get(f"live_{kind}_device")
    if value is None and sys.platform.startswith("linux"):
        value = settings.get("mic_device" if kind == "input" else "speaker_device")
    return None if value in (None, "", "default") else value


def resolve_device(sd, value, kind):
    if not isinstance(value, str):
        return value
    # ALSA routes used by the rest of BlindAssist refer to the same hardware
    # PortAudio enumerates as e.g. USB Audio Device (hw:3,0).
    match = re.fullmatch(r"(?:plug)?hw:([^,]+),(\d+)", value)
    if not match:
        return value
    card, dev = match.groups()
    devices = sd.query_devices()
    needle = f"(hw:{card},{dev})"
    for index, info in enumerate(devices):
        if needle in str(info.get("name", "")) and info.get(f"max_{kind}_channels", 0):
            return index
    raise LiveAudioError(
        f"The configured {kind} audio device was not found. "
        "Run scripts/check_live_translation.py --list-devices and select a live audio device."
    )


def negotiate(sd, value, kind, preferred_rate):
    try:
        device = resolve_device(sd, value, kind)
        info = sd.query_devices(device=device, kind=kind)
        max_channels = int(info[f"max_{kind}_channels"])
        rates = list(dict.fromkeys((preferred_rate, int(info["default_samplerate"]),
                                    48000, 44100, 16000, 24000)))
        check = sd.check_input_settings if kind == "input" else sd.check_output_settings
        for channels in (1, 2):
            if channels > max_channels:
                continue
            for rate in rates:
                try:
                    check(device=device, samplerate=rate, channels=channels, dtype="int16")
                    if kind == "output" and rate != 24000 and audioop is None:
                        raise LiveAudioError("Install audioop-lts for Python 3.13 speaker resampling.")
                    return AudioFormat(device, str(info["name"]), rate, channels)
                except LiveAudioError:
                    raise
                except Exception:
                    continue
    except LiveAudioError:
        raise
    except Exception:
        pass
    raise LiveAudioError(
        f"The {kind} audio device cannot open for live translation. "
        "Check the USB sound card and live audio device settings."
    )


class PlaybackConverter:
    """Preserve resampler state across packets; duplicate mono for stereo."""
    def __init__(self, output_format):
        self.format = output_format
        self.state = None

    def convert(self, pcm):
        if len(pcm) % 2:
            raise LiveAudioError("The translation service returned incomplete audio.")
        if self.format.rate != 24000:
            if audioop is None:
                raise LiveAudioError("Speaker resampling is unavailable. Install audioop-lts.")
            pcm, self.state = audioop.ratecv(pcm, 2, 1, 24000,
                                            self.format.rate, self.state)
        if self.format.channels == 2:
            pcm = np.repeat(np.frombuffer(pcm, dtype="<i2"), 2).astype("<i2").tobytes()
        return pcm


def microphone_pcm(data, channels, gain=1.0):
    """Downmix without integer overflow; saturate gain instead of wrapping."""
    samples = np.frombuffer(data, dtype="<i2")
    if len(samples) % channels:
        raise LiveAudioError("The microphone returned incomplete audio.")
    if channels > 1:
        samples = samples.astype(np.int32).reshape(-1, channels).mean(axis=1)
    if gain != 1.0:
        samples = samples.astype(np.float32) * gain
    return np.clip(samples, -32768, 32767).astype("<i2").tobytes()
