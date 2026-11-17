"""Conservative filtering for generated speech, before output routing.

This cannot remove noise introduced later by a DAC, amplifier or power supply.
"""
import io
import wave


def filter_speech_wav(data: bytes) -> bytes:
    """Filter 16-bit PCM WAV without changing rate, channels or duration.

    Keep 80 Hz–8 kHz for speech, adapting the upper cutoff at lower sample
    rates. Each channel is filtered independently. Five-millisecond fades
    soften discontinuities when a player starts and stops an utterance.
    """
    import numpy as np
    from scipy.signal import butter, sosfilt

    with wave.open(io.BytesIO(data), "rb") as source:
        params = source.getparams()
        if source.getsampwidth() != 2 or source.getcomptype() != "NONE":
            raise ValueError("Speech filtering requires uncompressed 16-bit PCM WAV")
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    samples = samples.reshape(-1, params.nchannels).astype(np.float64)
    if not samples.size:
        return data
    upper = min(8000.0, params.framerate * 0.45)
    if upper <= 80:
        raise ValueError("Speech sample rate is too low for the filter")
    sections = butter(4, (80.0, upper), btype="bandpass",
                      fs=params.framerate, output="sos")
    samples = sosfilt(sections, samples, axis=0)
    fade = min(round(params.framerate * 0.005), len(samples) // 2)
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade)[:, None]
        samples[:fade] *= ramp
        samples[-fade:] *= ramp[::-1]
    # Filtering can overshoot even when its input is within PCM bounds.
    samples = np.clip(np.rint(samples), -32768, 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setparams(params)
        target.writeframes(samples.tobytes())
    return output.getvalue()
