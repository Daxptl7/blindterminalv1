"""Signal-level checks for speech filtering; no speaker or cloud calls."""
import io
import unittest
import wave
from unittest import mock

import numpy as np

from modules import tts
from modules.audio_filters import filter_speech_wav
from modules.config_loader import SettingsError, validate_settings


def make_wav(samples, rate=24000, channels=1):
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        target.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return output.getvalue()


def read_wav(data):
    with wave.open(io.BytesIO(data), "rb") as source:
        params = source.getparams()
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return params, samples.reshape(-1, params.nchannels)


class SpeechFilterTests(unittest.TestCase):
    def test_reduces_rumble_and_high_hiss_preserving_speech_band(self):
        rate = 24000
        time = np.arange(rate) / rate
        tones = (30, 1000, 11000)
        original = sum(5000 * np.sin(2 * np.pi * freq * time) for freq in tones)
        _, samples = read_wav(filter_speech_wav(make_wav(original)))
        # Ignore filter startup and endpoint fades when measuring tone energy.
        region = slice(2400, -2400)
        for freq in tones:
            amplitude = abs(np.mean(samples[region, 0] *
                                   np.exp(-2j * np.pi * freq * time[region]))) * 2
            if freq == 1000:
                self.assertGreater(amplitude, 4500)
            else:
                self.assertLess(amplitude, 500)

    def test_stereo_shape_duration_and_silent_channel_are_preserved(self):
        time = np.arange(16000) / 16000
        left = 6000 * np.sin(2 * np.pi * 1000 * time)
        source = make_wav(np.column_stack((left, np.zeros_like(left))), 16000, 2)
        before, _ = read_wav(source)
        after, samples = read_wav(filter_speech_wav(source))
        self.assertEqual(before, after)
        self.assertFalse(np.any(samples[:, 1]))

    def test_fades_endpoints_without_pcm_wraparound(self):
        source = np.tile([32767, 32767, -32768, -32768], 600)
        _, samples = read_wav(filter_speech_wav(make_wav(source)))
        self.assertEqual(int(samples[0, 0]), 0)
        self.assertEqual(int(samples[-1, 0]), 0)
        self.assertLessEqual(np.max(np.abs(samples.astype(np.int32))), 32768)

    def test_transient_filtering_does_not_raise_input_peak(self):
        time = np.arange(24000) / 24000
        source = 6000 * np.sign(np.sin(2 * np.pi * 1000 * time))
        _, samples = read_wav(filter_speech_wav(make_wav(source)))
        peak = np.max(np.abs(samples.astype(np.int32)))
        self.assertLessEqual(peak, 6000)
        self.assertGreater(peak, 5000)

    def test_silence_empty_and_short_utterances(self):
        for length in (0, 1, 2, 10, 2400):
            with self.subTest(length=length):
                _, samples = read_wav(filter_speech_wav(make_wav(np.zeros(length))))
                self.assertEqual(len(samples), length)
                self.assertFalse(np.any(samples))


class SpeechPreparationTests(unittest.TestCase):
    def setUp(self):
        self.manager = tts.TTSManager.__new__(tts.TTSManager)
        self.manager.settings = {"tts_noise_filter_enabled": True}
        self.wav = make_wav(np.full(2400, 1000))

    def test_enabled_filter_runs_even_when_gain_and_tempo_are_unmodified(self):
        data, fmt = self.manager._prepare_audio(self.wav, "wav")
        self.assertEqual(fmt, "wav")
        self.assertNotEqual(data, self.wav)

    def test_disabled_filter_is_byte_exact(self):
        self.manager.settings["tts_noise_filter_enabled"] = False
        self.assertEqual(self.manager._prepare_audio(self.wav, "wav"), (self.wav, "wav"))

    def test_mp3_is_decoded_before_filtering(self):
        with mock.patch.object(tts, "_decode_mp3_to_wav", return_value=self.wav) as decode:
            data, fmt = self.manager._prepare_audio(b"mp3", "mp3")
        decode.assert_called_once_with(b"mp3")
        self.assertEqual(fmt, "wav")
        self.assertNotEqual(data, self.wav)

    def test_filtering_mp3_preserves_configured_playback_speed(self):
        self.manager.settings["tts_playback_speed"] = 1.2
        result = mock.Mock(returncode=0, stdout=self.wav)
        with mock.patch.object(tts, "_decode_mp3_to_wav", return_value=self.wav), \
                mock.patch.object(tts.shutil, "which", return_value="ffmpeg"), \
                mock.patch.object(tts.subprocess, "run", return_value=result) as run:
            self.manager._prepare_audio(b"mp3", "mp3")
        command = run.call_args.args[0]
        self.assertIn("atempo=1.2", command[command.index("-af") + 1])
        self.assertNotEqual(run.call_args.kwargs["input"], self.wav)

    def test_filter_failure_keeps_original_audio_available(self):
        with mock.patch("modules.audio_filters.filter_speech_wav", side_effect=ImportError), \
                self.assertLogs("TTSModule", level="WARNING"):
            self.assertEqual(self.manager._prepare_audio(self.wav, "wav"), (self.wav, "wav"))

    def test_decode_failure_keeps_original_mp3_available(self):
        with mock.patch.object(tts, "_decode_mp3_to_wav", return_value=b""), \
                self.assertLogs("TTSModule", level="WARNING"):
            self.assertEqual(self.manager._prepare_audio(b"mp3", "mp3"), (b"mp3", "mp3"))

    def test_filter_setting_requires_a_boolean(self):
        with self.assertRaises(SettingsError):
            validate_settings({"tts_noise_filter_enabled": "true"})


if __name__ == "__main__":
    unittest.main()
