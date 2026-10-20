"""Speech remains audible when cloud and Python speech drivers fail."""
import io
import subprocess
import threading
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from modules import tts
from modules.privacy_policy import PRIVATE, processing_session


def speech_wav():
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        target.writeframes(b"\x00\x10" * 2400)
    return output.getvalue()


class OfflineSpeechTests(unittest.TestCase):
    def setUp(self):
        self.manager = tts.TTSManager.__new__(tts.TTSManager)
        self.manager.settings = {"tts_engine": "pyttsx3"}
        self.manager.running = True
        self.manager._speaking = threading.Event()
        self.manager._current_rate = 180
        self.manager._engine_available = False
        self.manager._engine = None
        self.manager._engine_lock = threading.Lock()
        self.manager._gtts_failed_at = 0
        self.wav = speech_wav()

    def test_pi_english_speech_uses_espeak_without_python_driver(self):
        with mock.patch.object(tts, "IS_MACOS", False), \
                mock.patch.object(tts, "_espeak_to_wav", return_value=self.wav) as synth:
            self.assertEqual(self.manager._synthesize("Speaker test", "eng"),
                             (self.wav, "wav"))
        synth.assert_called_once_with("Speaker test", 180, "eng")

    def test_cloud_failure_still_reaches_selected_pi_speaker(self):
        self.manager.settings["tts_engine"] = "gtts"
        cloud = mock.Mock(side_effect=ConnectionError("offline"))
        with mock.patch.object(tts, "IS_MACOS", False), \
                mock.patch.object(tts, "_espeak_to_wav", return_value=self.wav), \
                mock.patch.dict("sys.modules", {"gtts": SimpleNamespace(gTTS=cloud)}), \
                mock.patch.object(self.manager, "_play_routed", return_value=True) as routed, \
                mock.patch.object(self.manager, "_play_pyttsx3_direct") as direct:
            self.manager._deliver("Speaker test", "eng", "plughw:3,0")
        routed.assert_called_once_with(self.wav, "wav", "plughw:3,0")
        direct.assert_not_called()

    def test_private_offline_speech_keeps_earphone_route(self):
        with processing_session(PRIVATE), \
                mock.patch.object(tts, "IS_MACOS", False), \
                mock.patch.object(tts, "require_private_output", return_value="earphone"), \
                mock.patch.object(tts, "_espeak_to_wav", return_value=self.wav), \
                mock.patch.object(self.manager, "_play_routed", return_value=True) as routed, \
                mock.patch.object(self.manager, "_play_system") as public:
            self.assertTrue(self.manager._deliver("Private test", "eng", "earphone"))
        routed.assert_called_once_with(self.wav, "wav", "earphone")
        public.assert_not_called()

    def test_espeak_bypasses_python_driver_event_loop(self):
        self.manager._engine_available = True
        self.manager._engine = mock.Mock()
        with mock.patch.object(tts, "_espeak_to_wav", return_value=self.wav):
            self.assertEqual(self.manager._synthesize("Speaker test", "eng"),
                             (self.wav, "wav"))
        self.manager._engine.runAndWait.assert_not_called()

    def test_python_driver_remains_available_when_native_synthesis_fails(self):
        self.manager._engine_available = True
        self.manager._engine = mock.Mock()
        self.manager._engine.save_to_file.side_effect = lambda text, path: (
            Path(path).write_bytes(self.wav))
        with mock.patch.object(tts, "IS_MACOS", False), \
                mock.patch.object(tts, "_espeak_to_wav", return_value=b""):
            self.assertEqual(self.manager._synthesize("Speaker test", "eng"),
                             (self.wav, "wav"))
        self.manager._engine.runAndWait.assert_called_once()

    def test_blocking_queue_speaks_when_python_driver_cannot_initialize(self):
        driver = SimpleNamespace(init=mock.Mock(side_effect=RuntimeError("no driver")))
        with mock.patch.dict("sys.modules", {"pyttsx3": driver}), \
                mock.patch.object(tts, "IS_MACOS", False), \
                mock.patch.object(tts, "PYGAME_AVAILABLE", False), \
                mock.patch.object(tts, "_espeak_to_wav", return_value=self.wav), \
                mock.patch.object(tts.TTSManager, "_load_settings", return_value=self.manager.settings), \
                mock.patch.object(tts.TTSManager, "_play_routed", return_value=True) as routed:
            manager = tts.TTSManager()
            try:
                self.assertTrue(manager.speak("Speaker test", block=True,
                                              device="plughw:3,0", timeout=2))
            finally:
                manager.shutdown()
        routed.assert_called_once_with(self.wav, "wav", "plughw:3,0")
        self.assertFalse(manager.worker_thread.is_alive())

    def test_espeak_command_generates_wav_without_playing_default_output(self):
        for binary in ("espeak-ng", "espeak"):
            with self.subTest(binary=binary):
                paths = []
                def synthesize(command, **kwargs):
                    self.assertEqual(command[0], "/usr/bin/" + binary)
                    self.assertEqual(command[command.index("-v") + 1], "en")
                    self.assertEqual(command[command.index("-s") + 1], "180")
                    self.assertIn("--stdin", command)
                    self.assertEqual(kwargs["input"], b"Speaker test")
                    path = Path(command[command.index("-w") + 1])
                    paths.append(path)
                    path.write_bytes(self.wav)
                    return SimpleNamespace(returncode=0)
                with mock.patch.object(tts.shutil, "which", side_effect=lambda name: (
                        "/usr/bin/" + name if name == binary else None)), \
                        mock.patch.object(tts.subprocess, "run", side_effect=synthesize):
                    self.assertEqual(tts._espeak_to_wav("Speaker test", 180), self.wav)
                self.assertTrue(paths)
                self.assertFalse(paths[0].exists())

    def test_failed_espeak_does_not_return_partial_audio(self):
        paths = []
        def fail(command, **kwargs):
            path = Path(command[command.index("-w") + 1])
            paths.append(path)
            path.write_bytes(self.wav)
            raise subprocess.CalledProcessError(1, command)
        with mock.patch.object(tts.shutil, "which", return_value="/usr/bin/espeak"), \
                mock.patch.object(tts.subprocess, "run", side_effect=fail):
            self.assertEqual(tts._espeak_to_wav("Speaker test", 180), b"")
        self.assertFalse(paths[0].exists())


if __name__ == "__main__":
    unittest.main()
