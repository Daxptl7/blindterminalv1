"""Exercise exclusive audio-device ownership without real audio hardware."""
import importlib.util
import threading
import unittest
from pathlib import Path
from unittest import mock

from modules import tts


class TTSDeviceLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.pg = mock.Mock()
        self.opened = False
        self.pg.mixer.get_init.side_effect = lambda: (
            (24000, -16, 1) if self.opened else None)
        self.pg.mixer.init.side_effect = lambda **kwargs: self.set_open(True)
        self.pg.mixer.quit.side_effect = lambda: self.set_open(False)
        self.pg.mixer.music.get_busy.return_value = False
        self.pg.mixer.get_busy.return_value = False
        self.manager = tts.TTSManager.__new__(tts.TTSManager)
        self.manager.running = True
        self.manager._proc_lock = threading.Lock()
        self.manager._proc = None

    def set_open(self, value):
        self.opened = value

    def test_import_does_not_reserve_default_output(self):
        spec = importlib.util.spec_from_file_location(
            "tts_import_check", Path(tts.__file__))
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict("sys.modules", {"pygame": self.pg}), \
                mock.patch("modules.config_loader.load_settings", return_value={}), \
                mock.patch("subprocess.run", return_value=mock.Mock(stdout="")):
            spec.loader.exec_module(module)
        self.assertTrue(module.PYGAME_AVAILABLE)
        self.pg.mixer.init.assert_not_called()

    def test_pygame_releases_output_after_mp3_and_wav(self):
        for fmt in ("mp3", "wav"):
            with self.subTest(fmt=fmt), \
                    mock.patch.object(tts, "PYGAME_AVAILABLE", True), \
                    mock.patch.object(tts, "pygame", self.pg, create=True):
                self.assertTrue(self.manager._play_pygame(b"audio", fmt))
                self.assertFalse(self.opened)
        self.assertEqual(self.pg.mixer.init.call_count, 2)
        self.assertEqual(self.pg.mixer.quit.call_count, 2)
        self.pg.mixer.music.load.assert_called_once()
        self.pg.mixer.Sound.assert_called_once()

    def test_pygame_releases_output_when_playback_fails(self):
        self.pg.mixer.music.load.side_effect = RuntimeError("bad audio")
        with mock.patch.object(tts, "PYGAME_AVAILABLE", True), \
                mock.patch.object(tts, "pygame", self.pg, create=True):
            self.assertFalse(self.manager._play_pygame(b"bad", "mp3"))
        self.assertFalse(self.opened)
        self.pg.mixer.quit.assert_called_once()

    def test_pygame_init_failure_allows_next_backend(self):
        self.pg.mixer.init.side_effect = RuntimeError("no default output")
        with mock.patch.object(tts, "PYGAME_AVAILABLE", True), \
                mock.patch.object(tts, "pygame", self.pg, create=True):
            self.assertFalse(self.manager._play_pygame(b"audio", "wav"))
        self.assertFalse(self.opened)

    def test_routed_player_releases_existing_mixer_before_opening_card(self):
        self.opened = True

        def open_player(command, **kwargs):
            # Model an exclusive ALSA card: opening aplay while SDL owns it
            # would return the Device or resource busy error seen on the Pi.
            self.assertFalse(self.opened, "SDL still owns the output card")
            self.assertEqual(command[command.index("-D") + 1], "plughw:3,0")
            return mock.Mock(returncode=0)

        with mock.patch.object(tts, "PYGAME_AVAILABLE", True), \
                mock.patch.object(tts, "pygame", self.pg, create=True), \
                mock.patch.object(tts.shutil, "which", side_effect=lambda name: (
                    "/usr/bin/aplay" if name == "aplay" else None)), \
                mock.patch.object(tts.subprocess, "Popen", side_effect=open_player) as player:
            self.assertTrue(self.manager._play_routed(b"wav", "wav", "plughw:3,0"))
        self.pg.mixer.quit.assert_called_once()
        player.assert_called_once()

    def test_routed_playback_works_without_pygame(self):
        with mock.patch.object(tts, "PYGAME_AVAILABLE", False), \
                mock.patch.object(tts.shutil, "which", side_effect=lambda name: (
                    "/usr/bin/aplay" if name == "aplay" else None)), \
                mock.patch.object(tts.subprocess, "Popen", return_value=mock.Mock(returncode=0)):
            self.assertTrue(self.manager._play_routed(b"wav", "wav", "plughw:3,0"))


if __name__ == "__main__":
    unittest.main()
