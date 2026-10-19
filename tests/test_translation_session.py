"""Translation session navigation and physical exit-gesture regressions."""

import importlib
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from modules.translation_controls import TranslationButtons, TranslationExited


class TranslationGestureTests(unittest.TestCase):
    def setUp(self):
        self.buttons = TranslationButtons(None)

    def send(self, message, at):
        self.buttons._accept(message, at)

    def test_single_live_selection_waits_five_seconds(self):
        self.send("RAW:3", 10)
        self.send("WORD_SPACE", 10.6)
        self.send(None, 14.99)
        self.assertIsNone(self.buttons.get_message(timeout=.001))
        self.send(None, 15)
        self.assertEqual(self.buttons.wait_for_raw_button(timeout=.01), 3)
        self.assertFalse(self.buttons.exited.is_set())

    def test_triple_exits_before_single_live_action_is_released(self):
        for message, at in [("RAW:3", 0), ("RAW:3", .2),
                            ("CONFIRM", .3), ("RAW:3", .4)]:
            self.send(message, at)
        self.assertTrue(self.buttons.exited.is_set())
        with self.assertRaises(TranslationExited):
            self.buttons.wait_for_raw_button(timeout=.01)

    def test_firmware_confirm_does_not_count_as_an_extra_click(self):
        for message, at in [("RAW:3", 0), ("RAW:3", .2), ("CONFIRM", .3)]:
            self.send(message, at)
        self.assertFalse(self.buttons.exited.is_set())
        self.send(None, 5)
        self.assertFalse(self.buttons.exited.is_set())

    def test_slow_or_interrupted_clicks_do_not_exit(self):
        for events in [
            [("RAW:3", 0), ("RAW:3", 6), ("RAW:3", 12)],
            [("RAW:3", 0), ("RAW:1", .1), ("RAW:3", .2), ("RAW:3", .3)],
        ]:
            with self.subTest(events=events):
                self.buttons = TranslationButtons(None)
                for message, at in events:
                    self.send(message, at)
                self.assertFalse(self.buttons.exited.is_set())

    def test_single_click_finishes_recording_only_after_wait(self):
        self.send("RAW:3", 0)
        self.assertFalse(self.buttons.recording_stopped())
        self.send(None, 5)
        self.assertTrue(self.buttons.recording_stopped())
        self.assertFalse(self.buttons.exited.is_set())

    def test_triple_stops_recording_as_exit(self):
        for at in (0, .2, .4):
            self.send("RAW:3", at)
        self.assertTrue(self.buttons.recording_stopped())
        self.assertTrue(self.buttons.exited.is_set())

    def test_morse_double_click_submits_text_after_wait(self):
        for message, at in [("LETTER:H", 0), ("LETTER:I", .2),
                            ("RAW:3", .4), ("RAW:3", .6), ("CONFIRM", .7)]:
            self.send(message, at)
        self.send(None, 5.4)
        self.assertEqual(self.buttons.type_word(timeout=.01), "HI")

    def test_morse_triple_does_not_submit_early_double_click(self):
        for message, at in [("LETTER:H", 0), ("RAW:3", .1),
                            ("RAW:3", .2), ("CONFIRM", .3), ("RAW:3", .4)]:
            self.send(message, at)
        with self.assertRaises(TranslationExited):
            self.buttons.type_word(timeout=.01)

    def test_stage_clear_does_not_erase_an_exit(self):
        self.buttons.exited.set()
        with self.assertRaises(TranslationExited):
            self.buttons.clear()

    def test_reader_stops_without_closing_shared_serial_connection(self):
        messages = queue.Queue()
        serial = mock.Mock()
        def read(timeout):
            try:
                return messages.get(timeout=timeout)
            except queue.Empty:
                return None
        serial.get_message.side_effect = read
        with TranslationButtons(serial) as buttons:
            for message in ("RAW:3", "RAW:3", "CONFIRM", "RAW:3"):
                messages.put(message)
            self.assertTrue(buttons.exited.wait(timeout=1))
        self.assertFalse(buttons._thread.is_alive())
        serial.close.assert_not_called()


class TranslationSessionTests(unittest.TestCase):
    def setUp(self):
        self.main = importlib.import_module("main")
        self.translator = mock.Mock()
        self.translator.LANG_TO_TTS = {"en": "eng", "hi": "hin", "gu": "guj"}
        self.translator.translate_ex.return_value = SimpleNamespace(
            text="नमस्ते", lang="hi", translated=True, source="mock", error=None)
        self.privacy = importlib.import_module("modules.confidential_mode")
        mock.patch.object(self.privacy.tts, 'require_private_output', return_value='earphone').start()
        self.live = mock.Mock()
        mock.patch.dict(self.main._modules, {
            "translator": self.translator, "privacy": self.privacy,
            "voice": mock.Mock(), "live_translator": self.live,
        }).start()
        mock.patch.object(self.main, "_morse_serial_singleton", None).start()
        self.speak = mock.patch.object(self.main, "_speak").start()
        self.listen_impl = self.main._translation_listen
        self.listen = mock.patch.object(self.main, "_translation_listen",
                                        return_value="hello").start()
        self.addCleanup(mock.patch.stopall)

    def run_choices(self, choices):
        with mock.patch.object(self.main, "_keyboard_input", side_effect=choices) as keyboard:
            self.main.mode_translate()
        self.assertEqual(keyboard.call_count, len(choices))

    def test_yes_records_another_translation_and_no_returns_to_input_menu(self):
        self.run_choices(["1", "1", "2", "2", "1", "1", "2", "2", "2", "333"])
        self.assertEqual(self.listen.call_count, 2)
        self.assertEqual(self.translator.translate_ex.call_count, 2)
        prompts = [call.args[0] for call in self.speak.call_args_list]
        self.assertEqual(sum("Would you like another translation" in p for p in prompts), 2)
        self.assertEqual(sum(p.startswith("Translation mode.") for p in prompts), 2)
        self.assertEqual(sum(p.startswith("Your result is ready.") for p in prompts), 2)

    def test_no_allows_switching_from_audio_to_morse(self):
        self.run_choices(["1", "1", "2", "2", "2", "2", ".-", "1", "3", "2", "2", "333"])
        self.listen.assert_called_once()
        self.assertEqual(self.translator.translate_ex.call_args_list, [
            mock.call("hello", from_lang="en", to_lang="hi", privacy=False),
            mock.call("A", from_lang="en", to_lang="gu", privacy=False),
        ])

    def test_physical_triple_during_input_prompt_never_launches_live(self):
        messages = queue.Queue()
        serial = mock.Mock()
        def read(timeout):
            try:
                return messages.get(timeout=timeout)
            except queue.Empty:
                return None
        serial.get_message.side_effect = read
        def speak(text, **kwargs):
            if text.startswith("Translation mode."):
                for message in ("RAW:3", "RAW:3", "CONFIRM", "RAW:3"):
                    messages.put(message)
        with mock.patch.object(self.main, "_morse_serial_singleton", serial), \
                mock.patch.object(self.main, "_BUTTON_LETTER_SETTLE_S", 0), \
                mock.patch.object(self.main, "_speak", side_effect=speak):
            self.main.mode_translate()
        self.live.LiveTranslator.assert_not_called()
        self.translator.translate_ex.assert_not_called()

    def test_exit_while_live_returns_without_repeat_prompt(self):
        buttons = TranslationButtons(None)
        self.live.LiveTranslator.return_value.run_live.side_effect = (
            lambda **kwargs: buttons.exited.set())
        with mock.patch("modules.translation_controls.TranslationButtons", return_value=buttons):
            self.run_choices(["3", "1"])
        self.live.LiveTranslator.assert_called_once_with(lang_pair="en-gu")
        self.assertFalse(any("Would you like another translation" in c.args[0]
                             for c in self.speak.call_args_list))

    def test_exit_during_speech_is_handled_as_navigation(self):
        buttons = TranslationButtons(None)
        def speak(text, **kwargs):
            if text.startswith("Translation mode."):
                buttons.exited.set()
                from modules.tts import SpeechCancelled
                raise SpeechCancelled("interrupted")
        with mock.patch("modules.translation_controls.TranslationButtons", return_value=buttons), \
                mock.patch.object(self.main, "_speak", side_effect=speak) as say:
            self.main.mode_translate()
        say.assert_any_call("Leaving translation. Returning to the main menu.", block=True)
        self.translator.translate_ex.assert_not_called()

    def test_triple_during_language_selection_exits_without_translation(self):
        buttons = TranslationButtons(None)
        def speak(text, **kwargs):
            if text.startswith("Choose the source language."):
                for at in (0, .2, .4):
                    buttons._accept("RAW:3", at)
        with mock.patch("modules.translation_controls.TranslationButtons", return_value=buttons), \
                mock.patch.object(self.main, "_speak", side_effect=speak):
            self.run_choices(["1"])
        self.listen.assert_not_called()
        self.translator.translate_ex.assert_not_called()

    def test_live_ending_normally_asks_repeat_then_offers_input_choices(self):
        self.run_choices(["3", "2", "2", "333"])
        self.live.LiveTranslator.assert_called_once_with(lang_pair="en-hi")
        self.assertTrue(any("Would you like another translation" in c.args[0]
                            for c in self.speak.call_args_list))

    def test_triple_during_result_output_choice_exits_without_playing_it(self):
        buttons = TranslationButtons(None)
        def speak(text, **kwargs):
            if text.startswith("Your result is ready."):
                for at in (0, .2, .4):
                    buttons._accept("RAW:3", at)
        with mock.patch("modules.translation_controls.TranslationButtons", return_value=buttons), \
                mock.patch.object(self.main, "_speak", side_effect=speak) as say:
            self.run_choices(["1", "1", "2"])
        self.assertFalse(any(c.args[0] == "नमस्ते" for c in say.call_args_list))

    def test_exit_during_processing_does_not_speak_a_late_result(self):
        buttons = TranslationButtons(None)
        def translate(*args, **kwargs):
            buttons.exited.set()
            return self.translator.translate_ex.return_value
        self.translator.translate_ex.side_effect = translate
        with mock.patch("modules.translation_controls.TranslationButtons", return_value=buttons):
            self.run_choices(["1", "1", "2"])
        self.assertFalse(any(c.args[0] == "नमस्ते" for c in self.speak.call_args_list))

    def test_voice_exit_skips_transcription(self):
        voice = importlib.import_module("modules.voice")
        with mock.patch.object(voice, "record", return_value="/tmp/unused.wav"), \
                mock.patch.object(voice, "transcribe_file") as transcribe:
            self.assertIsNone(voice.listen(cancel_check=lambda: True))
        transcribe.assert_not_called()

    def test_exit_cancels_slow_transcription_after_capture_has_closed(self):
        voice = importlib.import_module("modules.voice")
        buttons = TranslationButtons(None)
        started, release = threading.Event(), threading.Event()
        def transcribe(*args):
            started.set()
            release.wait(timeout=2)
            return "late recognition"
        def exit_after_transcription_starts():
            if started.wait(timeout=1):
                buttons.exited.set()
        worker = threading.Thread(target=exit_after_transcription_starts)
        worker.start()
        try:
            with mock.patch.dict(self.main._modules, {"voice": voice}), \
                    mock.patch.object(voice, "record", return_value="/tmp/unused.wav"), \
                    mock.patch.object(voice, "_active_engines", return_value=["mock"]), \
                    mock.patch.object(voice, "transcribe_file", side_effect=transcribe):
                with self.assertRaises(TranslationExited):
                    self.listen_impl(buttons, "en-IN")
            self.assertTrue(started.is_set())
        finally:
            release.set()
            worker.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
