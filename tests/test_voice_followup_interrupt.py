"""Button interruption and follow-up routing without audio or serial hardware."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from modules import tts
from modules.privacy_policy import PUBLIC, processing_session


def mode_scope(**values):
    tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ("mode_voice_ask", "_speak_voice_answer")]
    scope = {"logger": mock.Mock(), **values}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "main.py", "exec"), scope)
    return scope


class SpeechInterruptTests(unittest.TestCase):
    def manager(self):
        manager = tts.TTSManager.__new__(tts.TTSManager)
        manager.running = True
        manager.queue = mock.Mock()
        manager.stop = mock.Mock()
        return manager

    def test_press_during_wait_cancels_active_and_queued_speech(self):
        manager = self.manager()
        item = SimpleNamespace(done=mock.Mock(), error=None)
        item.done.wait.return_value = False
        check = mock.Mock(side_effect=[False, True])
        with processing_session(PUBLIC), mock.patch.object(tts, "_Utterance", return_value=item):
            with self.assertRaises(tts.SpeechCancelled):
                manager.speak("Answer", block=True, stop_check=check)
        manager.queue.put.assert_called_once()
        manager.stop.assert_called_once()

    def test_press_between_sections_prevents_next_section_being_queued(self):
        manager = self.manager()
        with self.assertRaises(tts.SpeechCancelled):
            manager.speak("Next section", block=True, stop_check=lambda: True)
        manager.queue.put.assert_not_called()
        manager.stop.assert_called_once()

    def test_interrupted_section_skips_remaining_sections(self):
        serial = mock.Mock()
        serial.wait_for_raw_button.return_value = 1

        def speak(_text, **options):
            self.assertTrue(options["stop_check"]())
            raise tts.SpeechCancelled("Stopped")

        scope = mode_scope(_morse_serial_singleton=serial,
                           _drain_button_messages=mock.Mock(),
                           _speak=mock.Mock(side_effect=speak))
        with mock.patch("modules.confidential_mode.split_text_for_speech",
                        return_value=["First section", "Unspoken section"]):
            self.assertTrue(scope["_speak_voice_answer"]("Answer"))
        scope["_speak"].assert_called_once()

    def test_interrupt_records_followup_with_context_without_yes_no_prompt(self):
        speak_answer = mock.Mock(side_effect=[True, False])
        ask = mock.Mock(side_effect=["Initial answer", "Follow-up answer"])
        scope = mode_scope(
            _has=lambda _name: True,
            _listen_until_stopped=mock.Mock(side_effect=["First question", "Explain that"]),
            _speak_voice_failure=mock.Mock(), _speak=mock.Mock(),
            _ask_ai_cancellable=ask,
            _select_with_buttons=mock.Mock(return_value="2"))
        scope["_speak_voice_answer"] = speak_answer
        scope["mode_voice_ask"]()
        self.assertEqual(scope["_listen_until_stopped"].call_count, 2)
        self.assertEqual(speak_answer.call_count, 2)
        scope["_select_with_buttons"].assert_called_once()
        context = ask.call_args.kwargs["context"]
        self.assertIn("First question", context)
        self.assertIn("Initial answer", context)
        self.assertIn("playback interrupted", context)


if __name__ == "__main__":
    unittest.main()
