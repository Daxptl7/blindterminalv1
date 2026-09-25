import ast
from pathlib import Path
import unittest
from unittest import mock
from modules import confidential_mode as privacy


class PrivacyButtonTests(unittest.TestCase):
    def setUp(self):
        self.serial = mock.Mock()
        self.serial.get_message.return_value = None
        self.audio = mock.patch.object(privacy, 'PrivateAudio').start()
        self.speak = mock.patch.object(privacy.tts, 'speak').start()
        self.addCleanup(mock.patch.stopall)

    def test_buttons_choose_private_or_speaker(self):
        for button, expected in ((1, 'PRIVATE'), (2, 'NORMAL')):
            self.serial.wait_for_raw_button.return_value = button
            self.assertEqual(privacy.ask_confidentiality(self.serial), expected)
        self.assertTrue(all('say ' not in call.args[0] for call in self.speak.call_args_list))

    def test_press_during_prompt_is_preserved(self):
        queue = []
        self.serial.get_message.side_effect = lambda **kwargs: queue.pop(0) if queue else None
        self.serial.wait_for_raw_button.side_effect = lambda **kwargs: queue.pop(0) if queue else None
        self.speak.side_effect = lambda *args, **kwargs: queue.append(1)
        self.assertEqual(privacy.ask_confidentiality(self.serial), 'PRIVATE')

    def test_unrelated_button_does_not_end_selection(self):
        self.serial.wait_for_raw_button.side_effect = [3, 1]
        self.assertEqual(privacy.ask_confidentiality(self.serial), 'PRIVATE')

    def test_timeout_cancels_without_document_playback(self):
        self.serial.wait_for_raw_button.return_value = None
        self.assertIsNone(privacy.speak_document_with_privacy_check('SECRET', 'eng', self.serial))
        self.assertFalse(any('SECRET' in call.args[0] for call in self.speak.call_args_list))

    def test_missing_buttons_cancels(self):
        self.assertIsNone(privacy.speak_with_privacy_check('SECRET', 'eng'))
        self.assertFalse(any('SECRET' in call.args[0] for call in self.speak.call_args_list))

    def test_no_voice_import_or_listen_path(self):
        source = Path('modules/confidential_mode.py').read_text()
        self.assertNotIn('_voice.listen', source)
        self.assertNotIn('import voice', source)


class ExplanationButtonTests(unittest.TestCase):
    def scope(self):
        names = {'mode_ocr_scan', '_select_with_buttons', '_button_choice', '_drain_button_messages'}
        nodes = [n for n in ast.parse(Path('main.py').read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
        import time
        serial = mock.Mock()
        serial.get_message.return_value = None
        ocr = mock.Mock()
        ocr.scan_and_read.return_value = 'Document content'
        ai = mock.Mock()
        ai.ask_ai.return_value = 'Explanation'
        scope = dict(time=time, logger=mock.Mock(), _modules={'ocr':ocr, 'ai_query':ai},
                     _has=lambda name: name in ('ocr', 'ai_query'), _speak=mock.Mock(),
                     _flush_speech=mock.Mock(), _morse_serial_singleton=serial,
                     _button3_stop_signal=lambda: (None, mock.Mock()),
                     _BUTTON_CHOICE_TIMEOUT_S=15, _BUTTON_DOUBLE_WINDOW_S=2,
                     _keyboard_input=mock.Mock(return_value=None))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'main.py', 'exec'), scope)
        return scope, serial, ai

    def test_yes_after_document_calls_explanation_once(self):
        scope, serial, ai = self.scope()
        serial.wait_for_raw_button.side_effect = [3, 1]
        with mock.patch.object(privacy, 'speak_document_with_privacy_check', return_value='NORMAL'):
            scope['mode_ocr_scan']()
        ai.ask_ai.assert_called_once()
        prompt = next(call for call in scope['_speak'].call_args_list if 'Would you like' in call.args[0])
        self.assertTrue(prompt.kwargs['block'])
        self.assertGreater(serial.wait_for_raw_button.call_args_list[0].kwargs['timeout'], 14)

    def test_no_and_timeout_do_not_call_ai(self):
        for button in (2, None):
            scope, serial, ai = self.scope()
            serial.wait_for_raw_button.return_value = button
            with mock.patch.object(privacy, 'speak_document_with_privacy_check', return_value='NORMAL'):
                scope['mode_ocr_scan']()
            ai.ask_ai.assert_not_called()

    def test_prompt_does_not_discard_early_press(self):
        scope, serial, _ = self.scope()
        queue = []
        serial.get_message.side_effect = lambda **kwargs: queue.pop(0) if queue else None
        serial.wait_for_raw_button.side_effect = lambda **kwargs: queue.pop(0) if queue else None
        scope['_speak'].side_effect = lambda *args, **kwargs: queue.append(1)
        self.assertEqual(scope['_select_with_buttons'](('1','2'), 'Choose'), '1')

    def test_cancelled_privacy_does_not_prompt_explanation(self):
        scope, _, ai = self.scope()
        with mock.patch.object(privacy, 'speak_document_with_privacy_check', return_value=None):
            scope['mode_ocr_scan']()
        ai.ask_ai.assert_not_called()
        self.assertFalse(any('Would you like' in c.args[0] for c in scope['_speak'].call_args_list))
