import threading
import unittest
from unittest import mock
import numpy as np
from modules.processing_feedback import run_with_feedback, ProcessingCancelled
from modules import ocr


class FeedbackTests(unittest.TestCase):
    def test_fast_work_is_silent(self):
        speak = mock.Mock()
        self.assertEqual(run_with_feedback(lambda: 'done', speak), 'done')
        speak.assert_not_called()

    def test_slow_work_announces_then_stops_before_return(self):
        finish = threading.Event()
        messages = []
        def speak(message):
            messages.append(message)
            if len(messages) == 2:
                finish.set()
        def operation():
            self.assertTrue(finish.wait(2))
            return 'done'
        self.assertEqual(run_with_feedback(operation, speak, first_delay=.01,
                         interval=.01), 'done')
        self.assertEqual(len(messages), 2)

    def test_cancellation_discards_result_and_suppresses_speech(self):
        speak = mock.Mock()
        stopped = threading.Event()
        def operation():
            stopped.set()
            return 'private text'
        with self.assertRaises(ProcessingCancelled):
            run_with_feedback(operation, speak, stopped.is_set)
        speak.assert_not_called()

    def test_error_propagates(self):
        with self.assertRaisesRegex(RuntimeError, 'offline'):
            run_with_feedback(mock.Mock(side_effect=RuntimeError('offline')), mock.Mock())

    def test_max_updates_is_bounded(self):
        # Control Event.wait while running the real worker to avoid timing races.
        event = threading.Event()
        operation = mock.Mock(return_value='done')
        class WaitSequence:
            def __init__(self):
                self.waits = 0
            def set(self):
                event.set()
            def wait(self, delay):
                self.waits += 1
                if self.waits < 6:
                    return False
                return event.wait(2)
        speak = mock.Mock()
        with mock.patch('modules.processing_feedback.threading.Event', return_value=WaitSequence()), \
             mock.patch('modules.processing_feedback.threading.Thread') as thread:
            thread.return_value.start.side_effect = lambda: thread.call_args.kwargs['target']()
            self.assertEqual(run_with_feedback(operation, speak), 'done')
        self.assertEqual(speak.call_count, 2)


class OCRFallbackFeedbackTests(unittest.TestCase):
    def test_online_failure_announces_local_fallback_and_reads_once(self):
        speak = mock.Mock()
        frame = np.full((100, 100, 3), 200, np.uint8)
        with mock.patch.dict(ocr._config, {'ocr_engine':'auto', 'ocr_gemini_first':True,
                'gemini_api_key':'test', 'ocr_guidance_enabled':False}), \
             mock.patch.object(ocr._camera, 'open', return_value=True), \
             mock.patch.object(ocr._camera, 'capture', return_value=frame), \
             mock.patch.object(ocr._gemini_engine, 'extract_text', side_effect=RuntimeError('offline')), \
             mock.patch.object(ocr._tesseract_engine, 'extract_text', return_value='Local document text') as local:
            self.assertEqual(ocr.scan_and_read(speak_fn=speak), 'Local document text')
        local.assert_called_once()
        messages = [call.args[0] for call in speak.call_args_list]
        self.assertTrue(any('online reader failed' in message for message in messages))
        self.assertTrue(any('Local processing may take longer' in message for message in messages))
        self.assertFalse(any('Local document text' in message for message in messages))

    def test_cancelled_online_call_does_not_start_fallback(self):
        stop = threading.Event()
        def online(*args, **kwargs):
            stop.set()
            return 'text'
        with mock.patch.dict(ocr._config, {'ocr_engine':'auto', 'ocr_gemini_first':True,
                'gemini_api_key':'test', 'ocr_guidance_enabled':False}), \
             mock.patch.object(ocr._camera, 'open', return_value=True), \
             mock.patch.object(ocr._camera, 'capture', return_value=np.zeros((10,10,3), np.uint8)), \
             mock.patch.object(ocr._gemini_engine, 'extract_text', side_effect=online), \
             mock.patch.object(ocr._tesseract_engine, 'extract_text') as local:
            self.assertEqual(ocr.scan_and_read(speak_fn=mock.Mock(), stop_check=stop.is_set),
                             'OCR scan cancelled.')
            local.assert_not_called()
