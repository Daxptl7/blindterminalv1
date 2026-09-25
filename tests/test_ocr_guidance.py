"""Framing decisions and guided scan lifecycle without physical cameras."""
import unittest
from unittest import mock
import cv2
import numpy as np
from modules import ocr
from modules.ocr_guidance import Assessment, FrameAnalyzer, GuidanceGate, guided_capture


def page(x=160, y=50, width=320, height=380, text=True):
    frame = np.full((480, 640, 3), 85, np.uint8)
    cv2.rectangle(frame, (x, y), (x+width, y+height), (245, 245, 245), -1)
    if text:
        for line in range(y+25, min(y+height-10, 470), 22):
            cv2.putText(frame, 'Read this page', (x+12, line),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (20, 20, 20), 1)
    return frame


class FramingTests(unittest.TestCase):
    def assess(self, frame, config=None):
        return FrameAnalyzer(config).assess(frame).code

    def test_centered_page_ready(self):
        self.assertEqual(self.assess(page()), 'ready')

    def test_directions(self):
        for expected, args in [('left', (30, 90, 240, 300)),
                               ('right', (365, 90, 240, 300)),
                               ('up', (200, 20, 240, 230)),
                               ('down', (200, 230, 240, 230))]:
            with self.subTest(expected=expected):
                self.assertEqual(self.assess(page(*args)), expected)

    def test_distance_and_darkness(self):
        self.assertEqual(self.assess(page(235, 150, 170, 180)), 'closer')
        # Paper edges are not evidence that printed text is cut off.
        self.assertEqual(self.assess(page(160, 5, 320, 470)), 'ready')
        self.assertEqual(self.assess(np.zeros((480, 640, 3), np.uint8)), 'dark')

    def test_blank_and_missing_pages_never_ready(self):
        self.assertNotEqual(self.assess(page(text=False)), 'ready')
        self.assertEqual(self.assess(np.full((480, 640, 3), 150, np.uint8)), 'search')

    def test_text_without_visible_paper_boundary_can_capture(self):
        frame = np.full((480, 640, 3), 245, np.uint8)
        for y in range(80, 400, 24):
            cv2.putText(frame, 'A clear readable line of text', (80, y),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (20, 20, 20), 1)
        self.assertEqual(self.assess(frame), 'ready')
        # One stray edge mark must not cause a move-away loop.
        cv2.putText(frame, 'X', (0, 35), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 20), 1)
        self.assertEqual(self.assess(frame), 'ready')

    def test_same_instruction_is_not_repeated_forever(self):
        gate = GuidanceGate({})
        assessment = Assessment('clipped', 'Move back')
        announcements = [gate.update(assessment, now)[1] for now in range(30)]
        self.assertEqual(sum(announcements), 1)

    def test_motion(self):
        analyzer = FrameAnalyzer()
        analyzer.assess(page())
        self.assertEqual(analyzer.assess(page(190)).code, 'moving')

    def test_mirrored_mount(self):
        self.assertEqual(self.assess(page(30, 90, 240, 300),
                                    {'ocr_guidance_mirror': True}), 'right')

    def test_gate_requires_stability_and_limits_speech(self):
        gate = GuidanceGate({})
        left = Assessment('left', 'left')
        self.assertEqual(gate.update(left, 0), (False, False))
        self.assertEqual(gate.update(left, .2), (False, True))
        self.assertEqual(gate.update(left, .4), (False, False))
        ready = Assessment('ready', 'ready')
        for now in (1, 1.2, 1.4):
            self.assertFalse(gate.update(ready, now)[0])
        self.assertTrue(gate.update(ready, 1.6)[0])


class CaptureTests(unittest.TestCase):
    def camera(self):
        camera = mock.Mock()
        camera.preview_frame.return_value = page()
        camera.capture.return_value = page()
        return camera

    @mock.patch('modules.ocr_guidance.time.sleep')
    def test_ready_captures_full_resolution_once(self, _):
        camera = self.camera()
        speak = mock.Mock()
        frame, status = guided_capture(camera, speak, None, {})
        self.assertIs(frame, camera.capture.return_value)
        self.assertEqual(status, '')
        camera.capture.assert_called_once()
        self.assertEqual(speak.call_args.args[0], 'Reading now.')
        camera.stop_preview.assert_called()

    def test_cancel_never_captures(self):
        camera = self.camera()
        frame, status = guided_capture(camera, mock.Mock(), lambda: True, {})
        self.assertIsNone(frame)
        self.assertIn('cancelled', status)
        camera.capture.assert_not_called()
        camera.stop_preview.assert_called()

    @mock.patch('modules.ocr_guidance.time.sleep')
    def test_rejected_still_restarts_preview(self, _):
        camera = self.camera()
        camera.capture.side_effect = [page(text=False), page()]
        frame, status = guided_capture(camera, mock.Mock(), None, {})
        self.assertEqual(status, '')
        self.assertIsNotNone(frame)
        self.assertEqual(camera.start_preview.call_count, 2)

    @mock.patch('modules.ocr_guidance.time.sleep')
    def test_uncertain_framing_gets_bounded_ocr_attempt(self, _):
        import itertools
        camera = self.camera()
        camera.preview_frame.return_value = page(30, 90, 240, 300)
        camera.capture.return_value = camera.preview_frame.return_value
        with mock.patch('modules.ocr_guidance.time.monotonic', side_effect=itertools.count()):
            frame, status = guided_capture(camera, mock.Mock(), None,
                                           {'ocr_guidance_auto_capture_s': 2})
        self.assertIsNotNone(frame)
        self.assertEqual(status, '')
        camera.capture.assert_called_once()
        camera.start_preview.assert_called_once()

    @mock.patch('modules.ocr_guidance.time.sleep')
    def test_still_crop_difference_does_not_restart_framing(self, _):
        camera = self.camera()
        camera.capture.return_value = page(30, 90, 240, 300)
        frame, status = guided_capture(camera, mock.Mock(), None, {})
        self.assertIsNotNone(frame)
        self.assertEqual(status, '')
        camera.start_preview.assert_called_once()

    @mock.patch('modules.ocr_guidance.time.sleep')
    def test_auto_attempt_does_not_capture_dark_view(self, _):
        import itertools
        camera = self.camera()
        camera.preview_frame.return_value = np.zeros((480, 640, 3), np.uint8)
        with mock.patch('modules.ocr_guidance.time.monotonic', side_effect=itertools.count()):
            frame, status = guided_capture(camera, mock.Mock(), None,
                    {'ocr_guidance_auto_capture_s': 2, 'ocr_guidance_timeout_s': 10})
        self.assertIsNone(frame)
        camera.capture.assert_not_called()

    def test_preview_failure(self):
        camera = self.camera()
        camera.start_preview.return_value = False
        frame, status = guided_capture(camera, mock.Mock(), None, {})
        self.assertIsNone(frame)
        self.assertIn('unavailable', status)
        camera.capture.assert_not_called()

    @mock.patch('modules.ocr_guidance.time.monotonic', side_effect=[0, 61])
    def test_timeout(self, _):
        camera = self.camera()
        self.assertEqual(guided_capture(camera, mock.Mock(), None, {})[1],
                         'OCR positioning timed out.')
        camera.stop_preview.assert_called()

    def test_scan_does_not_run_ocr_after_cancellation(self):
        with mock.patch.object(ocr._camera, 'open', return_value=True), \
             mock.patch('modules.ocr_guidance.guided_capture',
                        return_value=(None, 'OCR scan cancelled.')), \
             mock.patch.object(ocr._tesseract_engine, 'extract_text') as extract:
            self.assertEqual(ocr.scan_and_read(speak_fn=mock.Mock()), 'OCR scan cancelled.')
            extract.assert_not_called()


class PreviewTests(unittest.TestCase):
    def test_mjpeg_reader_decodes_chunked_stream_and_consumes_latest_once(self):
        import io
        encoded = cv2.imencode('.jpg', page())[1].tobytes()
        process = mock.Mock()
        process.stdout = io.BytesIO(encoded + encoded)
        process.poll.return_value = 0
        with mock.patch.object(ocr.subprocess, 'Popen', return_value=process):
            preview = ocr.LatestPreview(binary='/usr/bin/rpicam-vid')
            preview.thread.join(timeout=2)
            frame = preview.take()
            self.assertEqual(frame.shape, (480, 640, 3))
            self.assertIsNone(preview.take())
            preview.close()
            self.assertTrue(process.stdout.closed)

    def test_stale_preview_is_discarded(self):
        cap = mock.Mock()
        cap.read.return_value = (False, None)
        preview = ocr.LatestPreview(cap=cap)
        preview.thread.join(timeout=2)
        preview._publish(page())
        preview.timestamp -= 3
        self.assertIsNone(preview.take())
        preview.close()


class ModeTests(unittest.TestCase):
    def mode(self, results):
        # Execute only this mode, avoiding main's hardware/model imports.
        import ast
        from pathlib import Path
        tree = ast.parse(Path('main.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'mode_ocr_scan')
        module = mock.Mock()
        module.scan_and_read.side_effect = results
        speak = mock.Mock()
        cancel = mock.Mock()
        serial = mock.Mock()
        serial.wait_for_raw_button.return_value = 1
        scope = dict(logger=mock.Mock(), _has=lambda name: name == 'ocr',
                     _modules={'ocr': module}, _speak=speak,
                     _drain_button_messages=mock.Mock(),
                     _button3_stop_signal=lambda: (lambda: False, cancel),
                     _STOP_PRESSES=3, _morse_serial_singleton=serial)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'main.py', 'exec'), scope)
        scope['mode_ocr_scan']()
        return module, speak, cancel

    def test_cancel_announced_and_camera_released(self):
        module, speak, cancel = self.mode(['OCR scan cancelled.'])
        speak.assert_any_call('OCR scan cancelled.')
        module.release_camera.assert_called_once()
        cancel.assert_called_once()

    def test_timeout_retry_starts_new_scan(self):
        module, speak, cancel = self.mode(['OCR positioning timed out.', 'OCR scan cancelled.'])
        self.assertEqual(module.scan_and_read.call_count, 2)
        self.assertEqual(cancel.call_count, 2)
        module.release_camera.assert_called_once()


if __name__ == '__main__':
    unittest.main()
