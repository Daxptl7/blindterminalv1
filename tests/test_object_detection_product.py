import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


class ObjectDetectionProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.od = importlib.import_module("modules.object_detection")

    def detection(self, name="person", position="center", confidence=0.8,
                  box=(100, 50, 300, 450)):
        return {
            "name": name,
            "position": position,
            "conf": confidence,
            "box": box,
            "center_x": (box[0] + box[2]) / 2,
        }

    def test_model_resolution_uses_local_asset_without_runtime_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "yolov8n.pt"
            model.write_bytes(b"test model placeholder")
            resolved = self.od._resolve_model_path(
                settings={"yolo_allow_download": False},
                search_dirs=[root],
            )
        self.assertEqual(resolved, str(model))

    def test_model_resolution_refuses_implicit_production_download(self):
        with tempfile.TemporaryDirectory() as directory:
            resolved = self.od._resolve_model_path(
                settings={"yolo_allow_download": False},
                search_dirs=[Path(directory)],
            )
        self.assertIsNone(resolved)

    def test_temporal_filter_requires_repeated_evidence(self):
        filter_ = self.od.TemporalDetectionFilter(window=5, min_hits=3)
        person = self.detection()
        self.assertEqual(filter_.update([person]), [])
        self.assertEqual(filter_.update([]), [])
        self.assertEqual(filter_.update([person]), [])
        self.assertEqual(filter_.update([]), [])
        self.assertEqual(filter_.update([person]), [person])

    def test_temporal_filter_does_not_announce_stale_object(self):
        filter_ = self.od.TemporalDetectionFilter(window=3, min_hits=2)
        person = self.detection()
        filter_.update([person])
        self.assertEqual(filter_.update([person]), [person])
        self.assertEqual(filter_.update([]), [])

    def test_description_is_short_spatial_and_prioritized(self):
        detections = [
            self.detection("bottle", "center", 0.91, (250, 100, 390, 440)),
            self.detection("person", "left", 0.80, (0, 10, 250, 470)),
            self.detection("chair", "right", 0.75, (400, 150, 630, 470)),
        ]
        spoken = self.od.describe_detections(detections, max_items=2)
        self.assertEqual(spoken, "I see a person on your left, and a chair on your right.")
        self.assertNotIn("bottle", spoken)

    def test_auto_camera_falls_back_to_rpicam(self):
        cap = mock.Mock()
        with mock.patch.object(self.od, "_open_opencv_camera", return_value=None), \
             mock.patch.object(self.od, "_open_rpicam_camera", return_value=cap):
            self.assertIs(self.od._open_camera(0, backend="auto"), cap)

    def test_rpicam_adapter_decodes_mjpeg_stdout(self):
        expected = np.full((24, 32, 3), 127, dtype=np.uint8)
        encoded_ok, encoded = self.od.cv2.imencode(".jpg", expected)
        self.assertTrue(encoded_ok)

        stdout = mock.Mock()
        stdout.fileno.return_value = 99
        process = mock.Mock()
        process.poll.return_value = None
        process.stdout = stdout
        with mock.patch.object(self.od.subprocess, "Popen", return_value=process) as popen, \
             mock.patch.object(self.od.select, "select",
                               return_value=([stdout], [], [])), \
             mock.patch.object(self.od.os, "read", return_value=encoded.tobytes()):
            capture = self.od._RpicamMjpegCapture("/usr/bin/rpicam-vid")
            ok, decoded = capture.read()
            capture.release()

        self.assertTrue(ok)
        self.assertEqual(decoded.shape, expected.shape)
        command = popen.call_args.args[0]
        self.assertIn("mjpeg", command)
        self.assertEqual(command[-2:], ["--output", "-"])
        process.terminate.assert_called_once_with()

    def test_stream_callback_receives_stable_spatial_detection(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, frame)

        box = SimpleNamespace(
            xyxy=np.array([[240.0, 40.0, 400.0, 460.0]]),
            conf=np.array([0.88]),
            cls=np.array([0]),
        )
        result = SimpleNamespace(boxes=[box])
        model = mock.Mock()
        model.names = {0: "person"}
        model.predict.return_value = [result]
        passthrough = self.od.TemporalDetectionFilter(window=1, min_hits=1)
        received = []

        with mock.patch.object(self.od, "_get_model", return_value=model), \
             mock.patch.object(self.od, "_open_camera", return_value=cap), \
             mock.patch.object(self.od, "TemporalDetectionFilter",
                               return_value=passthrough), \
             mock.patch.object(self.od, "INFERENCE_INTERVAL_S", 0.0):
            self.od.stream_detect(
                callback=lambda text, detections: received.append(
                    (text, detections)
                ) or True,
                max_frames=1,
            )

        self.assertEqual(received[0][0], "I see a person ahead.")
        self.assertEqual(received[0][1][0]["position"], "center")
        cap.release.assert_called_once()

    def test_only_button_three_stops_object_detection(self):
        main = importlib.import_module("main")
        self.assertFalse(main._object_detection_stop_requested(None))
        self.assertFalse(main._object_detection_stop_requested("1"))
        self.assertFalse(main._object_detection_stop_requested("2"))
        self.assertTrue(main._object_detection_stop_requested("3"))

    def test_mode_ignores_buttons_one_and_two_before_button_three(self):
        main = importlib.import_module("main")
        serial = mock.Mock()
        serial.wait_for_raw_button.side_effect = ["1", "2", "3"]
        stopped = []

        def run_detection(callback, stop_event):
            stopped.append(stop_event.wait(timeout=1.0))

        detector = SimpleNamespace(
            prepare_model=mock.Mock(),
            run_detection=mock.Mock(side_effect=run_detection),
        )
        with mock.patch.dict(main._modules, {"objdetect": detector}), \
             mock.patch.object(main, "_morse_serial_singleton", serial), \
             mock.patch.object(main, "_drain_button_messages"), \
             mock.patch.object(main, "_speak"):
            main.mode_object_detection()

        self.assertEqual(stopped, [True])
        self.assertEqual(serial.wait_for_raw_button.call_count, 3)
        detector.prepare_model.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
