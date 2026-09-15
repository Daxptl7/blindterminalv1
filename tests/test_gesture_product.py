import importlib
from pathlib import Path
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import numpy as np


@dataclass
class Point:
    x: float = 0.5
    y: float = 0.5
    z: float = 0.0


def make_hand(extended=(False, False, False, False)):
    landmarks = [Point() for _ in range(21)]
    landmarks[0] = Point(0.50, 0.86)
    for index, x, y in ((1, 0.43, 0.74), (2, 0.39, 0.69),
                        (3, 0.42, 0.72), (4, 0.46, 0.75)):
        landmarks[index] = Point(x, y)
    chains = ((5, 6, 7, 8), (9, 10, 11, 12),
              (13, 14, 15, 16), (17, 18, 19, 20))
    for chain, is_extended, x in zip(chains, extended, (0.36, 0.46, 0.56, 0.66)):
        mcp, pip, dip, tip = chain
        landmarks[mcp] = Point(x, 0.62)
        landmarks[pip] = Point(x, 0.48)
        if is_extended:
            landmarks[dip] = Point(x, 0.34)
            landmarks[tip] = Point(x, 0.20)
        else:
            landmarks[dip] = Point(x + 0.08, 0.54)
            landmarks[tip] = Point(x + 0.09, 0.65)
    return landmarks


def rotate(hand, degrees):
    import math

    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    rotated = []
    for point in hand:
        x, y = point.x - 0.5, point.y - 0.5
        rotated.append(Point(
            0.5 + x * cosine - y * sine,
            0.5 + x * sine + y * cosine,
            point.z,
        ))
    return rotated


class GestureProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gesture = importlib.import_module("modules.gesture_control")

    def test_production_vocabulary_maps_to_real_actions(self):
        cases = {
            (True, True, True, True): "MODE_SCAN",
            (True, True, False, False): "MODE_VOICE",
            (True, False, False, False): "OBJECT_DETECT",
            (True, True, True, False): "GPS_CHECK",
            (False, False, False, False): "STOP",
        }
        for fingers, expected in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(
                    self.gesture.classify_gesture(make_hand(fingers)), expected
                )

    def test_gestures_survive_in_plane_hand_rotation(self):
        for degrees in (90, 180, 270):
            with self.subTest(degrees=degrees):
                hand = rotate(make_hand((True, True, False, False)), degrees)
                self.assertEqual(
                    self.gesture.classify_gesture(hand), "MODE_VOICE"
                )

    def test_framing_guidance_is_actionable(self):
        ready = make_hand((True, True, True, True))
        self.assertEqual(self.gesture.framing_guidance(ready), "READY")

        too_far = [Point(0.5 + (p.x - 0.5) * 0.2,
                         0.5 + (p.y - 0.5) * 0.2, p.z) for p in ready]
        self.assertEqual(self.gesture.framing_guidance(too_far), "MOVE_CLOSER")

        left = [Point(p.x - 0.25, p.y, p.z) for p in ready]
        self.assertEqual(self.gesture.framing_guidance(left), "MOVE_RIGHT")

        cropped = list(ready)
        cropped[8] = Point(-0.01, 0.2)
        self.assertEqual(self.gesture.framing_guidance(cropped), "HAND_CROPPED")

    def test_vote_counts_unrecognized_frames_as_misses(self):
        vote = self.gesture.GestureVoteWindow(
            window_ms=1000, fresh_ms=300, min_frames=5, min_support=0.60
        )
        for index, value in enumerate(
                ["MODE_SCAN", None, None, "MODE_SCAN", None, None, None]):
            vote.add(index * 50, value)
        self.assertIsNone(vote.vote(350))

    def test_vote_requires_stable_and_fresh_pose(self):
        vote = self.gesture.GestureVoteWindow(
            window_ms=1000, fresh_ms=300, min_frames=5, min_support=0.60
        )
        for index, value in enumerate(
                ["MODE_SCAN"] * 6 + [None] * 4):
            vote.add(index * 50, value)
        self.assertEqual(vote.vote(450), "MODE_SCAN")
        self.assertIsNone(vote.vote(900), "a pose lowered long ago must not fire")

    def test_missing_mediapipe_is_reported_to_main(self):
        received = []
        with mock.patch.object(self.gesture, "MEDIAPIPE_AVAILABLE", False):
            self.gesture.detect_gesture(callback_fn=received.append)
        self.assertEqual(received, ["MEDIAPIPE_UNAVAILABLE"])

    def test_missing_backend_is_reported_to_main(self):
        received = []
        with mock.patch.object(self.gesture, "MEDIAPIPE_AVAILABLE", True), \
             mock.patch.object(self.gesture, "CLASSIC_HANDS_AVAILABLE", False), \
             mock.patch.object(self.gesture, "TASKS_AVAILABLE", False):
            self.gesture.detect_gesture(callback_fn=received.append)
        self.assertEqual(received, ["BACKEND_UNAVAILABLE"])

    def test_missing_tasks_model_has_specific_recovery_message(self):
        received = []
        with mock.patch.object(self.gesture, "MEDIAPIPE_AVAILABLE", True), \
             mock.patch.object(self.gesture, "CLASSIC_HANDS_AVAILABLE", False), \
             mock.patch.object(self.gesture, "TASKS_AVAILABLE", True), \
             mock.patch.object(
                 self.gesture, "_resolve_hand_model_path",
                 return_value=Path("/definitely/missing/hand_landmarker.task"),
             ):
            self.gesture.detect_gesture(callback_fn=received.append)
        self.assertEqual(received, ["HAND_MODEL_MISSING"])

    def test_diagnostics_reports_ready_classic_backend_without_task_model(self):
        with mock.patch.object(self.gesture, "MEDIAPIPE_AVAILABLE", True), \
             mock.patch.object(self.gesture, "CLASSIC_HANDS_AVAILABLE", True), \
             mock.patch.object(self.gesture, "GESTURE_BACKEND", "auto"), \
             mock.patch.object(
                 self.gesture, "_resolve_hand_model_path",
                 return_value=Path("/definitely/missing/hand_landmarker.task"),
             ):
            info = self.gesture.diagnostics()
        self.assertTrue(info["backend_ready"])
        self.assertEqual(info["backend"], "classic")

    def test_auto_camera_can_fall_back_to_camera_module_three(self):
        rpicam = mock.Mock()
        with mock.patch.object(self.gesture, "GESTURE_CAMERA_BACKEND", "auto"), \
             mock.patch.object(self.gesture, "_open_usb_camera", return_value=None), \
             mock.patch.object(self.gesture, "_open_rpicam_camera", return_value=rpicam):
            self.assertIs(self.gesture._open_camera(0), rpicam)

    def test_shutter_uses_multiframe_vote_before_emitting_action(self):
        hand = make_hand((True, False, False, False))

        class Camera:
            def read(self):
                return True, np.zeros((480, 640, 3), dtype=np.uint8)

            def release(self):
                pass

        class Detector:
            def process(self, _frame):
                return SimpleNamespace(
                    multi_hand_landmarks=[SimpleNamespace(landmark=hand)]
                )

            def close(self):
                pass

        calls = 0

        def capture_check():
            nonlocal calls
            calls += 1
            return "capture" if calls >= 6 else None

        received = []

        def callback(name):
            received.append(name)
            return False

        with mock.patch.object(self.gesture, "MEDIAPIPE_AVAILABLE", True), \
             mock.patch.object(self.gesture, "CLASSIC_HANDS_AVAILABLE", True), \
             mock.patch.object(self.gesture, "GESTURE_BACKEND", "auto"), \
             mock.patch.object(
                 self.gesture, "_create_classic_detector", return_value=Detector()
             ), \
             mock.patch.object(self.gesture, "_open_camera", return_value=Camera()):
            self.gesture.detect_gesture(
                callback_fn=callback,
                capture_check=capture_check,
                max_frames=10,
            )

        self.assertEqual(received, ["OBJECT_DETECT"])

    def test_main_connects_every_advertised_action(self):
        main = importlib.import_module("main")
        self.assertEqual(
            set(main._GESTURE_ACTIONS),
            {"MODE_SCAN", "MODE_VOICE", "OBJECT_DETECT", "GPS_CHECK"},
        )


if __name__ == "__main__":
    unittest.main()
