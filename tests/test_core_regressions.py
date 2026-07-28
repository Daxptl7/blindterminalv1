import builtins
import importlib
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "Blindterminal"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


class CoreRegressionTests(unittest.TestCase):
    def _hand(self, gesture):
        @dataclass
        class Point:
            x: float = 0.5
            y: float = 0.5
            z: float = 0.0

        lm = [Point() for _ in range(21)]

        def set_point(index, x, y):
            lm[index] = Point(x, y, 0.0)

        set_point(0, 0.5, 0.9)
        set_point(2, 0.4, 0.55)
        set_point(3, 0.4, 0.6)
        set_point(4, 0.4, 0.65)
        set_point(5, 0.5, 0.55)
        set_point(6, 0.5, 0.5)
        set_point(8, 0.5, 0.65)
        set_point(10, 0.55, 0.5)
        set_point(12, 0.55, 0.65)
        set_point(14, 0.6, 0.5)
        set_point(16, 0.6, 0.65)
        set_point(18, 0.65, 0.5)
        set_point(20, 0.65, 0.65)

        if gesture == "open_palm":
            set_point(3, 0.34, 0.45)
            set_point(4, 0.22, 0.35)
            for tip, pip, x in ((8, 6, 0.48), (12, 10, 0.54), (16, 14, 0.6), (20, 18, 0.66)):
                set_point(pip, x, 0.45)
                set_point(tip, x, 0.2)
        elif gesture == "thumbs_up":
            set_point(2, 0.4, 0.55)
            set_point(3, 0.4, 0.4)
            set_point(4, 0.4, 0.22)
        elif gesture == "two_fingers":
            set_point(6, 0.48, 0.45)
            set_point(8, 0.48, 0.2)
            set_point(10, 0.55, 0.45)
            set_point(12, 0.55, 0.2)
        elif gesture == "point_down":
            set_point(5, 0.5, 0.45)
            set_point(6, 0.5, 0.55)
            set_point(8, 0.5, 0.75)

        return lm

    def test_gesture_classifier_maps_core_commands(self):
        gesture = importlib.import_module("modules.gesture_control")

        self.assertEqual(gesture.classify_gesture(self._hand("open_palm")), "MODE_SCAN")
        self.assertEqual(gesture.classify_gesture(self._hand("thumbs_up")), "CONFIRM")
        self.assertEqual(gesture.classify_gesture(self._hand("two_fingers")), "MODE_VOICE")
        self.assertEqual(gesture.classify_gesture(self._hand("fist")), "STOP")
        self.assertEqual(gesture.classify_gesture(self._hand("point_down")), "REPEAT")

    def test_gesture_stop_callback_can_end_detection_loop(self):
        gesture = importlib.import_module("modules.gesture_control")

        self.assertFalse(gesture._callback_allows_continue(lambda name: False, "STOP"))

    def test_morse_mode_decodes_without_name_error(self):
        main = importlib.import_module("main")

        spoken = []
        asked = []

        main._modules["morse"] = importlib.import_module("modules.morse")
        main._modules["ai_query"] = mock.Mock(
            ask_ai=lambda question: asked.append(question) or "answer"
        )

        with mock.patch.object(main, "_speak", side_effect=spoken.append):
            with mock.patch.object(builtins, "input", return_value=".-"):
                main.mode_morse_type()

        self.assertEqual(asked, ["A"])
        self.assertIn("You typed: A", spoken)

    def test_vector_store_ignores_untrusted_legacy_pickle(self):
        from services.vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "metadata.pkl").write_bytes(b"not a trusted pickle")

            store = VectorStore(store_path=str(path), dimension=3)

            self.assertEqual(store.metadata, [])
            self.assertFalse((path / "metadata.json").exists())

    def test_vector_store_dimension_change_drops_incompatible_vectors(self):
        import numpy as np
        from services.vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmp:
            store = VectorStore(store_path=tmp, dimension=3)
            store.add_text("old local embedding", np.array([1.0, 0.0, 0.0]))
            store.add_text("new gemini embedding", np.array([1.0, 0.0, 0.0, 0.0]))

            self.assertEqual(store.dimension, 4)
            self.assertEqual(store.metadata, ["new gemini embedding"])
            self.assertEqual(store.retrieve(np.array([1.0, 0.0, 0.0])), [])

    def test_rag_index_file_rejects_binary_suffix(self):
        from services.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(
            embedder=mock.Mock(),
            vector_store=mock.Mock(),
            retriever=mock.Mock(),
            gemini_agent=mock.Mock(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "secret.bin"
            binary.write_bytes(b"secret")

            self.assertFalse(pipeline.index_file(str(binary)))

    def test_translate_uses_google_web_response(self):
        from modules import translator

        response = mock.Mock()
        response.json.return_value = [[["नमस्ते", "hello", None, None]], None, "en"]
        response.raise_for_status.return_value = None

        with mock.patch.object(translator.requests, "get", return_value=response) as get:
            result = translator.translate("hello", "en", "hi")

        self.assertEqual(result, "नमस्ते")
        get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
