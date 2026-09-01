import builtins
import importlib
import os
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT
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
            ask_ai=lambda question, **kwargs: asked.append(question) or "answer"
        )

        with mock.patch.object(main, "_speak", side_effect=lambda t, **k: spoken.append(t)):
            with mock.patch.object(main, "_keyboard_input", return_value=".-"):
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

        # translate() now short-circuits on the cache and skips every network
        # provider when the device is offline, so both have to be neutralised
        # or this asserts nothing about the Google path.
        with mock.patch.object(translator, "is_online", return_value=True), \
                mock.patch.object(translator, "_cache_get", return_value=None), \
                mock.patch.object(translator, "_cache_put"), \
                mock.patch.object(translator.requests, "get", return_value=response) as get:
            result = translator.translate("hello", "en", "hi")

        self.assertEqual(result, "नमस्ते")
        get.assert_called_once()

    def test_translate_falls_back_offline_instead_of_stalling(self):
        """No network must mean a fast, honest answer — not a 30s DNS stall."""
        from modules import translator

        def unreachable(*_a, **_k):
            raise AssertionError("network provider called while offline")

        with mock.patch.object(translator, "is_online", return_value=False), \
                mock.patch.object(translator, "_cache_get", return_value=None), \
                mock.patch.object(translator.requests, "get", unreachable), \
                mock.patch.object(translator.requests, "post", unreachable):
            hit = translator.translate_ex("thank you", "en", "gu")
            miss = translator.translate_ex("an unlisted sentence", "en", "gu")

        # Phrasebook entry: still translated, offline.
        self.assertTrue(hit.translated)
        self.assertEqual(hit.source, "phrasebook")
        self.assertEqual(hit.lang, "gu")

        # No entry: reports failure rather than returning an error string as
        # though it were a translation, and tags the text as still-English so
        # the caller speaks it with the English voice.
        self.assertFalse(miss.translated)
        self.assertEqual(miss.text, "an unlisted sentence")
        self.assertEqual(miss.lang, "en")

    def test_detect_language_works_without_network(self):
        from modules import translator

        def unreachable(*_a, **_k):
            raise AssertionError("network used for script detection")

        with mock.patch.object(translator.requests, "get", unreachable):
            self.assertEqual(translator.detect_language("hello there"), "en")
            self.assertEqual(translator.detect_language("प्रकाश संश्लेषण"), "hi")
            self.assertEqual(translator.detect_language("પ્રકાશસંશ્લેષણ"), "gu")

    def test_ocr_auto_falls_back_to_tesseract_not_surya(self):
        import numpy as np
        from modules import ocr

        original_config = dict(ocr._config)
        try:
            ocr._config.update({
                "ocr_engine": "auto",
                "gemini_api_key": "configured",
                "ocr_surya_fallback": False,
            })

            with mock.patch.object(ocr._camera, "open", return_value=True):
                with mock.patch.object(
                    ocr._camera,
                    "capture",
                    return_value=np.zeros((20, 20, 3), dtype=np.uint8),
                ):
                    with mock.patch.object(
                        ocr._gemini_engine,
                        "extract_text",
                        side_effect=RuntimeError("network down"),
                    ):
                        with mock.patch.object(
                            ocr._tesseract_engine,
                            "extract_text",
                            return_value="fast local text",
                        ):
                            with mock.patch.object(ocr._engine, "extract_text") as surya:
                                self.assertEqual(ocr.scan_and_read(), "fast local text")
                                surya.assert_not_called()
        finally:
            ocr._config.clear()
            ocr._config.update(original_config)

    def test_ocr_auto_uses_gemini_first_when_configured(self):
        import numpy as np
        from modules import ocr

        original_config = dict(ocr._config)
        try:
            ocr._config.update({
                "ocr_engine": "auto",
                "gemini_api_key": "configured",
                "ocr_gemini_first": True,
                "ocr_surya_fallback": False,
            })

            with mock.patch.object(ocr._camera, "open", return_value=True):
                with mock.patch.object(
                    ocr._camera,
                    "capture",
                    return_value=np.zeros((20, 20, 3), dtype=np.uint8),
                ):
                    with mock.patch.object(
                        ocr._gemini_engine,
                        "extract_text",
                        return_value="high accuracy cloud text",
                    ):
                        with mock.patch.object(ocr._tesseract_engine, "extract_text") as tess:
                            with mock.patch.object(ocr._engine, "extract_text") as surya:
                                self.assertEqual(
                                    ocr.scan_and_read(),
                                    "high accuracy cloud text",
                                )
                                tess.assert_not_called()
                                surya.assert_not_called()
        finally:
            ocr._config.clear()
            ocr._config.update(original_config)

    def test_tesseract_candidate_keeps_best_confidence_text(self):
        from modules import ocr

        weak = {
            "text": ["", "H3llo", "w0rld"],
            "conf": ["-1", "20", "22"],
            "block_num": [0, 1, 1],
            "par_num": [0, 1, 1],
            "line_num": [0, 1, 1],
        }
        strong = {
            "text": ["", "Hello", "world"],
            "conf": ["-1", "91", "89"],
            "block_num": [0, 1, 1],
            "par_num": [0, 1, 1],
            "line_num": [0, 1, 1],
        }

        weak_candidate = ocr.TesseractOCREngine._candidate_from_data(weak, "full-adaptive", 6)
        strong_candidate = ocr.TesseractOCREngine._candidate_from_data(strong, "full-enhanced", 4)

        self.assertGreater(strong_candidate.score, weak_candidate.score)
        self.assertEqual(strong_candidate.text, "Hello world")

    def test_ocr_does_not_use_surya_after_empty_tesseract_by_default(self):
        import numpy as np
        from modules import ocr

        original_config = dict(ocr._config)
        try:
            ocr._config.update({
                "ocr_engine": "tesseract",
                "gemini_api_key": "",
                "ocr_surya_fallback": False,
            })

            with mock.patch.object(ocr._camera, "open", return_value=True):
                with mock.patch.object(
                    ocr._camera,
                    "capture",
                    return_value=np.zeros((20, 20, 3), dtype=np.uint8),
                ):
                    with mock.patch.object(
                        ocr._tesseract_engine,
                        "extract_text",
                        return_value="No text detected in the image.",
                    ):
                        with mock.patch.object(ocr._engine, "extract_text") as surya:
                            self.assertEqual(
                                ocr.scan_and_read(),
                                "No text detected in the image.",
                            )
                            surya.assert_not_called()
        finally:
            ocr._config.clear()
            ocr._config.update(original_config)

    def test_ocr_auto_does_not_use_surya_when_tesseract_unavailable_by_default(self):
        import numpy as np
        from modules import ocr

        original_config = dict(ocr._config)
        try:
            ocr._config.update({
                "ocr_engine": "auto",
                "gemini_api_key": "",
                "ocr_surya_fallback": False,
            })

            with mock.patch.object(ocr._camera, "open", return_value=True):
                with mock.patch.object(
                    ocr._camera,
                    "capture",
                    return_value=np.zeros((20, 20, 3), dtype=np.uint8),
                ):
                    with mock.patch.object(
                        ocr._tesseract_engine,
                        "extract_text",
                        return_value="OCR unavailable: missing tesseract",
                    ):
                        with mock.patch.object(ocr._engine, "extract_text") as surya:
                            self.assertEqual(
                                ocr.scan_and_read(),
                                "OCR unavailable: missing tesseract",
                            )
                            surya.assert_not_called()
        finally:
            ocr._config.clear()
            ocr._config.update(original_config)


class RepairedDefectTests(unittest.TestCase):
    """Regressions for defects that shipped silently — each of these
    reproduced a real failure on the device before the fix."""

    # ── TTS: the device was completely silent ────────────────────────────
    def test_tts_never_swallows_speech_when_no_audio_backend(self):
        """Audio was synthesised and then dropped when pygame was missing:
        no sound, no error, no console fallback."""
        from modules import tts

        manager = tts.TTSManager.__new__(tts.TTSManager)
        manager.running = True
        manager._speaking = __import__("threading").Event()
        manager._engine_available = False
        manager._engine = None

        with mock.patch.object(manager, "_synthesize", return_value=(b"\x00\x01", "wav")), \
             mock.patch.object(manager, "_play_pygame", return_value=False), \
             mock.patch.object(manager, "_play_system", return_value=False), \
             mock.patch.object(manager, "_play_pyttsx3_direct", return_value=False), \
             mock.patch("builtins.print") as printed:
            manager._deliver("important message", "eng")

        self.assertTrue(
            any("important message" in str(c) for c in printed.call_args_list),
            "text must still reach the user when every audio backend fails",
        )

    def test_tts_module_exposes_shutdown_used_by_main(self):
        """main.py called _modules['tts'].tts_manager.shutdown(); that
        attribute never existed, so TTS was never shut down cleanly."""
        from modules import tts

        self.assertTrue(hasattr(tts, "shutdown"))
        self.assertTrue(hasattr(tts, "flush"))
        self.assertTrue(hasattr(tts, "wait_until_idle"))

    def test_tts_mp3_uses_music_channel_not_sound(self):
        """gTTS returns MP3; pygame.mixer.Sound cannot decode MP3 buffers."""
        from modules import tts

        if not tts.PYGAME_AVAILABLE:
            self.skipTest("pygame not installed")
        manager = tts._get_manager()
        with mock.patch.object(tts, "pygame") as pg:
            manager._play_pygame(b"fake-mp3", "mp3")
            pg.mixer.music.load.assert_called_once()
            pg.mixer.Sound.assert_not_called()

    # ── Morse: word completion never fired ───────────────────────────────
    def test_morse_emits_word_event_after_word_gap(self):
        import modules.morse as morse

        decoder = morse.MorseDecoder()
        try:
            decoder.add_dot()
            decoder.add_dash()          # ".-" == A
            events = []
            deadline = time.time() + (morse.WORD_GAP_MS / 1000.0) + 2.0
            while time.time() < deadline:
                out = decoder.get_output(timeout=0.2)
                if out:
                    events.append(out)
                if any(e[0] == "WORD" for e in events):
                    break
            self.assertIn(("LETTER", "A"), events)
            self.assertTrue(any(e[0] == "WORD" for e in events),
                            "WORD_GAP branch was unreachable before the fix")
        finally:
            decoder.shutdown()

    # ── Embedder: batch cache-warm was a no-op ───────────────────────────
    def test_embedder_batch_populates_cache(self):
        import numpy as np
        from services.embedder import Embedder

        emb = Embedder.__new__(Embedder)
        emb.use_gemini = False
        emb.local_model = mock.Mock()
        emb.local_model.encode.return_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        emb._model_lock = __import__("threading").Lock()
        emb._cache = __import__("collections").OrderedDict()
        emb._cache_size = 16
        emb._cache_lock = __import__("threading").Lock()
        emb._cache_hits = emb._cache_misses = 0
        emb._init_local_model = lambda: None

        emb.get_embeddings(["alpha", "beta"])
        self.assertEqual(emb.cache_info["currsize"], 2,
                         "batch results must land in the cache, not be recomputed")

        emb._compute_embedding = mock.Mock(side_effect=AssertionError("recomputed a cached text"))
        np.testing.assert_allclose(emb.get_embedding("alpha"), [1.0, 2.0])

    def test_embedder_cache_is_bounded_and_per_instance(self):
        import numpy as np
        from services.embedder import Embedder

        def make():
            e = Embedder.__new__(Embedder)
            e._cache = __import__("collections").OrderedDict()
            e._cache_size = 2
            e._cache_lock = __import__("threading").Lock()
            e._cache_hits = e._cache_misses = 0
            e._compute_embedding = lambda t: np.array([len(t)], dtype=np.float32)
            return e

        a, b = make(), make()
        for key in ("x", "y", "z"):
            a._cached_embedding(key)
        self.assertEqual(a.cache_info["currsize"], 2, "cache must evict, not grow forever")
        self.assertEqual(b.cache_info["currsize"], 0, "caches must not be shared between instances")

    # ── RAG: duplicate index_file killed diagram descriptions ────────────
    def test_rag_pipeline_describes_markdown_images(self):
        from services.rag_pipeline import RAGPipeline

        agent = mock.Mock()
        agent.describe_image.return_value = "A labelled diagram of the water cycle."
        pipeline = RAGPipeline(
            embedder=mock.Mock(), vector_store=mock.Mock(),
            retriever=mock.Mock(), gemini_agent=agent,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            try:
                from PIL import Image
                Image.new("RGB", (8, 8)).save(root / "fig1.png")
            except Exception:
                self.skipTest("Pillow unavailable")
            (root / "chapter.md").write_text("Intro\n\n![Water cycle](fig1.png)\n\nEnd")

            with mock.patch.object(pipeline, "index_document") as indexed:
                self.assertTrue(pipeline.index_file(str(root / "chapter.md")))

            content = indexed.call_args[0][0]
            self.assertIn("water cycle", content.lower())
            agent.describe_image.assert_called_once()

    def test_rag_pipeline_ignores_remote_and_escaping_image_paths(self):
        from services.rag_pipeline import RAGPipeline

        agent = mock.Mock()
        pipeline = RAGPipeline(
            embedder=mock.Mock(), vector_store=mock.Mock(),
            retriever=mock.Mock(), gemini_agent=agent,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            md = root / "chapter.md"
            md.write_text("![a](https://evil.example/x.png)\n![b](../../../etc/passwd)")
            pipeline._describe_markdown_images(md, md.read_text())
            agent.describe_image.assert_not_called()

    # ── main.py: blocking input() on a headless device ───────────────────
    def test_keyboard_input_returns_default_without_tty(self):
        main = importlib.import_module("main")

        with mock.patch.object(main, "_STDIN_IS_TTY", False):
            with mock.patch.object(builtins, "input",
                                   side_effect=AssertionError("input() must not be called headless")):
                self.assertEqual(main._keyboard_input("prompt: ", default="fallback"), "fallback")

    def test_shutdown_uses_real_tts_api(self):
        main = importlib.import_module("main")
        tts_stub = mock.Mock()
        main._shutdown_done = False
        try:
            with mock.patch.dict(main._modules, {"tts": tts_stub}):
                main.shutdown()
            tts_stub.wait_until_idle.assert_called_once()
            tts_stub.shutdown.assert_called_once()
        finally:
            main._shutdown_done = False

    # ── Object detection: no way out of Mode 6 ───────────────────────────
    def test_object_detection_accepts_stop_event(self):
        import inspect
        from modules import object_detection

        for fn in (object_detection.stream_detect, object_detection.run_detection):
            self.assertIn("stop_event", inspect.signature(fn).parameters,
                          f"{fn.__name__} needs a stop signal — Mode 6 trapped the device")

    def test_object_detection_stops_when_callback_returns_false(self):
        import numpy as np
        from modules import object_detection

        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, frame)

        result = mock.Mock()
        result.boxes = []
        model = mock.Mock()
        model.predict.return_value = [result]
        model.names = {}

        calls = []
        with mock.patch.object(object_detection, "_get_model", return_value=model), \
             mock.patch.object(object_detection.cv2, "VideoCapture", return_value=cap):
            object_detection.stream_detect(
                callback=lambda text, dets: calls.append(text) and False or False
            )
        self.assertEqual(len(calls), 1, "loop must exit on the first False from the callback")

    # ── ai_query: timeouts that did not time out ─────────────────────────
    def test_ask_ai_does_not_create_a_pool_per_call(self):
        from modules import ai_query

        self.assertIsInstance(ai_query._EXECUTOR,
                              __import__("concurrent.futures", fromlist=["x"]).ThreadPoolExecutor)

    def test_offline_fallback_returns_fast_when_unconfigured(self):
        from modules import ai_query

        with mock.patch.dict(ai_query._settings, {"offline_model_path": ""}, clear=False):
            start = time.time()
            self.assertIsNone(ai_query._ask_offline("hello"))
        self.assertLess(time.time() - start, 1.0,
                        "unconfigured offline model used to block for 60s")

    def test_latency_narrator_flushes_stale_messages(self):
        from modules.ai_query import LatencyNarrator

        spoken, flushed = [], []
        narrator = LatencyNarrator(
            speak_fn=spoken.append, flush_fn=lambda: flushed.append(True),
            first_delay=0.01, interval=0.01,
        )
        narrator.start()
        time.sleep(0.2)
        narrator.stop()

        self.assertTrue(spoken, "narrator should have spoken")
        self.assertTrue(flushed, "queued reassurance must be flushed before the answer")

    def test_narrator_does_not_flush_when_it_never_spoke(self):
        from modules.ai_query import LatencyNarrator

        flushed = []
        narrator = LatencyNarrator(
            speak_fn=lambda m: None, flush_fn=lambda: flushed.append(True),
            first_delay=30.0,
        )
        narrator.start()
        narrator.stop()
        self.assertEqual(flushed, [], "a fast answer must not have its speech flushed")

    # ── confidential_mode: unimportable without pyserial ─────────────────
    def test_confidential_mode_imports_without_pyserial(self):
        with mock.patch.dict(sys.modules, {"serial": None}):
            importlib.reload(importlib.import_module("modules.confidential_mode"))

    # ── voice: hands-free capture ────────────────────────────────────────
    def _run_listen_headless(self, tmpdir):
        """Drive voice.listen() with no TTY and no real microphone, returning
        the argv arecord was invoked with."""
        from modules import voice

        captured = {}

        class FakeProc:
            def __init__(self, cmd, **kw):
                captured["cmd"] = cmd

            def poll(self):
                return 0

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

        class FakeWave:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def getnframes(self):
                return 16000

            def readframes(self, n):
                return b"\x00\x01" * n

        fake_stdin = mock.Mock()
        fake_stdin.isatty.return_value = False   # headless: no terminal

        with mock.patch.object(voice, "VAD_AVAILABLE", False), \
             mock.patch.object(voice, "RECORDING_DIR", tmpdir), \
             mock.patch.object(voice.shutil, "which", return_value="/usr/bin/arecord"), \
             mock.patch("subprocess.Popen", FakeProc), \
             mock.patch.object(voice.wave, "open", return_value=FakeWave()), \
             mock.patch.object(voice, "_multi_engine_transcribe", return_value="hello there"), \
             mock.patch.object(voice.sys, "stdin", fake_stdin), \
             mock.patch.object(builtins, "input",
                               side_effect=AssertionError("listen() must not require a keypress")):
            result = voice.listen("en-IN")

        return result, captured["cmd"]

    def test_voice_listen_is_hands_free(self):
        """listen() used to call input('press ENTER to stop') and could not
        return without a keypress — fatal on a keyboard-less device."""
        with tempfile.TemporaryDirectory() as tmp:
            result, cmd = self._run_listen_headless(tmp)

        self.assertEqual(result, "hello there")
        # The recording must be self-terminating (-d <seconds>).
        self.assertIn("-d", cmd, "recording must have its own duration limit")

    def test_voice_recording_target_is_configurable(self):
        """Device and output path were hardcoded to one developer's machine
        ('plughw:2,0' and the /mnt/aet_usb mount)."""
        from modules import voice

        with tempfile.TemporaryDirectory() as tmp:
            _, cmd = self._run_listen_headless(tmp)

        self.assertIn(voice.MIC_DEVICE, cmd)
        self.assertTrue(any(str(tmp) in str(part) for part in cmd),
                        "recordings must go to the configured directory")

    def test_vad_capture_uses_configured_mic_index(self):
        import inspect
        from modules import voice

        source = inspect.getsource(voice.listen_with_vad)
        self.assertIn("input_device_index=MIC_INDEX", source)

    def test_bandpass_filter_preserves_length_and_silence(self):
        import numpy as np
        from modules import voice

        raw = (np.random.randn(16000) * 2000).astype("<i2").tobytes()
        out = voice._bandpass_filter(raw, 16000)
        self.assertEqual(len(out), len(raw))

        silence = np.zeros(16000, dtype="<i2").tobytes()
        self.assertEqual(
            np.abs(np.frombuffer(voice._bandpass_filter(silence, 16000), dtype="<i2")).max(), 0
        )

    def test_prune_recordings_bounds_storage(self):
        from modules import voice

        with tempfile.TemporaryDirectory() as tmp:
            for i in range(10):
                p = Path(tmp) / f"voice_{i}.wav"
                p.write_bytes(b"x")
                os.utime(p, (i, i))
            voice._prune_recordings(tmp, keep=3)
            self.assertEqual(len(list(Path(tmp).glob("voice_*.wav"))), 3)


if __name__ == "__main__":
    unittest.main()
