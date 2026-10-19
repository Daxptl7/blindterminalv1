"""Online/local fallback is independent of per-result audio routing."""

import importlib
import threading
import unittest
from unittest import mock

from modules import ai_query, confidential_mode as output, ocr, translator, tts, voice
from modules.privacy_policy import PUBLIC, current_policy, processing_session


class AutomaticFallbackTests(unittest.TestCase):
    def test_ai_provider_errors_fall_through_to_local_answer(self):
        events = []
        def failed(name):
            def call(*args, **kwargs):
                events.append(name)
                raise ConnectionError("unreachable")
            return call
        def local(*args, **kwargs):
            events.append("local")
            return "Local answer"
        with processing_session(PUBLIC), \
                mock.patch.object(ai_query, "get_rag_pipeline", return_value=None), \
                mock.patch.object(ai_query, "_ask_groq", side_effect=failed("groq")), \
                mock.patch.object(ai_query, "_ask_openai", side_effect=failed("openai")), \
                mock.patch.object(ai_query, "_ask_gemini", side_effect=failed("gemini")), \
                mock.patch.object(ai_query, "_ask_offline", side_effect=local):
            self.assertEqual(ai_query.ask_ai("Question"), "Local answer")
        self.assertEqual(events, ["groq", "openai", "gemini", "local"])

    def test_successful_online_ai_does_not_run_offline_model(self):
        with processing_session(PUBLIC), \
                mock.patch.object(ai_query, "get_rag_pipeline", return_value=None), \
                mock.patch.object(ai_query, "_ask_groq", return_value="Online answer"), \
                mock.patch.object(ai_query, "_ask_offline") as local:
            self.assertEqual(ai_query.ask_ai("Question"), "Online answer")
        local.assert_not_called()

    def test_translation_upgrades_existing_local_first_settings_and_falls_back(self):
        for online, succeeds, expected in ((True, True, ["online"]),
                                           (True, False, ["online", "local"]),
                                           (False, False, ["local"])):
            with self.subTest(online=online, succeeds=succeeds):
                events = []
                def cloud(*args):
                    events.append("online")
                    if not succeeds:
                        raise ConnectionError("offline")
                    return "नमस्ते"
                def local(*args):
                    events.append("local")
                    return "नमस्ते"
                providers = {"indictrans2": (local, False), "google": (cloud, True)}
                with processing_session(PUBLIC), \
                        mock.patch.object(translator, "PROVIDER_ORDER", ("indictrans2", "google")), \
                        mock.patch.object(translator, "_PROVIDER_MAP", providers), \
                        mock.patch.object(translator, "_provider_blocked_until", {}), \
                        mock.patch.object(translator, "_provider_failures", {}), \
                        mock.patch.object(translator, "_cache_get_record", return_value=None), \
                        mock.patch.object(translator, "_cache_put"), \
                        mock.patch.object(translator, "_provider_model_id", return_value="mock"), \
                        mock.patch.object(translator, "is_online", return_value=online):
                    result = translator.translate_ex("hello", "en", "hi")
                self.assertTrue(result.translated)
                self.assertEqual(events, expected)

    def test_recognition_tries_online_then_local_despite_old_config_order(self):
        events = []
        def cloud(*args):
            events.append("online")
            raise ConnectionError("unreachable")
        def local(*args):
            events.append("local")
            return "hello"
        with processing_session(PUBLIC), \
                mock.patch.object(voice, "STT_ENGINES", ["vosk", "google", "sphinx"]), \
                mock.patch.object(voice, "_engine_available", return_value=True), \
                mock.patch.object(voice, "_transcribe_google", side_effect=cloud), \
                mock.patch.object(voice, "_transcribe_vosk", side_effect=local):
            self.assertEqual(voice._run_engines(b"pcm", "en-IN"), "hello")
        self.assertEqual(events, ["online", "local"])

    def test_online_ocr_failure_automatically_uses_local_reader(self):
        import numpy as np
        events = []
        def cloud(*args, **kwargs):
            events.append("online")
            raise ConnectionError("unreachable")
        def local(*args, **kwargs):
            events.append("local")
            return "Local text from the page."
        with processing_session(PUBLIC), \
                mock.patch.dict(ocr._config, {"ocr_engine": "auto", "ocr_gemini_first": True,
                                              "ocr_guidance_enabled": False}), \
                mock.patch.object(ocr._camera, "open", return_value=True), \
                mock.patch.object(ocr._camera, "capture", return_value=np.full((100, 100, 3), 200, np.uint8)), \
                mock.patch.object(ocr, "_gemini_key", return_value="test"), \
                mock.patch.object(ocr._gemini_engine, "extract_text", side_effect=cloud), \
                mock.patch.object(ocr._tesseract_engine, "extract_text", side_effect=local):
            self.assertEqual(ocr.scan_and_read(), "Local text from the page.")
        self.assertEqual(events, ["online", "local"])


class ResultOutputTests(unittest.TestCase):
    def test_earphone_playback_keeps_normal_synthesis_policy(self):
        manager = tts.TTSManager.__new__(tts.TTSManager)
        manager._speaking = threading.Event()
        policies = []
        def synthesize(*args):
            policies.append(current_policy())
            return b"audio", "wav"
        with processing_session(PUBLIC), \
                mock.patch.object(tts, "require_private_output", return_value=tts.EARPHONE_DEVICE), \
                mock.patch.object(manager, "_synthesize", side_effect=synthesize), \
                mock.patch.object(manager, "_play_routed", return_value=True):
            self.assertTrue(manager._deliver("Result", "en", tts.EARPHONE_DEVICE))
        self.assertEqual(policies, [PUBLIC])

    def test_earphone_mp3_failure_retries_offline_wav_on_same_output(self):
        manager = tts.TTSManager.__new__(tts.TTSManager)
        manager._speaking = threading.Event()
        with processing_session(PUBLIC), \
                mock.patch.object(tts, "require_private_output", return_value=tts.EARPHONE_DEVICE), \
                mock.patch.object(manager, "_synthesize", return_value=(b"online", "mp3")), \
                mock.patch.object(manager, "_synthesize_offline_wav", return_value=b"local"), \
                mock.patch.object(manager, "_prepare_audio", return_value=(b"local", "wav")), \
                mock.patch.object(manager, "_play_routed", side_effect=[False, True]) as routed, \
                mock.patch.object(manager, "_play_pygame") as public:
            self.assertTrue(manager._deliver("Result", "en", tts.EARPHONE_DEVICE))
        self.assertEqual(routed.call_args_list, [
            mock.call(b"online", "mp3", tts.EARPHONE_DEVICE),
            mock.call(b"local", "wav", tts.EARPHONE_DEVICE),
        ])
        public.assert_not_called()

    def test_earphone_result_does_not_change_global_output_or_processing(self):
        with processing_session(PUBLIC), \
                mock.patch.object(output, "ask_confidentiality", return_value="PRIVATE"), \
                mock.patch.object(tts, "require_private_output", return_value="earphone"), \
                mock.patch.object(tts, "_current_device", "speaker"), \
                mock.patch.object(tts, "speak_on_device") as routed, \
                mock.patch.object(output, "enable_earphone") as switch:
            output.speak_with_privacy_check("One result.", "eng")
            self.assertEqual(tts._current_device, "speaker")
            self.assertEqual(current_policy(), PUBLIC)
        routed.assert_called_once_with("One result.", "earphone", lang="eng", block=True, stop_check=None)
        switch.assert_not_called()

    def test_output_choice_is_asked_fresh_for_each_result_in_one_session(self):
        with processing_session(PUBLIC), \
                mock.patch.object(output, "ask_confidentiality", side_effect=["PRIVATE", "NORMAL"]) as choose, \
                mock.patch.object(tts, "require_private_output", return_value="earphone"), \
                mock.patch.object(tts, "speak_on_device") as routed:
            output.speak_with_privacy_check("First answer.", "eng")
            output.speak_with_privacy_check("Next answer.", "eng")
        self.assertEqual(choose.call_count, 2)
        self.assertEqual([call.args[1] for call in routed.call_args_list], ["earphone", tts.SPEAKER_DEVICE])

    def test_long_result_stays_on_selected_output_until_its_last_chunk(self):
        result = " ".join(f"word{number}" for number in range(200))
        with mock.patch.object(output, "ask_confidentiality", return_value="PRIVATE") as choose, \
                mock.patch.object(tts, "require_private_output", return_value="earphone"), \
                mock.patch.object(tts, "speak_on_device") as routed:
            output.speak_document_with_privacy_check(result, "eng", chunk_chars=90)
        choose.assert_called_once()
        self.assertGreater(routed.call_count, 1)
        self.assertEqual(" ".join(call.args[0] for call in routed.call_args_list), result)
        self.assertTrue(all(call.args[1] == "earphone" for call in routed.call_args_list))

    def test_unavailable_earphone_does_not_fall_back_to_reading_result_on_speaker(self):
        main = importlib.import_module("main")
        with mock.patch.dict(main._modules, {"privacy": output}), \
                mock.patch.object(output, "ask_confidentiality", return_value="PRIVATE"), \
                mock.patch.object(tts, "require_private_output", side_effect=tts.AudioRoutingError("missing")), \
                mock.patch.object(main, "_speak") as speak:
            self.assertIsNone(main._deliver_result("RESULT_SENTINEL"))
        self.assertFalse(any("RESULT_SENTINEL" in call.args[0] for call in speak.call_args_list))

    def test_mode_processes_before_asking_output_and_next_prompt_uses_speaker(self):
        main = importlib.import_module("main")
        events = []
        fake_tts = mock.Mock()
        fake_tts.speak.side_effect = lambda text, **kwargs: events.append((text, "speaker"))
        fake_tts.speak_on_device.side_effect = lambda text, device, **kwargs: events.append((text, device))
        def online(*args, **kwargs):
            events.append(("online", None))
            return None
        def local(*args, **kwargs):
            events.append(("offline", None))
            return "RESULT_SENTINEL"
        def choose(*args):
            events.append(("choose_output", None))
            return "PRIVATE"
        with mock.patch.dict(main._modules, {"privacy": output, "ai_query": ai_query,
                                            "tts": fake_tts, "voice": mock.Mock()}), \
                mock.patch.dict(main._settings, {"enabled_modes": ["2"]}), \
                mock.patch.object(main, "_morse_serial_singleton", None), \
                mock.patch.object(main, "_morse_type_sentence", return_value="Question"), \
                mock.patch.object(ai_query, "get_rag_pipeline", return_value=None), \
                mock.patch.object(ai_query, "_ask_groq", side_effect=online), \
                mock.patch.object(ai_query, "_ask_openai", return_value=None), \
                mock.patch.object(ai_query, "_ask_gemini", return_value=None), \
                mock.patch.object(ai_query, "_ask_offline", side_effect=local), \
                mock.patch.object(output, "ask_confidentiality", side_effect=choose), \
                mock.patch.object(tts, "require_private_output", return_value="earphone"):
            main.run_mode("2")
            main._speak("Next menu prompt")
        self.assertLess(events.index(("online", None)), events.index(("offline", None)))
        self.assertLess(events.index(("offline", None)), events.index(("choose_output", None)))
        self.assertLess(events.index(("choose_output", None)), events.index(("RESULT_SENTINEL", "earphone")))
        self.assertEqual(events[-1], ("Next menu prompt", "speaker"))


if __name__ == "__main__":
    unittest.main()
