"""Product-safety tests for multilingual translation and speech routing."""

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class TranslationSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.translator = importlib.import_module("modules.translator")

    def setUp(self):
        self.translator._provider_failures.clear()
        self.translator._provider_blocked_until.clear()

    def test_phrasebook_supports_all_six_directions_offline(self):
        cases = (
            ("hello", "en", "hi", "नमस्ते"),
            ("hello", "en", "gu", "નમસ્તે"),
            ("नमस्ते", "hi", "en", "hello"),
            ("નમસ્તે", "gu", "en", "hello"),
            ("नमस्ते", "hi", "gu", "નમસ્તે"),
            ("નમસ્તે", "gu", "hi", "नमस्ते"),
        )
        with mock.patch.object(self.translator, "_ordered_providers", return_value=()):
            for source, src, dest, expected in cases:
                with self.subTest(pair=f"{src}->{dest}"):
                    result = self.translator.translate_ex(
                        source, src, dest, privacy=True
                    )
                    self.assertTrue(result.translated)
                    self.assertEqual(result.text, expected)
                    self.assertEqual(result.lang, dest)

    def test_validation_rejects_wrong_script_unchanged_and_lost_numbers(self):
        passed, report = self.translator.validate_translation(
            "The dose is 25 mg on 21/02/2023", "The dose is 20 mg", "en", "hi"
        )
        self.assertFalse(passed)
        reasons = " ".join(report["reasons"])
        self.assertIn("wrong target script", reasons)
        self.assertIn("missing protected tokens", reasons)

        passed, report = self.translator.validate_translation(
            "This is a complete sentence", "This is a complete sentence", "en", "gu"
        )
        self.assertFalse(passed)
        self.assertIn("output is unchanged", report["reasons"])

    def test_contraction_suffix_is_not_treated_as_measurement_unit(self):
        passed, report = self.translator.validate_translation(
            "I'm studying and I'm working",
            "मैं पढ़ाई कर रहा हूँ और काम कर रहा हूँ",
            "en",
            "hi",
        )

        self.assertTrue(passed, report["reasons"])
        self.assertTrue(report["protected_tokens_preserved"])

    def test_private_request_never_reads_or_writes_cache_or_calls_cloud(self):
        cloud = mock.Mock(return_value="नमस्ते")
        providers = (("google", cloud, True),)
        with mock.patch.object(self.translator, "_ordered_providers", return_value=providers), \
             mock.patch.object(self.translator, "_cache_get_record") as cache_get, \
             mock.patch.object(self.translator, "_cache_put") as cache_put, \
             mock.patch.object(self.translator, "is_online") as online:
            result = self.translator.translate_ex("hello", "en", "hi", privacy=True)

        self.assertTrue(result.translated)
        self.assertEqual(result.source, "phrasebook")
        cloud.assert_not_called()
        online.assert_not_called()
        cache_get.assert_not_called()
        cache_put.assert_not_called()

    def test_local_libre_does_not_depend_on_public_internet_probe(self):
        local = mock.Mock(return_value="આ એક સંપૂર્ણ વાક્ય છે")
        providers = (("libre", local, False),)
        with mock.patch.object(self.translator, "_ordered_providers", return_value=providers), \
             mock.patch.object(self.translator, "LIBRE_URL", "http://127.0.0.1:5000"), \
             mock.patch.object(self.translator, "_cache_get_record", return_value=None), \
             mock.patch.object(self.translator, "_cache_put"), \
             mock.patch.object(self.translator, "is_online") as online:
            result = self.translator.translate_ex(
                "This is a complete sentence", "en", "gu"
            )

        self.assertTrue(result.translated)
        self.assertEqual(result.source, "libre")
        local.assert_called_once()
        online.assert_not_called()

    def test_sentence_chunker_preserves_text_order_and_size(self):
        source = (
            "First sentence contains several words. Second sentence follows.\n\n"
            "The final paragraph also contains several words."
        )
        chunks = self.translator._sentence_chunks(source, limit=45)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(text) <= 45 for text, _ in chunks))
        rebuilt = "".join(text + separator for text, separator in chunks).strip()
        self.assertEqual(" ".join(rebuilt.split()), " ".join(source.split()))

    def test_cache_is_versioned_and_records_provider_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "translations.json"
            with mock.patch.object(self.translator, "CACHE_PATH", cache_path), \
                 mock.patch.object(self.translator, "_cache", None):
                passed, report = self.translator.validate_translation(
                    "hello", "नमस्ते", "en", "hi"
                )
                self.assertTrue(passed)
                self.translator._cache_put(
                    "hello", "en", "hi", "नमस्ते",
                    provider="test", model="model-v1", validation=report,
                )
                payload = json.loads(cache_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], 2)
        record = next(iter(payload["entries"].values()))
        self.assertEqual(record["provider"], "test")
        self.assertEqual(record["model"], "model-v1")
        self.assertTrue(record["validated"])

    def test_diagnostics_reports_all_six_pairs_without_network(self):
        with mock.patch.object(self.translator, "_load_cache", return_value={
            "schema_version": 2, "entries": {}
        }):
            info = self.translator.diagnostics()
        self.assertEqual(len(info["supported_pairs"]), 6)
        self.assertIn("hi->gu", info["supported_pairs"])
        self.assertIn("gu->hi", info["supported_pairs"])

    def test_configured_provider_order_can_disable_cloud_fallbacks(self):
        with mock.patch.object(self.translator, "PROVIDER_ORDER", ("indictrans2",)):
            providers = self.translator._ordered_providers()
        self.assertEqual([name for name, _provider, _network in providers], ["indictrans2"])

    def test_indictrans2_maps_all_six_pairs_to_the_right_bundle(self):
        from modules.indictrans2_engine import IndicTrans2Engine

        expected = {
            ("en", "hi"): "en-indic", ("en", "gu"): "en-indic",
            ("hi", "en"): "indic-en", ("gu", "en"): "indic-en",
            ("hi", "gu"): "indic-indic", ("gu", "hi"): "indic-indic",
        }
        for pair, bundle in expected.items():
            with self.subTest(pair=pair):
                self.assertEqual(IndicTrans2Engine._bundle_for(*pair), bundle)


class LanguageAwareVoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.voice = importlib.import_module("modules.voice")

    def test_sphinx_is_never_used_for_hindi_or_gujarati(self):
        with mock.patch.object(self.voice, "SR_AVAILABLE", True), \
             mock.patch.object(self.voice, "SPHINX_AVAILABLE", True):
            self.assertTrue(self.voice._engine_available("sphinx", "en-IN"))
            self.assertFalse(self.voice._engine_available("sphinx", "hi-IN"))
            self.assertFalse(self.voice._engine_available("sphinx", "gu-IN"))

    def test_vosk_loads_the_model_for_the_requested_language(self):
        hindi_model = object()
        with mock.patch.object(self.voice, "_get_vosk_model", return_value=hindi_model) as get, \
             mock.patch.object(self.voice, "KaldiRecognizer") as recognizer_class:
            recognizer = recognizer_class.return_value
            recognizer.FinalResult.return_value = '{"text": "नमस्ते"}'
            result = self.voice._transcribe_vosk(b"audio", "hi-IN")

        self.assertEqual(result, "नमस्ते")
        get.assert_called_once_with("hi-IN")
        recognizer_class.assert_called_once_with(hindi_model, self.voice.SAMPLE_RATE)


class TranslationModeIntegrationTests(unittest.TestCase):
    def test_private_choice_is_passed_to_translation(self):
        main = importlib.import_module("main")
        translator = mock.Mock()
        translator.LANG_TO_TTS = {"en": "eng", "hi": "hin", "gu": "guj"}
        translator.translate_ex.return_value = SimpleNamespace(
            text="નમસ્તે", lang="gu", translated=True, source="mock", error=None
        )
        privacy = mock.MagicMock()
        privacy.ask_confidentiality.return_value = "PRIVATE"

        with mock.patch.dict(main._modules, {
            "translator": translator,
            "privacy": privacy,
        }), \
             mock.patch.object(main, "_select_with_buttons", side_effect=["1", "1", "3"]), \
             mock.patch.object(main, "_keyboard_input", return_value="hello"), \
             mock.patch.object(main, "_speak"):
            main.mode_translate()

        translator.translate_ex.assert_called_once_with(
            "hello", from_lang="en", to_lang="gu", privacy=True
        )
        privacy.PrivateAudio.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
