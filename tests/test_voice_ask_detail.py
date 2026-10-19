"""Voice answer instructions and generation budgets, without network or audio."""

import unittest
from unittest import mock

from modules import ai_query
from modules.privacy_policy import PRIVATE, PUBLIC, processing_session


class DetailedVoiceAnswerTests(unittest.TestCase):
    def test_chat_providers_allow_long_explanations(self):
        for provider, init_name, client_name in (
            (ai_query._ask_groq, "_init_groq", "_groq_client"),
            (ai_query._ask_openai, "_init_openai", "_openai_client"),
        ):
            with self.subTest(provider=provider.__name__):
                client = mock.Mock()
                client.with_options.return_value = client
                create = client.chat.completions.create
                create.return_value.choices = [
                    mock.Mock(message=mock.Mock(content="Explanation.\n\nExample."))
                ]
                with mock.patch.object(ai_query, init_name, return_value=True), \
                        mock.patch.object(ai_query, client_name, client):
                    self.assertEqual(
                        provider("Explain gravity", simplify=True, detailed=True),
                        "Explanation.\n\nExample.",
                    )
                    detailed = create.call_args.kwargs
                    provider("A quick question")
                    brief = create.call_args.kwargs
                self.assertGreater(detailed["max_tokens"], brief["max_tokens"])
                system = detailed["messages"][0]["content"]
                self.assertIn("step by step", system)
                self.assertIn("Use very simple words.", system)
                self.assertNotIn("1-3 short sentences", system)

    def test_gemini_fallback_allows_long_explanations(self):
        client = mock.Mock()
        client.models.generate_content.return_value.text = "A detailed explanation."
        with mock.patch.object(ai_query, "_init_gemini", return_value=True), \
                mock.patch.object(ai_query, "_gemini_client", client):
            self.assertEqual(ai_query._ask_gemini("Why?", detailed=True),
                             "A detailed explanation.")
        config = client.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.max_output_tokens, ai_query.DETAILED_MAX_TOKENS)
        self.assertEqual(config.system_instruction, ai_query.DETAILED_SYSTEM_PROMPT)

    def test_detail_survives_each_provider_failure(self):
        with processing_session(PUBLIC), \
                mock.patch.object(ai_query, "get_rag_pipeline", return_value=None), \
                mock.patch.object(ai_query, "_ask_groq", return_value=None) as groq, \
                mock.patch.object(ai_query, "_ask_openai", return_value=None) as openai, \
                mock.patch.object(ai_query, "_ask_gemini", return_value=None) as gemini, \
                mock.patch.object(ai_query, "_ask_offline", return_value="Long answer") as offline:
            self.assertEqual(ai_query.ask_ai("Why?", detailed=True), "Long answer")
        for provider in (groq, openai, gemini, offline):
            self.assertTrue(provider.call_args.kwargs["detailed"])
        for provider in (groq, openai, gemini):
            self.assertGreaterEqual(provider.call_args.kwargs["timeout"], 29)

    def test_private_detailed_answer_stays_local(self):
        with processing_session(PRIVATE), \
                mock.patch.object(ai_query, "get_rag_pipeline") as rag, \
                mock.patch.object(ai_query, "_ask_groq") as groq, \
                mock.patch.object(ai_query, "_ask_openai") as openai, \
                mock.patch.object(ai_query, "_ask_gemini") as gemini, \
                mock.patch.object(ai_query, "_ask_offline", return_value="Local explanation") as offline:
            self.assertEqual(ai_query.ask_ai("Why?", detailed=True), "Local explanation")
        self.assertTrue(offline.call_args.kwargs["detailed"])
        for provider in (rag, groq, openai, gemini):
            provider.assert_not_called()

    def test_offline_detail_includes_instructions_and_allows_paragraphs(self):
        model = mock.Mock(return_value={"choices": [{"text": "Explanation.\n\nExample."}]})
        for path in ("tinyllama.gguf", "qwen.gguf", "llama-3.gguf"):
            with self.subTest(model=path), \
                    mock.patch.dict(ai_query._settings, {"offline_model_path": path}), \
                    mock.patch.object(ai_query, "_start_offline_preload"), \
                    mock.patch.object(ai_query, "_offline_loading"), \
                    mock.patch.object(ai_query, "_offline_ready", True), \
                    mock.patch.object(ai_query, "_offline_model", model):
                self.assertEqual(ai_query._ask_offline("Why?", detailed=True),
                                 "Explanation.\n\nExample.")
                self.assertIn(ai_query.DETAILED_SYSTEM_PROMPT, model.call_args.args[0])
                self.assertNotIn("\n\n", model.call_args.kwargs["stop"])
                self.assertGreater(model.call_args.kwargs["max_tokens"], 150)


if __name__ == "__main__":
    unittest.main()
