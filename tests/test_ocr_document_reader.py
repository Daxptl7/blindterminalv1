"""Regression tests for complete, privacy-safe OCR document playback."""

import re
import unittest
from unittest import mock

from modules import confidential_mode


class OCRDocumentReaderTests(unittest.TestCase):
    def test_splitter_keeps_text_from_the_end_of_a_long_page(self):
        text = (
            "A first textbook sentence with enough words to require chunking. "
            "A second sentence contains more body text for the reader.\n\n"
            "The final paragraph must also be spoken. FINAL_PAGE_SENTINEL"
        )

        chunks = confidential_mode.split_text_for_speech(text, max_chars=80)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 80 for chunk in chunks))
        self.assertIn("FINAL_PAGE_SENTINEL", chunks[-1])
        self.assertEqual(
            re.sub(r"\s+", " ", " ".join(chunks)).strip(),
            re.sub(r"\s+", " ", text).strip(),
        )

    def test_private_document_asks_once_and_speaks_every_chunk(self):
        text = " ".join(f"word{index}" for index in range(90))

        with mock.patch.object(
            confidential_mode, "ask_confidentiality", return_value="PRIVATE"
        ) as ask, mock.patch.object(
            confidential_mode, "PrivateAudio"
        ) as private_audio, mock.patch.object(
            confidential_mode.tts, "speak"
        ) as speak:
            mode = confidential_mode.speak_document_with_privacy_check(
                text, "eng", morse_serial=object(), chunk_chars=100
            )

        self.assertEqual(mode, "PRIVATE")
        ask.assert_called_once()
        private_audio.assert_called_once()
        self.assertGreater(speak.call_count, 1)
        spoken = " ".join(call.args[0] for call in speak.call_args_list)
        self.assertIn("word89", spoken)
        self.assertTrue(all(call.kwargs.get("block") for call in speak.call_args_list))

    def test_public_document_uses_speaker_and_speaks_every_chunk(self):
        text = " ".join(f"line{index}" for index in range(80))

        with mock.patch.object(
            confidential_mode, "ask_confidentiality", return_value="NORMAL"
        ) as ask, mock.patch.object(
            confidential_mode, "enable_speaker"
        ) as speaker, mock.patch.object(
            confidential_mode.tts, "speak"
        ) as speak:
            mode = confidential_mode.speak_document_with_privacy_check(
                text, "eng", chunk_chars=90
            )

        self.assertEqual(mode, "NORMAL")
        ask.assert_called_once()
        speaker.assert_called_once()
        self.assertGreater(speak.call_count, 1)
        self.assertIn("line79", " ".join(call.args[0] for call in speak.call_args_list))


if __name__ == "__main__":
    unittest.main()
