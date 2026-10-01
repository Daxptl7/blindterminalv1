"""Focused contract tests for the Gemini Live audio-send boundary."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

from modules import live_translator


class LiveTranslatorTests(unittest.TestCase):
    def setUp(self):
        self.translator = live_translator.LiveTranslator()

    def test_uses_current_sdk_realtime_audio_method_when_available(self):
        session = type("Session", (), {})()
        session.send_realtime_input = AsyncMock()
        session.send = AsyncMock()

        asyncio.run(self.translator._send_audio_chunk(session, b"pcm"))

        session.send_realtime_input.assert_awaited_once()
        self.assertEqual(
            session.send_realtime_input.await_args.kwargs["audio"].data, b"pcm"
        )
        session.send.assert_not_awaited()

    def test_uses_legacy_sdk_envelope_when_realtime_method_is_unavailable(self):
        session = type("Session", (), {})()
        session.send = AsyncMock()

        asyncio.run(self.translator._send_audio_chunk(session, b"pcm"))

        session.send.assert_awaited_once()
        envelope = session.send.await_args.kwargs["input"]
        self.assertEqual(envelope.media_chunks[0].data, b"pcm")

    def test_default_model_is_audio_capable_live_model(self):
        self.assertEqual(self.translator.model, live_translator.DEFAULT_LIVE_MODEL)

    def test_live_audio_uses_clean_input_by_default(self):
        self.assertEqual(self.translator.mic_gain, 1.0)

    def test_enter_stop_is_safe_without_an_interactive_terminal(self):
        with patch.object(live_translator.sys.stdin, "isatty", return_value=False):
            self.assertFalse(live_translator.LiveTranslator._enter_pressed())

    def test_receiver_stays_alive_for_a_second_turn(self):
        self.translator.running = True

        class Session:
            def __init__(self, owner):
                self.owner = owner
                self.receive_calls = 0

            async def receive(self):
                self.receive_calls += 1
                yield SimpleNamespace(
                    server_content=SimpleNamespace(
                        input_transcription=None,
                        output_transcription=None,
                        model_turn=None,
                        interrupted=False,
                        turn_complete=True,
                    )
                )
                if self.receive_calls == 2:
                    self.owner.running = False

        session = Session(self.translator)
        asyncio.run(self.translator._receive_audio_loop(session))
        self.assertEqual(session.receive_calls, 2)


if __name__ == "__main__":
    unittest.main()
