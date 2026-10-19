"""Exercise the recorded-audio API boundary without a network or microphone."""
import asyncio
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import benchmark_live_translation as benchmark


class LiveBenchmarkTests(unittest.TestCase):
    def test_clip_loader_rejects_non_pcm16(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"clip.wav"
            with wave.open(str(path),"wb") as out:
                out.setnchannels(1)
                out.setsampwidth(1)
                out.setframerate(16000)
                out.writeframes(bytes(1600))
            with self.assertRaisesRegex(ValueError,"PCM16"):
                benchmark.load_clip(path)

    def test_clip_loader_downmixes_stereo_and_reports_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"clip.wav"
            with wave.open(str(path),"wb") as out:
                out.setnchannels(2)
                out.setsampwidth(2)
                out.setframerate(48000)
                out.writeframes(b"\x01\x00\x03\x00" * 4800)
            pcm,rate,duration=benchmark.load_clip(path)
        self.assertEqual(pcm,b"\x02\x00"*4800)
        self.assertEqual(rate,48000)
        self.assertAlmostEqual(duration,0.1)

    def test_collect_turn_preserves_all_audio_and_transcript_parts(self):
        class Session:
            def __init__(self):
                self.send_realtime_input=mock.AsyncMock()
                self.receive_count=0
            async def receive(self):
                self.receive_count += 1
                if self.receive_count > 1:
                    await asyncio.Event().wait()
                for complete in (False,True):
                    yield SimpleNamespace(server_content=SimpleNamespace(
                        input_transcription=SimpleNamespace(text="source "),
                        output_transcription=SimpleNamespace(text="translation "),
                        model_turn=SimpleNamespace(parts=[SimpleNamespace(
                            inline_data=SimpleNamespace(data=b"\x01\x00"*20))]),
                        turn_complete=complete))
        session=Session()
        result=asyncio.run(benchmark.collect_turn(session,bytes(3200),16000,3))
        self.assertEqual(result["audio"],b"\x01\x00"*40)
        self.assertEqual(result["output_transcript"],"translation translation ")
        self.assertIsNotNone(result["first_audio_s"])
        self.assertTrue(session.send_realtime_input.await_args.kwargs["audio_stream_end"])

    def test_no_audio_cannot_pass_the_benchmark(self):
        class Session:
            send_realtime_input=mock.AsyncMock()
            async def receive(self):
                await asyncio.Event().wait()
                if False:
                    yield
        with self.assertRaisesRegex(benchmark.live.LiveTranslationError,"timed out"):
            asyncio.run(benchmark.collect_turn(Session(),bytes(3200),16000,0.02))


if __name__ == "__main__":
    unittest.main()
