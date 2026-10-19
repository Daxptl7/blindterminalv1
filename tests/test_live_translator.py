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


class LiveReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.settings = {"gemini_api_key": "test-key", "live_input_device": 0,
                         "live_output_device": 1, "live_max_reconnects": 1}
        self.loader = patch.object(live_translator, "_load_settings", return_value=self.settings)
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.translator = live_translator.LiveTranslator()

    def test_configuration_preserves_device_zero_and_translation_instructions(self):
        self.assertEqual(self.translator.input_device, 0)
        config = self.translator._build_config()
        self.assertIn("Preserve negation", config.system_instruction)
        self.assertIn("do not summarize", config.system_instruction)
        self.assertEqual(config.realtime_input_config.automatic_activity_detection.silence_duration_ms, 700)
        self.assertIsNotNone(config.context_window_compression)

    def test_missing_audio_runtime_raises_actionable_error(self):
        with patch.object(live_translator, "SD_AVAILABLE", False):
            with self.assertRaisesRegex(live_translator.LiveTranslationError, "sounddevice"):
                self.translator.run_live()
        self.assertFalse(self.translator.running)

    def test_cloud_errors_do_not_expose_exception_details(self):
        error = RuntimeError("api_key=PRIVATE_SENTINEL")
        error.code = 403
        safe = live_translator._service_error(error)
        self.assertFalse(safe.retryable)
        self.assertNotIn("PRIVATE_SENTINEL", str(safe))

    def test_microphone_queue_never_silently_drops_speech(self):
        t = self.translator
        t.running = True
        for _ in range(live_translator.MAX_PENDING_AUDIO_CHUNKS + 1):
            t._queue_microphone_audio(b"\x01\x00" * 1600)
        self.assertEqual(t.audio_in_queue.qsize(), live_translator.MAX_PENDING_AUDIO_CHUNKS)
        self.assertIsNotNone(t._failure)
        self.assertEqual(t.stats["input_overflows"], 1)

    def test_echo_guard_keeps_silence_flowing_to_server(self):
        t = self.translator
        t.running = True
        t._model_speaking.set()
        t._queue_microphone_audio(b"\x01\x00" * 1600)
        data, muted = t.audio_in_queue.get_nowait()
        self.assertTrue(muted)
        self.assertEqual(data, bytes(3200))

    def test_headset_can_keep_microphone_open_during_translation(self):
        t = self.translator
        t.running = True
        t.echo_guard = False
        t._model_speaking.set()
        t._queue_microphone_audio(b"\x01\x00" * 1600)
        data, muted = t.audio_in_queue.get_nowait()
        self.assertFalse(muted)
        self.assertNotEqual(data, bytes(3200))

    def test_microphone_overflow_is_reported(self):
        t = self.translator
        t.running = True
        t._microphone_callback(b"", 0, None, SimpleNamespace(input_overflow=True))
        self.assertIn("Microphone audio was lost", str(t._failure))

    def test_native_microphone_rate_is_sent_accurately(self):
        from modules.live_audio import AudioFormat
        self.translator.input_format = AudioFormat(0, "USB", 48000, 2)
        session = SimpleNamespace(send_realtime_input=AsyncMock())
        asyncio.run(self.translator._send_audio_chunk(session, b"\x00\x00"))
        self.assertEqual(session.send_realtime_input.await_args.kwargs["audio"].mime_type,
                         "audio/pcm;rate=48000")

    def test_playback_backlog_is_bounded_and_reported(self):
        t = self.translator
        for _ in range(t.playback_queue.maxsize):
            t._put_playback(b"pcm")
        with self.assertRaisesRegex(live_translator.LiveTranslationError, "speaker could not keep up"):
            t._put_playback(b"overflow")

    def test_receive_processes_every_audio_part_across_two_turns(self):
        t = self.translator
        t.running = True
        class Session:
            count = 0
            async def receive(self):
                self.count += 1
                parts = [SimpleNamespace(inline_data=SimpleNamespace(data=b"\xe8\x03" * 10,
                         mime_type="audio/pcm;rate=24000")) for _ in range(2)]
                yield SimpleNamespace(server_content=SimpleNamespace(
                    model_turn=SimpleNamespace(parts=parts), turn_complete=True))
                if self.count == 2:
                    t.running = False
        asyncio.run(t._receive_audio_loop(Session()))
        self.assertEqual(t.stats["turns"], 2)
        self.assertEqual(t.stats["output_chunks"], 4)
        self.assertEqual(t.playback_queue.qsize(), 6)

    def test_closed_stream_exits_instead_of_spinning(self):
        self.translator.running = True
        class EmptySession:
            async def receive(self):
                if False:
                    yield
        with self.assertRaisesRegex(live_translator.LiveTranslationError, "stream closed"):
            asyncio.run(self.translator._receive_audio_loop(EmptySession()))

    def test_no_audio_response_times_out(self):
        t = self.translator
        t.running = True
        t._pending_response = True
        t._last_voice = 0
        with self.assertRaisesRegex(live_translator.LiveTranslationError, "No spoken translation"):
            asyncio.run(t._monitor())

    def test_connection_errors_retry_but_authentication_errors_do_not(self):
        for retryable, expected_calls in ((True, 2), (False, 1)):
            with self.subTest(retryable=retryable):
                client = SimpleNamespace(aio=SimpleNamespace(aclose=AsyncMock()), close=lambda: None)
                failure = live_translator.LiveTranslationError("Unavailable", retryable=retryable)
                with patch.object(live_translator.genai, "Client", return_value=client), \
                        patch.object(self.translator, "_connected_session", AsyncMock(side_effect=failure)) as connect, \
                        patch.object(self.translator, "_announce"):
                    with self.assertRaises(live_translator.LiveTranslationError):
                        asyncio.run(self.translator._conversation())
                self.assertEqual(connect.await_count, expected_calls)
                client.aio.aclose.assert_awaited_once()

    def test_stop_cancels_an_in_progress_connection(self):
        async def exercise():
            self.translator.running = True
            cancelled = asyncio.Event()
            async def connecting():
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            with patch.object(self.translator, "_conversation", connecting):
                await asyncio.wait_for(self.translator._run_async(lambda: True), timeout=0.5)
            self.assertTrue(cancelled.is_set())
        asyncio.run(exercise())

    def test_stop_discards_old_audio(self):
        self.translator.running = True
        self.translator._queue_microphone_audio(b"\x01\x00")
        self.translator._put_playback(b"\x02\x00")
        self.translator.stop()
        self.assertTrue(self.translator.audio_in_queue.empty())
        self.assertTrue(self.translator.playback_queue.empty())


class LiveAudioTests(unittest.TestCase):
    def test_linux_routes_reuse_device_configuration(self):
        from modules.live_audio import configured_device
        with patch("modules.live_audio.sys.platform", "linux"):
            self.assertEqual(configured_device({"mic_device":"plughw:4,0"}, "input"), "plughw:4,0")
            self.assertEqual(configured_device({"live_input_device":0,"mic_device":"other"}, "input"), 0)
        with patch("modules.live_audio.sys.platform", "darwin"):
            self.assertIsNone(configured_device({"mic_device":"plughw:4,0"}, "input"))

    def test_alsa_route_selects_the_correct_portaudio_index(self):
        from modules.live_audio import resolve_device
        from unittest.mock import Mock
        sd = Mock()
        sd.query_devices.return_value = [
            {"name":"USB (hw:3,01)","max_output_channels":2},
            {"name":"USB (hw:3,0)","max_output_channels":2},
        ]
        self.assertEqual(resolve_device(sd, "plughw:3,0", "output"), 1)

    def test_unsupported_rates_fall_back_to_native_usb_rate(self):
        from modules.live_audio import negotiate
        from unittest.mock import Mock
        sd = Mock()
        sd.query_devices.return_value = {"name":"USB", "default_samplerate":48000,
                                        "max_input_channels":1}
        def check(**kw):
            if kw["samplerate"] != 48000:
                raise ValueError("Unsupported rate")
        sd.check_input_settings.side_effect = check
        fmt = negotiate(sd, 0, "input", 16000)
        self.assertEqual(fmt.rate, 48000)
        self.assertEqual(fmt.device, 0)

    def test_stereo_only_device_is_supported(self):
        from modules.live_audio import negotiate
        from unittest.mock import Mock
        sd = Mock()
        sd.query_devices.return_value = {"name":"USB", "default_samplerate":48000,
                                        "max_input_channels":2}
        def check(**kw):
            if kw["channels"] != 2:
                raise ValueError("Stereo required")
        sd.check_input_settings.side_effect = check
        self.assertEqual(negotiate(sd, 0, "input", 16000).channels, 2)

    def test_downmix_and_gain_do_not_wrap(self):
        import numpy as np
        from modules.live_audio import microphone_pcm
        source = np.array([32000,32000,-32000,-32000], dtype="<i2").tobytes()
        out = np.frombuffer(microphone_pcm(source, 2, 2), dtype="<i2")
        self.assertEqual(out.tolist(), [32767,-32768])

    def test_output_resampling_keeps_state_across_packets(self):
        import numpy as np
        from modules.live_audio import AudioFormat, PlaybackConverter
        fmt = AudioFormat(1, "USB", 48000, 2)
        samples = (np.sin(np.arange(4800) * 0.1) * 10000).astype("<i2").tobytes()
        whole = PlaybackConverter(fmt).convert(samples)
        chunked = PlaybackConverter(fmt)
        joined = b"".join(chunked.convert(samples[i:i+4800]) for i in range(0,len(samples),4800))
        self.assertEqual(joined, whole)
        self.assertEqual(np.frombuffer(joined,dtype="<i2")[::2].tolist(),
                         np.frombuffer(joined,dtype="<i2")[1::2].tolist())

    def test_invalid_live_settings_are_rejected(self):
        from modules.config_loader import validate_settings, SettingsError
        for settings in ({"live_input_device":True}, {"live_mic_gain":float("nan")},
                         {"live_echo_guard":"false"}, {"live_max_reconnects":-1},
                         {"live_vad_silence_ms":20}):
            with self.subTest(settings=settings), self.assertRaises(SettingsError):
                validate_settings(settings)


class LiveLifecycleTests(unittest.TestCase):
    def test_real_session_supervision_cleans_up_and_can_run_twice(self):
        from unittest.mock import Mock, MagicMock
        from modules.live_audio import AudioFormat
        settings = {"gemini_api_key":"fake", "live_input_device":0,"live_output_device":1}
        with patch.object(live_translator,"_load_settings",return_value=settings):
            t = live_translator.LiveTranslator()
        t.input_format = AudioFormat(0,"USB mic",48000,1)
        t.output_format = AudioFormat(1,"USB speaker",48000,2)
        calls = []
        class Session:
            count = 0
            async def send_realtime_input(self, **kwargs):
                pass
            async def receive(self):
                self.count += 1
                if self.count > 2:
                    await asyncio.Event().wait()
                yield SimpleNamespace(server_content=SimpleNamespace(
                    model_turn=SimpleNamespace(parts=[SimpleNamespace(inline_data=SimpleNamespace(
                        data=b"\xe8\x03" * 2400, mime_type="audio/pcm;rate=24000"))]),
                    turn_complete=True))
        for _ in range(2):
            session = Session()
            context = MagicMock()
            context.__aenter__ = AsyncMock(return_value=session)
            context.__aexit__ = AsyncMock(return_value=False)
            client = Mock()
            client.aio.live.connect.return_value = context
            client.aio.aclose = AsyncMock()
            speaker = Mock(latency=0.1)
            speaker.write.return_value = False
            mic = Mock()
            sd = Mock()
            sd.RawOutputStream.side_effect = lambda **kwargs: calls.append("speaker") or speaker
            sd.RawInputStream.return_value = mic
            with patch.object(live_translator,"SD_AVAILABLE",True), \
                    patch.object(live_translator,"sd",sd), \
                    patch.object(t,"_check_audio_devices",return_value=True), \
                    patch.object(live_translator.genai,"Client",return_value=client):
                self.assertTrue(t.run_live(stop_check=lambda:session.count > 2,
                                          status_fn=lambda text:calls.append("announcement")))
            self.assertEqual(t.stats["turns"],2)
            self.assertFalse(t._playback_thread.is_alive())
            mic.close.assert_called_once()
            speaker.close.assert_called_once()
            context.__aexit__.assert_awaited_once()
            client.aio.aclose.assert_awaited_once()
        self.assertEqual(calls,["announcement","speaker","announcement","speaker"])


class LiveMenuTests(unittest.TestCase):
    def test_hindi_gujarati_pair_and_spoken_failure(self):
        import importlib
        from unittest.mock import Mock
        main = importlib.import_module("main")
        live_module = Mock()
        live_module.LiveTranslationError = live_translator.LiveTranslationError
        interpreter = live_module.LiveTranslator.return_value
        interpreter.run_live.side_effect = live_translator.LiveTranslationError("Check the USB speaker.")
        cancel = Mock()
        with patch.dict(main._modules,{"translator":Mock(),"live_translator":live_module}), \
                patch.object(main,"_select_with_buttons",side_effect=["33","3"]), \
                patch.object(main,"_button3_stop_signal",return_value=(lambda:False,cancel)), \
                patch.object(main,"_settle_button_input"), \
                patch.object(main,"_speak") as speak:
            main.mode_translate()
        live_module.LiveTranslator.assert_called_once_with(lang_pair="hi-gu")
        speak.assert_any_call("Check the USB speaker.",block=True)
        cancel.assert_called_once()


class LiveSpeechBoundaryTests(unittest.TestCase):
    def test_first_speech_is_retained_and_silence_finalizes_once(self):
        async def exercise():
            t=live_translator.LiveTranslator()
            t.running=True
            voice=b"\xe8\x03" * 1600
            silence=bytes(3200)
            for data in [voice]*2+[silence]*8:
                t._queue_microphone_audio(data)
            session=SimpleNamespace()
            async def send(**kwargs):
                if kwargs.get("audio_stream_end"):
                    t.running=False
            session.send_realtime_input=AsyncMock(side_effect=send)
            await asyncio.wait_for(t._send_audio_loop(session),1)
            calls=session.send_realtime_input.await_args_list
            self.assertEqual([c.kwargs["audio"].data for c in calls[:3]],[silence]*3)
            self.assertEqual(calls[3].kwargs["audio"].data,voice)
            self.assertEqual(sum(bool(c.kwargs.get("audio_stream_end")) for c in calls),1)
        asyncio.run(exercise())

    def test_all_zero_audio_is_not_accepted_as_a_translation(self):
        t=live_translator.LiveTranslator()
        t.running=True
        class Session:
            async def receive(self):
                yield SimpleNamespace(server_content=SimpleNamespace(
                    model_turn=SimpleNamespace(parts=[SimpleNamespace(inline_data=SimpleNamespace(
                        data=bytes(4800),mime_type="audio/pcm;rate=24000"))]),
                    turn_complete=True))
        with self.assertRaisesRegex(live_translator.LiveTranslationError,"silent translation"):
            asyncio.run(t._receive_audio_loop(Session()))


if __name__ == "__main__":
    unittest.main()
