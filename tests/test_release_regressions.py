"""Release regressions for privacy boundaries, index integrity and cancellation."""
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from modules.privacy_policy import PRIVATE, PUBLIC, current_policy, processing_session
from modules import tts, confidential_mode as privacy, ai_query, gps_navigator as gps
from modules.processing_feedback import run_with_feedback, ProcessingCancelled
from services.embedder import Embedder
from services.vector_store import VectorStore
from services.rag_pipeline import RAGPipeline


class PrivateSessionTests(unittest.TestCase):
    def test_private_route_failure_never_uses_public_players(self):
        manager = tts.TTSManager.__new__(tts.TTSManager)
        manager._speaking = threading.Event()
        with processing_session(PRIVATE), \
                mock.patch.object(tts, 'require_private_output', return_value='private'), \
                mock.patch.object(manager, '_synthesize', return_value=(b'wav', 'wav')), \
                mock.patch.object(manager, '_play_routed', return_value=False), \
                mock.patch.object(manager, '_play_pygame') as pygame, \
                mock.patch.object(manager, '_play_system') as system, \
                mock.patch.object(manager, '_play_pyttsx3_direct') as direct:
            with self.assertRaises(tts.AudioRoutingError):
                manager._deliver('PRIVATE_SENTINEL', 'en', 'private')
        pygame.assert_not_called()
        system.assert_not_called()
        direct.assert_not_called()

    def test_private_synthesis_never_calls_gtts(self):
        manager = tts.TTSManager.__new__(tts.TTSManager)
        manager.settings = {'tts_engine': 'gtts'}
        manager._gtts_failed_at = 0
        fake_gtts = SimpleNamespace(gTTS=mock.Mock())
        with processing_session(PRIVATE), mock.patch.dict('sys.modules', {'gtts': fake_gtts}), \
                mock.patch.object(manager, '_synthesize_offline_wav', return_value=b'local'):
            self.assertEqual(manager._synthesize('PRIVATE_SENTINEL', 'en'), (b'local', 'wav'))
        fake_gtts.gTTS.assert_not_called()

    def test_private_endpoint_must_be_verified_and_distinct(self):
        for available, selected in ((['speaker'], 'speaker'), (['speaker'], 'missing'), ([], 'private')):
            with self.subTest(available=available, selected=selected), \
                    mock.patch.object(tts, 'IS_MACOS', False), \
                    mock.patch.object(tts, 'SPEAKER_DEVICE', 'speaker'), \
                    mock.patch.object(tts, '_alsa_playback_devices', return_value=available):
                with self.assertRaises(tts.AudioRoutingError):
                    tts.require_private_output(selected)

    def test_timeout_does_not_speak_answer(self):
        with mock.patch.object(ai_query, '_speaker_device', return_value='speaker'), \
                mock.patch.object(ai_query, '_earphone_device', return_value='private'), \
                mock.patch.object(ai_query, '_wait_for_route_choice', return_value=None), \
                mock.patch.object(ai_query, '_say_on_device') as say:
            self.assertEqual(ai_query.deliver_answer('PRIVATE_SENTINEL'), 'none')
        self.assertFalse(any(call.args[0] == 'PRIVATE_SENTINEL' for call in say.call_args_list))

    def test_private_ai_bypasses_retrieval_and_cloud(self):
        with processing_session(PRIVATE), \
                mock.patch.object(ai_query, 'get_rag_pipeline') as rag, \
                mock.patch.object(ai_query, '_ask_groq') as groq, \
                mock.patch.object(ai_query, '_ask_openai') as openai, \
                mock.patch.object(ai_query, '_ask_gemini') as gemini, \
                mock.patch.object(ai_query, '_ask_offline', return_value='local answer'):
            self.assertEqual(ai_query.ask_ai('PRIVATE_SENTINEL'), 'local answer')
            self.assertFalse(ai_query.index_text_in_rag('PRIVATE_SENTINEL'))
        for provider in (rag, groq, openai, gemini):
            provider.assert_not_called()

    def test_private_ocr_does_not_upload_the_image(self):
        from modules import ocr
        with processing_session(PRIVATE), \
                mock.patch.dict(ocr._config, {'ocr_engine':'auto','ocr_gemini_first':True,
                    'gemini_api_key':'synthetic-key','ocr_guidance_enabled':False}), \
                mock.patch.object(ocr._camera, 'open', return_value=True), \
                mock.patch.object(ocr._camera, 'capture', return_value=np.zeros((100,100,3), np.uint8)), \
                mock.patch.object(ocr._gemini_engine, 'extract_text') as cloud, \
                mock.patch.object(ocr._tesseract_engine, 'extract_text', return_value='Private local document'):
            self.assertEqual(ocr.scan_and_read(), 'Private local document')
        cloud.assert_not_called()

    def test_dispatch_selects_privacy_before_capture_and_preserves_explanation(self):
        import main
        events, spoken = [], []
        ocr = mock.Mock()
        ocr.open_camera.side_effect = lambda: events.append('capture') or True
        ocr.scan_and_read.return_value = 'PRIVATE_DOCUMENT_SENTINEL'
        ai = mock.Mock()
        ai.ask_ai.side_effect = lambda *a, **k: 'PRIVATE_EXPLANATION_SENTINEL'
        def say(text, *args, **kwargs):
            spoken.append((text, current_policy().private))
        with mock.patch.dict(main._modules, {'ocr':ocr,'ai_query':ai,'privacy':privacy}), \
                mock.patch.dict(main._settings, {'enabled_modes':['1']}), \
                mock.patch.object(privacy, 'ask_confidentiality', side_effect=lambda *a: events.append('privacy') or 'PRIVATE'), \
                mock.patch.object(tts, 'require_private_output', return_value='private'), \
                mock.patch.object(tts, 'speak', side_effect=say), \
                mock.patch.object(main, '_speak', side_effect=say), \
                mock.patch.object(main, '_button3_stop_signal', return_value=(None, lambda:None)), \
                mock.patch.object(main, '_drain_button_messages'), \
                mock.patch.object(main, '_select_with_buttons', return_value='1'):
            main.run_mode('1')
        self.assertLess(events.index('privacy'), events.index('capture'))
        self.assertIn(('PRIVATE_EXPLANATION_SENTINEL', True), spoken)
        self.assertFalse(any('SENTINEL' in text and not private for text,private in spoken))
        ai.index_text_in_rag.assert_not_called()
        self.assertFalse(current_policy().private)

    def test_background_processing_inherits_policy(self):
        with processing_session(PRIVATE):
            self.assertTrue(run_with_feedback(lambda: current_policy().private))

    def test_private_stt_never_calls_google(self):
        from modules import voice
        with processing_session(PRIVATE), mock.patch.object(voice, 'sr') as recognizer:
            self.assertIsNone(voice._transcribe_google(b'pcm', 'en-IN'))
        recognizer.Recognizer.assert_not_called()

    def test_private_recording_is_removed_after_transcription_error(self):
        from modules import voice
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/'private.wav')
            Path(path).write_bytes(b'synthetic')
            voice._private_recordings.add(path)
            with processing_session(PRIVATE), mock.patch.object(voice,'record',return_value=path), \
                    mock.patch.object(voice,'_active_engines',return_value=['vosk']), \
                    mock.patch.object(voice,'transcribe_file',side_effect=ValueError('failure')):
                with self.assertRaises(ValueError):
                    voice.listen()
            self.assertFalse(Path(path).exists())


class GPSReleaseTests(unittest.TestCase):
    def test_missing_gps_never_returns_test_location(self):
        with mock.patch.dict(gps._settings, {'app_mode':'device'}), \
                mock.patch.object(gps,'USE_GPIO',True), \
                mock.patch.object(gps,'_read_gps_hardware',return_value=None), \
                mock.patch('os.path.exists',return_value=False), \
                mock.patch.object(gps,'_read_gps_simulation') as simulation, \
                mock.patch.object(gps,'_reverse_geocode') as geocode:
            with self.assertRaises(gps.NoGPSFix):
                gps.get_location()
        simulation.assert_not_called()
        geocode.assert_not_called()

    def test_explicit_simulation_is_labelled(self):
        with mock.patch.dict(gps._settings, {'app_mode':'simulation'}), \
                mock.patch.object(gps,'USE_GPIO',False), mock.patch('os.path.exists',return_value=False), \
                mock.patch.object(gps,'_reverse_geocode',return_value='test address'):
            result=gps.get_location()
        self.assertTrue(result['simulated'])
        self.assertEqual(result['source'],'simulation')

    def test_invalid_and_stale_fixes_are_rejected(self):
        from datetime import datetime, timezone, timedelta
        now=datetime.now(timezone.utc)
        fix=SimpleNamespace(sentence_type='RMC',status='V',latitude=20,longitude=72,
                            timestamp=now.timetz(),datestamp=now.date())
        self.assertFalse(gps._valid_fix(fix))
        fix.status='A'
        self.assertTrue(gps._valid_fix(fix))
        old=now-timedelta(minutes=2)
        fix.timestamp,fix.datestamp=old.timetz(),old.date()
        self.assertFalse(gps._valid_fix(fix))

    def test_navigation_disabled_without_contacting_services(self):
        with mock.patch.dict(gps._settings, {'navigation_enabled':False}), \
                mock.patch.object(gps,'get_location') as location:
            self.assertIn('disabled',gps.get_directions('station'))
        location.assert_not_called()


class IndexIntegrityTests(unittest.TestCase):
    def store(self, directory):
        return VectorStore(directory, dimension=3, model_id='synthetic-model')

    def test_partial_embedding_batch_is_rejected(self):
        emb=Embedder.__new__(Embedder)
        emb.use_gemini=True
        emb.get_embedding=mock.Mock(side_effect=[np.array([]),np.array([1.,2.,3.])])
        with self.assertRaises(ValueError):
            emb.get_embeddings(['failed chunk','good chunk'])

    def test_invalid_batch_cannot_mutate_store(self):
        with tempfile.TemporaryDirectory() as directory:
            store=self.store(directory)
            store.add_text('original',np.array([1.,0.,0.]))
            for texts,vectors,tags in [(['a','b'],np.array([[1.,2.,3.]]),None),
                                       (['a'],np.array([[1.,np.nan,3.]]),None),
                                       (['a'],np.array([[1.,2.,3.]]),[{},{}])]:
                with self.assertRaises(ValueError):
                    store.add_batch(texts,vectors,tags)
            self.assertEqual(self.store(directory).metadata,['original'])

    def test_reingestion_is_idempotent_and_survives_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            store=self.store(directory)
            for _ in range(3):
                store.add_text('original',np.array([1.,0.,0.]),{'chapter':'one'})
            self.assertEqual(self.store(directory).metadata,['original'])

    def test_failed_publish_preserves_committed_state(self):
        with tempfile.TemporaryDirectory() as directory:
            store=self.store(directory)
            store.add_text('original',np.array([1.,0.,0.]))
            original_write=store._atomic_write
            def fail_publish(path,data):
                if path==store.snapshot_file:
                    raise OSError('synthetic disk full')
                return original_write(path,data)
            with mock.patch.object(store,'_atomic_write',side_effect=fail_publish):
                with self.assertRaises(OSError):
                    store.add_text('not committed',np.array([0.,1.,0.]))
            self.assertEqual(store.metadata,['original'])
            self.assertEqual(self.store(directory).metadata,['original'])

    def test_corrupt_primary_recovers_previous_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            store=self.store(directory)
            store.add_text('first',np.array([1.,0.,0.]))
            store.add_text('second',np.array([0.,1.,0.]))
            store.snapshot_file.write_text('{incomplete')
            self.assertEqual(self.store(directory).metadata,['first'])

    def test_different_model_identity_cannot_open_same_store(self):
        with tempfile.TemporaryDirectory() as directory:
            store=self.store(directory)
            store.add_text('original',np.array([1.,0.,0.]))
            with self.assertRaises(ValueError):
                VectorStore(directory,dimension=3,model_id='different-model')
            self.assertEqual(self.store(directory).metadata,['original'])

    def test_private_store_cannot_write(self):
        with tempfile.TemporaryDirectory() as directory:
            store=self.store(directory)
            with processing_session(PRIVATE), self.assertRaises(PermissionError):
                store.add_text('private',np.array([1.,0.,0.]))
            self.assertFalse(store.snapshot_file.exists())

    def test_failed_embedding_keeps_source_but_reports_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            store=self.store(str(Path(directory)/'model'))
            embedder=mock.Mock()
            embedder.get_embeddings.return_value=np.array([])
            pipeline=RAGPipeline(embedder,store,mock.Mock(),mock.Mock())
            source=Path(directory)/'chapter.txt'
            source.write_text('Original source text')
            self.assertFalse(pipeline.index_file(str(source)))
            saved=list((Path(directory)/'sources').glob('*.json'))
            self.assertEqual(len(saved),1)
            self.assertEqual(json.loads(saved[0].read_text())['text'],'Original source text')
            self.assertEqual(store.metadata,[])

    def test_lexical_search_works_without_dense_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            store=self.store(directory)
            store.add_text('Photosynthesis uses sunlight',np.array([1.,0.,0.]))
            self.assertEqual(store.retrieve(np.array([]),query_text='photosynthesis')[0][2],
                             'Photosynthesis uses sunlight')


class CancellationAndConfigurationTests(unittest.TestCase):
    def test_cancel_returns_before_blocked_worker_and_never_publishes_late_speech(self):
        started,finish,cancel=threading.Event(),threading.Event(),threading.Event()
        spoken=mock.Mock()
        def operation():
            started.set()
            finish.wait(2)
            return 'PRIVATE_SENTINEL'
        def stop():
            return started.is_set()
        try:
            before=time.monotonic()
            with self.assertRaises(ProcessingCancelled):
                run_with_feedback(operation,spoken,stop,first_delay=10)
            self.assertLess(time.monotonic()-before,.5)
            spoken.assert_not_called()
        finally:
            finish.set()

    def test_operation_has_an_overall_deadline(self):
        finish=threading.Event()
        try:
            with self.assertRaises(TimeoutError):
                run_with_feedback(lambda: finish.wait(2),timeout=.05)
        finally:
            finish.set()

    def test_invalid_configuration_is_explicit(self):
        from modules.config_loader import validate_settings, SettingsError
        for settings in ({'ai_race_timeout_s':'bad'},{'allow_ip_geolocation':'false'},
                         {'privacy':[]},{'app_mode':'unknown'},{'enabled_modes':['10']}):
            with self.subTest(settings=settings), self.assertRaises(SettingsError):
                validate_settings(settings)

    def test_no_offline_mock_answer(self):
        from services.gemini_agent import GeminiAgent
        agent=GeminiAgent.__new__(GeminiAgent)
        agent.client=None
        answer=agent.generate_response('question','textbook context')
        self.assertIn('unavailable',answer)
        self.assertNotIn('Mock',answer)

    def test_late_synthesis_cannot_play_after_stop_generation_changes(self):
        manager=tts.TTSManager.__new__(tts.TTSManager)
        manager._speaking=threading.Event()
        manager._generation=0
        def synthesize(*args):
            manager._generation+=1
            return b'audio','wav'
        with mock.patch.object(manager,'_synthesize',side_effect=synthesize), \
                mock.patch.object(manager,'_play_pygame') as play:
            with self.assertRaises(tts.AudioRoutingError):
                manager._deliver('late answer','en')
        play.assert_not_called()

    def test_private_exception_details_are_redacted(self):
        import logging
        from modules.privacy_policy import PrivateLogFilter
        record=logging.makeLogRecord({'msg':'PRIVATE_SENTINEL %s','args':('secret',),
                                      'exc_text':'PRIVATE_SENTINEL'})
        with processing_session(PRIVATE):
            PrivateLogFilter().filter(record)
        self.assertNotIn('SENTINEL',record.getMessage())
        self.assertIsNone(record.exc_text)

    def test_cloud_embedding_failure_never_changes_model(self):
        emb=Embedder.__new__(Embedder)
        emb.use_gemini=True
        emb.gemini_client=mock.Mock()
        emb.gemini_client.models.embed_content.side_effect=RuntimeError('offline')
        emb.gemini_embedding_model='synthetic'
        emb._gemini_config=None
        emb._init_local_model=mock.Mock()
        self.assertEqual(emb._compute_embedding('text').size,0)
        emb._init_local_model.assert_not_called()

    def test_private_live_translation_cannot_start(self):
        from modules.live_translator import LiveTranslator
        interpreter=LiveTranslator()
        with processing_session(PRIVATE), self.assertRaises(RuntimeError):
            interpreter.run_live()


class EvaluationAndConcurrentWriteTests(unittest.TestCase):
    def test_numeric_benchmark_rejects_swapped_quantities(self):
        from scripts import benchmark_ocr
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            Image.new('RGB',(10,10)).save(root/'image.png')
            (root/'manifest.json').write_text(json.dumps({'pages':[
                {'image':'image.png','text':'First dose 25 mg. Second dose 50 mg.','lang':'eng'}]}))
            with mock.patch.object(benchmark_ocr,'_recognize',
                    return_value='First dose 50 mg. Second dose 25 mg.'):
                result=benchmark_ocr.evaluate(root/'manifest.json','tesseract')
        self.assertEqual(result['numeric_accuracy'],0)

    def test_review_score_is_bound_to_exact_generated_output(self):
        import hashlib
        from scripts import benchmark_translation
        from modules.translator import TranslationResult
        output='नमस्ते'
        result=TranslationResult(output,'hi',True,'synthetic',validation={'passed':True})
        row={'id':'1','src':'en','dest':'hi','source':'hello','references':[output],
             'human_adequate':True,'reviewed_output_sha256':'wrong-output'}
        with mock.patch.object(benchmark_translation.translator,'translate_ex',return_value=result):
            unbound=benchmark_translation.run([row],False,.5)[0]
            row['reviewed_output_sha256']=hashlib.sha256(output.encode()).hexdigest()
            bound=benchmark_translation.run([row],False,.5)[0]
        self.assertIsNone(unbound['human_adequate'])
        self.assertTrue(bound['human_adequate'])

    def test_two_store_instances_do_not_lose_each_others_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            stores=[VectorStore(directory,dimension=3,model_id='same') for _ in range(2)]
            barrier=threading.Barrier(2)
            errors=[]
            def write(index):
                try:
                    barrier.wait(timeout=2)
                    stores[index].add_text(f'document {index}',np.eye(3)[index])
                except BaseException as error:
                    errors.append(error)
            threads=[threading.Thread(target=write,args=(i,)) for i in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(3)
            self.assertEqual(errors,[])
            restored=VectorStore(directory,dimension=3,model_id='same')
            self.assertEqual(set(restored.metadata),{'document 0','document 1'})


if __name__=='__main__':
    unittest.main()
