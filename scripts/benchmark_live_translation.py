#!/usr/bin/env python3
"""Translate prerecorded test utterances through the real Gemini Live API.

Sends the WAV files listed in the manifest to Gemini. No microphone access.
Saves translated audio and transcripts for native-speaker review. Each file
must contain a single utterance (PCM16, mono/stereo, no longer than 30 seconds).
"""
import argparse
import asyncio
import hashlib
import json
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from modules import live_translator as live
from modules.live_audio import microphone_pcm
from modules.privacy_policy import current_policy
from modules.translator import validate_translation
from scripts.benchmark_translation import chrf


def load_clip(path):
    with wave.open(str(path), "rb") as source:
        rate, channels = source.getframerate(), source.getnchannels()
        if source.getsampwidth() != 2 or channels not in (1, 2):
            raise ValueError("Use PCM16 mono/stereo WAV files")
        duration = source.getnframes() / rate
        if not 0 < duration <= 30 or rate not in (16000, 24000, 44100, 48000):
            raise ValueError("Use 16/24/44.1/48 kHz WAV files lasting at most 30 seconds")
        return microphone_pcm(source.readframes(source.getnframes()), channels), rate, duration


async def collect_turn(session, pcm, rate, timeout):
    """Send one test utterance at recording speed and collect one spoken turn."""
    started = time.monotonic()
    result = {"input_transcript":"", "output_transcript":"", "audio":bytearray(),
              "first_audio_s":None, "server_events":0}
    async def send():
        packet_bytes = int(rate * 0.1) * 2
        # Simulate a continuously open microphone, including enough trailing
        # silence for the same automatic VAD used in the live application.
        stream_pcm = bytes(packet_bytes * 3) + pcm + bytes(packet_bytes * 12)
        for offset in range(0, len(stream_pcm), packet_bytes):
            await session.send_realtime_input(audio=live.types.Blob(
                data=stream_pcm[offset:offset+packet_bytes], mime_type=f"audio/pcm;rate={rate}"))
            await asyncio.sleep(0.1)
        await session.send_realtime_input(audio_stream_end=True)
    async def receive():
        while True:
            received = False
            async for response in session.receive():
                received = True
                result["server_events"] += 1
                sc = getattr(response,"server_content",None)
                if sc is None:
                    continue
                for field, key in (("input_transcription","input_transcript"),
                                   ("output_transcription","output_transcript")):
                    transcript = getattr(sc,field,None)
                    if transcript and getattr(transcript,"text",None):
                        result[key] += transcript.text
                turn = getattr(sc,"model_turn",None)
                for part in (getattr(turn,"parts",None) or []):
                    inline = getattr(part,"inline_data",None)
                    if inline and getattr(inline,"data",None):
                        if result["first_audio_s"] is None:
                            result["first_audio_s"] = time.monotonic()-started
                        result["audio"].extend(inline.data)
                        if len(result["audio"]) > 24000 * 2 * 90:
                            raise ValueError("Translated audio exceeded the test limit")
                if getattr(sc,"turn_complete",False) and result["audio"]:
                    return
            if not received:
                raise RuntimeError("Live stream closed without a translated turn")
    async def drain_transcripts():
        # Transcription processing can lag audio completion. Preserve late text
        # rather than declaring a valid spoken response to be empty immediately.
        async for response in session.receive():
            sc = getattr(response,"server_content",None)
            if sc is None:
                continue
            for field,key in (("input_transcription","input_transcript"),
                              ("output_transcription","output_transcript")):
                transcript = getattr(sc,field,None)
                if transcript and getattr(transcript,"text",None):
                    result[key] += transcript.text
    tasks = [asyncio.create_task(send()),asyncio.create_task(receive())]
    try:
        await asyncio.wait_for(asyncio.gather(*tasks),timeout)
        try:
            await asyncio.wait_for(drain_transcripts(),0.5)
        except TimeoutError:
            pass
    except TimeoutError:
        error = live.LiveTranslationError("No complete spoken translation was received before the test timed out.")
        error.diagnostics = {key:value for key,value in result.items() if key != "audio"}
        error.diagnostics["audio_bytes"] = len(result["audio"])
        raise error from None
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
    result["audio"] = bytes(result["audio"])
    result["elapsed_s"] = time.monotonic()-started
    return result


async def translate_clip(interpreter, pcm, rate, duration):
    if not current_policy().allow_cloud:
        raise live.LiveTranslationError("Cloud evaluation is unavailable in private mode.")
    client = live.genai.Client(api_key=interpreter.api_key)
    context = client.aio.live.connect(model=interpreter.model,config=interpreter._build_config())
    entered = False
    try:
        session = await asyncio.wait_for(context.__aenter__(),interpreter.connect_timeout)
        entered = True
        return await collect_turn(session,pcm,rate,duration+2+interpreter.response_timeout)
    finally:
        if entered:
            try:
                await asyncio.wait_for(context.__aexit__(None,None,None),3)
            except Exception:
                pass
        try:
            await asyncio.wait_for(client.aio.aclose(),3)
        finally:
            client.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest",type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    rows=[json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    results=[]
    failed=False
    for index,row in enumerate(rows):
        record={"id":row["id"],"src":row["src"],"dest":row["dest"]}
        try:
            pair=next(pair for pair in live.LANGUAGE_PAIRS
                      if set(pair.split("-")) == {row["src"],row["dest"]})
            interpreter=live.LiveTranslator(pair)
            pcm,rate,duration=load_clip(args.manifest.parent / row["audio"])
            response=asyncio.run(translate_clip(interpreter,pcm,rate,duration))
            audio=response.pop("audio")
            target=args.output_dir / f"{index:04d}-translation.wav"
            with wave.open(str(target),"wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(24000)
                output.writeframes(audio)
            text=response["output_transcript"].strip()
            refs=row["references"]
            if isinstance(refs,str):
                refs=[refs]
            passed,report=validate_translation(row["source_text"],text,row["src"],row["dest"])
            digest=hashlib.sha256(text.encode()+b"\0"+audio).hexdigest()
            record.update(response,model=interpreter.model,output_audio=target.name,
                          output_sha256=digest,chrf=max(chrf(text,ref) for ref in refs),
                          validation=report,source_text=row["source_text"],
                          human_adequate=(row.get("human_adequate")
                            if type(row.get("human_adequate")) is bool
                            and row.get("reviewed_output_sha256") == digest else None))
            failed = failed or not passed
        except Exception as exc:
            record["error"]=str(live._service_error(exc))
            record["error_type"]=type(exc).__name__
            if hasattr(exc,"diagnostics"):
                record["diagnostics"]=exc.diagnostics
            failed=True
        results.append(record)
        (args.output_dir/"results.json").write_text(json.dumps(results,indent=2,ensure_ascii=False))
        print(json.dumps({k:v for k,v in record.items() if k not in ("source_text",)},ensure_ascii=False),flush=True)
    print("Review saved audio with native speakers. chrF and token checks do not establish semantic accuracy.")
    return 1 if failed or not results else 0


if __name__ == "__main__":
    raise SystemExit(main())
