#!/usr/bin/env python3
"""Readiness report and opt-in audio/API probes for the live interpreter.

Default operation enumerates devices but never records or contacts the API.
--probe-api sends only synthetic silence, never microphone or saved speech.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules import live_translator as live


async def probe_api(interpreter):
    from modules.privacy_policy import current_policy
    if not current_policy().allow_cloud:
        raise live.LiveTranslationError("Cloud probing is unavailable in private mode.")
    client = live.genai.Client(api_key=interpreter.api_key)
    connection = client.aio.live.connect(model=interpreter.model, config=interpreter._build_config())
    entered = False
    try:
        session = await asyncio.wait_for(connection.__aenter__(), interpreter.connect_timeout)
        entered = True
        # Some setup errors surface only after the first audio packet. Listen
        # concurrently while sending one second of synthetic 16 kHz silence.
        async def receive():
            async for response in session.receive():
                if getattr(response, "go_away", None):
                    raise live.LiveTranslationError("The live service requested disconnection.")
        receiver = asyncio.create_task(receive())
        try:
            for _ in range(10):
                await asyncio.wait_for(session.send_realtime_input(audio=live.types.Blob(
                    data=bytes(3200), mime_type="audio/pcm;rate=16000")), interpreter.send_timeout)
                await asyncio.sleep(0.1)
            await asyncio.sleep(1)
            if receiver.done():
                receiver.result()
                raise live.LiveTranslationError("The service closed the audio stream during the probe.")
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
    finally:
        if entered:
            try:
                await asyncio.wait_for(connection.__aexit__(None, None, None), 3)
            except Exception:
                pass
        try:
            await asyncio.wait_for(client.aio.aclose(), 3)
        finally:
            client.close()


def probe_audio(interpreter):
    """Open negotiated devices; record one second and play a quiet test tone."""
    import numpy as np
    if not interpreter._check_audio_devices():
        raise live.LiveTranslationError(interpreter.last_error)
    incoming, outgoing = interpreter.input_format, interpreter.output_format
    with live.sd.RawInputStream(device=incoming.device, channels=incoming.channels,
                                samplerate=incoming.rate, dtype="int16") as stream:
        recorded, overflow = stream.read(incoming.rate)
    samples = np.frombuffer(recorded, dtype="<i2").astype(np.float32)
    print(json.dumps({"microphone_peak":int(np.max(np.abs(samples))),
                      "microphone_rms":float(np.sqrt(np.mean(samples*samples))),
                      "input_overflow":bool(overflow)}))
    tone = (np.sin(2*np.pi*440*np.arange(6000)/24000) * 2000).astype("<i2").tobytes()
    from modules.live_audio import PlaybackConverter
    with live.sd.RawOutputStream(device=outgoing.device, channels=outgoing.channels,
                                 samplerate=outgoing.rate, dtype="int16") as stream:
        stream.write(PlaybackConverter(outgoing).convert(tone))
    if overflow:
        raise live.LiveTranslationError("Microphone probe overflowed. Reduce device load and retry.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--probe-audio", action="store_true", help="record one second and play a quiet tone")
    parser.add_argument("--probe-api", action="store_true", help="connect to Gemini and send synthetic silence")
    parser.add_argument("--pair", choices=live.LANGUAGE_PAIRS, default="en-gu")
    args = parser.parse_args()
    interpreter = live.LiveTranslator(args.pair)
    info = interpreter.diagnostics()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if args.list_devices and live.SD_AVAILABLE:
        print(live.sd.query_devices())
    try:
        if args.probe_audio:
            if not live.SD_AVAILABLE:
                raise live.LiveTranslationError("Install sounddevice and PortAudio.")
            probe_audio(interpreter)
            print("Audio capture and test-tone playback completed. Confirm the tone was audible.")
        if args.probe_api:
            if not live.GENAI_AVAILABLE or not interpreter.api_key:
                raise live.LiveTranslationError("Install google-genai and configure a Gemini API key.")
            started = time.monotonic()
            asyncio.run(probe_api(interpreter))
            print(f"Gemini accepted the live connection and synthetic PCM audio ({time.monotonic()-started:.1f}s).")
            print("This checks connectivity, not translation accuracy.")
    except Exception as exc:
        print(str(live._service_error(exc)), file=sys.stderr)
        return 1
    return 0 if info["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
