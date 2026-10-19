# Live translation validation record — 19 October 2026

Implementation validation used macOS, Python 3.13, google-genai 2.31.0,
sounddevice 0.5.6, and Gemini `gemini-3.8-live`. Raspberry Pi hardware was not
available in this workspace.

## Automated and installation checks

- 36 live translation/audio/benchmark tests passed, including native USB rate
  fallback, stereo conversion, index zero, two turns in one session, repeated
  sessions, speech boundaries, silence, queue bounds, cancellation, API errors,
  private-mode rejection (in the release suite), and the main menu error path.
- The installation-layout check and changed-file whitespace check passed.
- Before pushing, the isolated staged tree executed all 233 regression tests:
  232 passed and one was skipped. The installation-layout check also passed.
  The development checkout's earlier ten OCR failures belonged to separate,
  uncommitted OCR edits, which are excluded from this Voice Ask/live update.
- The local virtual environment's `pip check` still reports existing unrelated
  missing PyObjC optional frameworks and a Hugging Face Hub/Click version
  conflict. This development environment is not a validated Pi release image.

## Real API speech smoke check

Three gTTS-generated clips expressed “Do not open the door” in English, Hindi,
and Gujarati. Six separate live sessions tested the six directions. The
benchmark streamed PCM at recording speed with initial context and trailing
silence; it saved actual 24 kHz translated audio. All six returned non-silent
audio. Google speech recognition was then used as an independent automated
check of the generated audio, not as a native-speaker review.

| Direction | First audio from clip start | Independent recognition of the translated audio |
|---|---:|---|
| English → Hindi | 3.07 s | दरवाजा मत खोलिए |
| English → Gujarati | 3.13 s | દરવાજો ખોલશો નહીં |
| Hindi → English | 3.42 s | do not open the door |
| Hindi → Gujarati | 3.64 s | અને દરવાજો ના ખોલશો |
| Gujarati → English | 3.61 s | do not open the door |
| Gujarati → Hindi | 3.88 s | दरवाजा मत खोलिए |

Negation was preserved in all six recognized audio outputs. Hindi → Gujarati
included an extra conjunction (“and”), so the check does not establish exact
word-for-word fidelity. The Gemini output transcript was empty for one case and
included extra annotation text for another, despite recognizable translated
speech. The benchmark preserves those raw transcripts and reports their
validation failures rather than silently treating them as clean text.

First-audio latency starts at the beginning of a supplied clip, includes
streaming/warm-up time, and excludes actual speaker playback. It is not the
end-of-utterance latency perceived by a person using the Pi.

## Remaining release evidence

This is a six-case synthetic smoke check, not a percentage accuracy result.
No human adequacy scores were assigned. Use the recorded-speech benchmark in
[LIVE_TRANSLATION_SETUP.md](LIVE_TRANSLATION_SETUP.md) with native speakers,
real Pi microphone recordings, noise, accents, numbers, names, questions,
negation and mixed-language speech. Complete the existing six-direction release
gate and a sustained Pi microphone/speaker/network/cancellation trial before
claiming device readiness or 95% semantic accuracy.
