# Speech noise filter

Restart the application after changing `config/settings.json`:

```json
"tts_noise_filter_enabled": true,
"tts_gain": 1.0,
"tts_normalize_enabled": false
```

The filter reduces frequencies below 80 Hz and above 8 kHz in generated
speech, with a lower upper cutoff when the source sample rate requires it.
Five-millisecond fades soften start/end discontinuities. It uses the existing
NumPy/SciPy dependencies and runs locally before speaker/earphone routing.
MP3 speech is decoded to PCM through the existing decoder chain first.

The lower gain and disabled normalization avoid boosting the signal during
the initial noise comparison. Adjust physical speaker volume gradually.
Set `tts_noise_filter_enabled` to `false` and restart to compare with the
original audio. The filter slightly changes voice tone; it does not remove
all noise within the speech frequency range.

If decoding or filtering is unavailable, a warning is logged and speech
continues through the existing playback chain. Direct-engine emergency speech
and the separate live-translation stream bypass this generated-file filter.

Noise that continues with the application stopped needs investigation of the
speaker, amplifier, power supply, wiring or audio-device monitoring. Software
filtering cannot remove electrical noise introduced after playback.
