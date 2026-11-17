# Speech noise filter

The filter is disabled by default while the reported continuous tone is being
diagnosed. To disable it in an existing installation, edit
`config/settings.json` and restart the application:

```json
"tts_noise_filter_enabled": false,
"tts_gain": 1.0,
"tts_normalize_enabled": false
```

The filter reduces frequencies below 80 Hz and above 8 kHz in generated
speech, with a lower upper cutoff when the source sample rate requires it.
Five-millisecond fades soften start/end discontinuities. It uses the existing
NumPy/SciPy dependencies and runs locally before speaker/earphone routing.
MP3 speech is decoded to PCM through the existing decoder chain first.
Filtered peaks are limited to the original signal peak with 5% PCM headroom,
using attenuation rather than clipping to avoid adding transient distortion.

The lower gain and disabled normalization avoid boosting the signal during
the initial noise comparison. Adjust physical speaker volume gradually.
Set `tts_noise_filter_enabled` to `true` only for a controlled comparison once
the continuous tone has been isolated. The filter slightly changes voice tone; it does not remove
all noise within the speech frequency range.

If decoding or filtering is unavailable, a warning is logged and speech
continues through the existing playback chain. Direct-engine emergency speech
and the separate live-translation stream bypass this generated-file filter.

Noise that continues with the application stopped needs investigation of the
speaker, amplifier, power supply, wiring or audio-device monitoring. Software
filtering cannot remove electrical noise introduced after playback.
