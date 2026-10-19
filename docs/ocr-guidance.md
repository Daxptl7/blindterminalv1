# OCR: documents and phone screens

Public OCR tries Gemini first with the configured key, then falls back to the
local reader when the request fails or no text is found. Private OCR stays local
and announces this before reading. Preview guidance always runs locally.

## Spoken scan flow

1. Hold the page or phone still. Turn on the phone screen and stop scrolling.
2. Move the camera in the direction spoken. Only one correction is given at a
   time: light, movement, framing, distance, or focus. Directions describe camera
   movement in the configured upright view. An uncertain view gets a search
   instruction rather than an invented direction.
3. A stable view must remain in position for at least one second before capture.
   Corrections can repeat after ten seconds, up to three times per continuous
   instruction. Search/focus reminders include screen reflections and camera angle.
   If only framing remains uncertain while sharp, steady text stays visible for
   eight seconds, the scanner announces that it will capture the visible text.
   Darkness, motion, blur or missing text resets this window.
4. The device says to hold still while taking the photograph. It selects the
   sharpest of the configured stills and checks the final photograph. A blurry,
   dark or unrecognized still gets one guided recapture by default. Possible
   edge clipping is announced, and the visible text is passed to the reader.
5. “Picture taken” means the user can relax their hand. Online/local reading
   progress continues separately. The camera burst does not control upload time.
6. Document playback and the optional explanation keep the existing privacy and
   button controls. Button 3's configured stop count cancels scanning.

There is no forced capture of dark, moving, blurry or unrecognized views. A
positioning timeout gives an actionable instruction and returns to the menu.
The framing grace period requires consecutive readable preview observations;
elapsed time alone cannot trigger a capture. The old
`ocr_guidance_auto_capture_s` setting is ignored.

The analyzer uses page/screen bounds or visible character-like marks, supports
both light and dark screen themes, and measures illumination in the target region
instead of averaging a bright screen with its dark surroundings. Text-sized
patches are used for sharpness so a sharp bezel cannot by itself certify focus.
Character-like marks need a similar-sized neighbor on the same text line, and
page membership uses the detected polygon rather than its enclosing rectangle.
A few marks near an image edge no longer mean cut-off text: the warning requires
several supported marks touching the outer image boundary. A fully visible page
outline also prevents background marks from causing that warning.
These are framing heuristics, not a guarantee of complete or accurate transcription.

## Gemini requests and fallback

Only the accepted photograph is uploaded. A separate upload copy is resized to a
maximum 3200-pixel long edge, JPEG compressed and limited to 2 MB by default. The
original still remains full resolution for local OCR. A requested PNG is used
only when it fits the byte budget; otherwise JPEG is used. Upload copies stay in
memory. The image is sent inline in one request, following Google's
[image input documentation](https://ai.google.dev/gemini-api/docs/image-understanding).

Each online attempt has a 60-second request timeout, with at most two attempts
inside a 95-second overall online budget. The caller stops waiting at that budget
plus five seconds for preparation/cleanup; an in-flight HTTP call may finish later
but cannot announce its result. Cancellation discards its result and stops retries.
SDK retries are disabled so they cannot silently multiply the application retry
count; timeouts use milliseconds as described by the
[Google Gen AI SDK](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpOptions).

Only connection/timeouts and selected temporary server errors get a retry.
A model-specific rejection of the optional media-resolution setting gets one
retry without that setting. Bad credentials, unavailable models and exhausted
quota go directly to local fallback, each with a different spoken explanation.
A truncated/blocked response is not passed off as a complete document.

Short readings, including one-word labels, are valid. The old minimum-word and
image-area coverage rules are removed: a 12-megapixel image can contain just one
word. Phrases such as “I cannot attend” inside a real phone message are no longer
mistaken for an OCR failure. The old `ocr_gemini_min_words` and
`ocr_gemini_coverage_check_min_pixels` settings are ignored.

Logs report request model, upload dimensions/bytes, elapsed time, attempt number,
HTTP status and failure category. They do not log the image, transcription, API
key or provider error body. These categories distinguish a connection problem
from credentials, quota, model and request-settings failures on the device.

## Settings

### Direction and movement reference

The scanner says left, right, up or down and asks the user to pause after each
movement. These describe sliding the camera parallel to the text: up means
toward the top of the text, down toward its bottom. Closer/farther instructions
refer separately to camera distance. The opening prompt explains this reference.

By default the movement is “one small step”; the camera cannot measure absolute
centimetres from an unknown page or phone. For approximate centimetre prompts,
configure the actual target dimensions in the device's `config/settings.json`.
For an upright portrait A4 page, for example:

```json
"ocr_guidance_target_width_cm": 21,
"ocr_guidance_target_height_cm": 29.7
```

For landscape orientation swap those dimensions. For a phone use the dimensions
of the detected screen/outline, not the A4 values. Leave both settings at zero
when switching between targets of unknown size. The example is a reference setup,
not an assumption made automatically for every scan.

When a matching, complete, nearly parallel outline is visible, the scanner
estimates the image offset using those dimensions and says, for example,
“Move the camera about 3 centimetres left. Keep it facing the text, then pause
for guidance.” Steps are limited to 1–5 centimetres before observing again.
When the outline is missing, clipped, tilted or inconsistent with the configured
dimensions, it falls back to small-step guidance. Distances are estimates and
depend on recognizing the correct target outline.

New defaults apply when keys are absent. Existing explicit settings override them;
in particular, change an existing `ocr_gemini_timeout_s: 20` to `60` to get the longer
request window. Keep API keys in the device's untracked `config/settings.json`.
Do not replace that file with an example. Environment keys `GEMINI_API_KEY` and
`GOOGLE_API_KEY` are also supported if the settings key is empty.

| Setting | Default | Meaning |
| --- | --- | --- |
| `ocr_gemini_timeout_s` | 60 | Timeout for each HTTP request, seconds |
| `ocr_gemini_total_timeout_s` | 95 | Total online processing budget, seconds |
| `ocr_gemini_attempts` | 2 | Maximum application attempts, including first request |
| `ocr_gemini_max_image_edge` | 3200 | Maximum upload long edge, pixels |
| `ocr_gemini_max_image_bytes` | 2000000 | Maximum encoded upload image bytes |
| `ocr_gemini_image_quality` | 92 | Initial JPEG quality, reduced no lower than 85 |
| `ocr_gemini_media_resolution` | high | Optional model image-detail setting |
| `ocr_gemini_model_name` | empty in code | OCR-specific model override; otherwise uses `gemini_model_name` |
| `ocr_guidance_timeout_s` | 60 | Positioning window per capture attempt |
| `ocr_guidance_hold_seconds` | 1.0 | Minimum stable time before capture |
| `ocr_guidance_framing_grace_s` | 8.0 | Readable preview window before capturing despite uncertain framing |
| `ocr_guidance_stable_frames` | 4 | Minimum consecutive ready observations |
| `ocr_guidance_prompt_interval_s` | 3 | Minimum interval between different corrections |
| `ocr_guidance_repeat_interval_s` | 10 | Reminder interval for a persistent correction |
| `ocr_guidance_capture_attempts` | 2 | Maximum still capture attempts per scan |
| `ocr_guidance_target_width_cm` | 0 | Known target width; zero disables centimetre guidance |
| `ocr_guidance_target_height_cm` | 0 | Known target height; must also be set for centimetre guidance |
| `ocr_guidance_rotation` | 0 | Clockwise mount correction: 0, 90, 180, 270 |
| `ocr_guidance_mirror` | false | Undo a mirrored camera image after rotation |

The example files use the image-capable
[`gemini-3.5-flash-lite`](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)
for OCR. Existing device model choices remain in effect unless the OCR-specific
override is set. Rotation and mirror corrections now apply to both guidance and
the photograph passed to the readers.

Pi preview uses `rpicam-vid`/`libcamera-vid` with the configured still aspect ratio.
It stops before `rpicam-still` takes full-resolution photographs. Still capture
keeps ownership on the scan thread; cancellation terminates the Pi capture process
before release or another scan. USB reads depend on the camera driver's read timeout.

