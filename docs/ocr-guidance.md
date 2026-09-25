# Spoken OCR positioning

OCR mode now guides positioning before capture, then uses the existing OCR and
private/public document playback. Guidance runs locally using OpenCV; it does
not send preview frames to a cloud service. Existing OCR cloud settings still
apply to the accepted photograph.

Keep the page still and move the camera as instructed. Directions describe
camera movement in the configured upright, unmirrored view. A page to the right
of that view produces “Move the camera slightly right.” When no page boundary
can be located, guidance asks the user to search rather than inventing a direction.

Press Button 3 three times (or the configured voice stop count) to cancel.
On a development terminal, Ctrl-C cancels positioning. Each invocation makes one
scan attempt. Timeout or a rejected still returns to the menu, without restarting
preview or offering an automatic retry loop. Select OCR again to scan another page.
An explanation runs only after an explicit yes; silence does not start another task.

Slow image capture and OCR operations speak a status update after 8 seconds,
then at most one more after another 20 seconds. Completion stops updates before
document playback. An online failure is announced as a service failure, followed
by an explanation that local processing may take longer. An unreadable response
is described separately from a service failure. Progress messages contain no
recognized document text; normal privacy routing still controls document playback.

A capture subprocess or in-flight OCR request must finish or hit its existing
timeout before cancellation returns; cancelled results are discarded and further
status speech is suppressed.

## Camera support

Pi preview uses `rpicam-vid` (or `libcamera-vid`) MJPEG stdout, continuous focus on
rpicam, and a background reader that retains only the latest frame. It stops
before `rpicam-still` captures full-resolution photos. USB preview similarly
drains frames continuously, then stops before still capture. No picamera2 Python
dependency is added. Missing or stalled preview produces a spoken error.
The stream options follow the [Raspberry Pi camera documentation](https://www.raspberrypi.com/documentation/computers/camera_software.html).

The final still is checked again because video and still sensor modes can have
different fields of view. Readable text is accepted even when the paper touches
the frame boundary or its outline cannot be found. Only several text-like marks
near an edge trigger a possible clipping warning; paper edges alone do not.
Each corrective instruction is spoken once per positioning attempt.

After about 10 seconds, the scanner attempts full-resolution OCR on a steady,
adequately lit view even when framing or text-shape detection remains uncertain.
OCR determines whether text is actually readable. The final still's framing alone
cannot restart guidance when text-like detail is present. This prioritizes reading
the visible text; it does not guarantee that the complete page is in view.
Dark, moving or detected blurry previews still require correction. Short prompts
play serially, so old corrections do not queue up.

## Configuration

Defaults apply even when existing `config/settings.json` lacks these keys. Both
example settings files include all options. Most useful settings:

| Key | Default | Purpose |
| --- | --- | --- |
| `ocr_guidance_enabled` | true | Enable guided positioning in OCR mode |
| `ocr_guidance_auto_capture_s` | 10 | Try OCR after brief guidance despite uncertain framing |
| `ocr_guidance_rotation` | 0 | Clockwise mount correction: 0, 90, 180, 270 |
| `ocr_guidance_mirror` | false | Undo a mirrored camera view after rotation |
| `ocr_guidance_timeout_s` | 60 | Positioning time before returning to menu |
| `ocr_guidance_prompt_interval_s` | 3 | Minimum interval between corrective prompts |
| `ocr_guidance_stable_frames` | 4 | Consecutive acceptable observations before capture |
| `ocr_guidance_center_tolerance` | 0.12 | Allowed center offset relative to frame dimensions |
| `ocr_guidance_min_fill` | 0.55 | Minimum page extent on its larger normalized axis |
| `ocr_guidance_edge_margin` | 0.025 | Required margin around page boundary |
| `ocr_guidance_min_brightness` | 55 | Minimum mean grayscale brightness, 0–255 |
| `ocr_guidance_min_sharpness` | 35 | Minimum page-interior Laplacian variance |
| `ocr_guidance_max_motion` | 9 | Maximum mean difference between preview samples |

Rotation and mirroring correct guidance coordinates, not the saved image; OCR
retains its existing orientation handling.

## Device acceptance checks

With the intended mount and a sighted tester, verify all four camera movement
instructions reduce the error. Check near/far positioning, a clipped page, low
light, movement, blur, and a page completely outside the frame. Confirm a stable
page captures automatically and the captured photograph contains all its edges.
Repeat with portrait/landscape pages, small print and different backgrounds.
Check Button 3 cancellation during prompts, positioning and still capture; then
check timeout returns to menu, unplugged camera, and confidential document playback.

These are conservative page/text-shape heuristics, not a trained document detector.
Blank rectangles do not trigger immediate capture, but the bounded fallback may
attempt OCR on them; textured rectangles may resemble text. Boundaries can be missed on low-contrast backgrounds, curved books, strong
perspective, or with multiple pages. Complete-page framing also cannot guarantee
that very small print is legible. The actual Pi camera and mounting must be used
to tune thresholds and validate usability with a blind user before deployment.
