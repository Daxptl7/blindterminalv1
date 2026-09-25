"""Local page framing and stable, rate-limited spoken OCR guidance.

Framing is advisory: page edges are not text edges. A bounded guidance
period is followed by a still-image OCR attempt instead of endless corrections.
"""
from dataclasses import dataclass
import time

import cv2
import numpy as np

DEFAULTS = {
    "ocr_guidance_enabled": True,
    "ocr_guidance_auto_capture_s": 10,
    "ocr_guidance_timeout_s": 60,
    "ocr_guidance_prompt_interval_s": 3,
    "ocr_guidance_stable_frames": 4,
    "ocr_guidance_rotation": 0,
    "ocr_guidance_mirror": False,
    "ocr_guidance_center_tolerance": 0.12,
    "ocr_guidance_min_fill": 0.55,
    "ocr_guidance_edge_margin": 0.025,
    "ocr_guidance_min_brightness": 55,
    "ocr_guidance_min_sharpness": 35,
    "ocr_guidance_max_motion": 9,
}


@dataclass(frozen=True)
class Assessment:
    code: str
    message: str
    readable: bool = False


def number(config, key, low, high):
    try:
        value = float(config.get(key, DEFAULTS[key]))
        if not np.isfinite(value):
            raise ValueError()
    except (TypeError, ValueError):
        value = float(DEFAULTS[key])
    return max(low, min(high, value))


class FrameAnalyzer:
    def __init__(self, config=None):
        self.config = config or {}
        self.previous = None

    def assess(self, frame):
        cfg = self.config
        rotation = int(number(cfg, "ocr_guidance_rotation", 0, 270))
        frame = np.rot90(frame, -(rotation // 90))
        if cfg.get("ocr_guidance_mirror", False):
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(cv2.resize(frame, (int(w * 640 / max(h, w)),
                                                     int(h * 640 / max(h, w)))), cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        small = cv2.resize(gray, (160, 120))
        motion = 0 if self.previous is None else float(np.mean(cv2.absdiff(small, self.previous)))
        self.previous = small
        if np.mean(gray) < number(cfg, "ocr_guidance_min_brightness", 1, 200):
            return Assessment("dark", "The page is too dark. Add more light.")

        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        page = None
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(contour) < h * w * 0.06:
                break
            polygon = cv2.approxPolyDP(contour, 0.025 * cv2.arcLength(contour, True), True)
            if len(polygon) == 4 and cv2.isContourConvex(polygon):
                page = polygon
                break

        # Text-like connected components, not an OCR transcription. This also
        # prevents blank rectangular objects from triggering automatic capture.
        ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 12)
        count, _, stats, _ = cv2.connectedComponentsWithStats(ink)
        marks = [s for s in stats[1:count] if 2 <= s[2] <= w * .08
                 and 3 <= s[3] <= h * .10 and 4 <= s[4] <= h * w * .003]
        if motion > number(cfg, "ocr_guidance_max_motion", 1, 80):
            return Assessment("moving", "Hold the camera steady.")
        # A paper edge near the border is normal. Only multiple text-like
        # components near that border justify a possible clipping warning.
        margin = number(cfg, "ocr_guidance_edge_margin", .005, .1)
        edge_marks = [s for s in marks if s[0] < w * margin or s[1] < h * margin
                      or s[0]+s[2] > w*(1-margin) or s[1]+s[3] > h*(1-margin)]
        clipped_text = len(edge_marks) >= max(4, len(marks) * .15)
        if page is None:
            x, y, bw, bh = 0, 0, w, h
        else:
            x, y, bw, bh = cv2.boundingRect(page)
        inside = [s for s in marks if x < s[0] < x+bw and y < s[1] < y+bh]
        if len(inside) < 8:
            return Assessment("search", "Point the camera toward the printed text and hold steady.")
        roi = gray[y+5:y+bh-5, x+5:x+bw-5]
        if roi.size and cv2.Laplacian(roi, cv2.CV_64F).var() < number(cfg, "ocr_guidance_min_sharpness", 1, 1000):
            return Assessment("blur", "Text is blurry. Hold steady while the camera focuses.")
        if clipped_text:
            return Assessment("clipped", "Some text may be cut off. Move back slightly if possible.", True)
        if page is None:
            return Assessment("ready", "Text found. Hold steady.", True)

        dx, dy = (x + bw/2) / w - .5, (y + bh/2) / h - .5
        tolerance = number(cfg, "ocr_guidance_center_tolerance", .03, .3)
        if max(abs(dx), abs(dy)) > tolerance:
            direction = ("right" if dx > 0 else "left") if abs(dx) >= abs(dy) else ("down" if dy > 0 else "up")
            return Assessment(direction, f"Move the camera slightly {direction}.", True)
        if max(bw/w, bh/h) < number(cfg, "ocr_guidance_min_fill", .2, .9):
            return Assessment("closer", "Move the camera closer.", True)
        return Assessment("ready", "Text found. Hold steady.", True)


class GuidanceGate:
    def __init__(self, config):
        self.required = int(number(config, "ocr_guidance_stable_frames", 2, 20))
        self.interval = number(config, "ocr_guidance_prompt_interval_s", 1, 15)
        self.code = None
        self.count = 0
        self.last_spoken = -float("inf")
        self.spoken_codes = set()

    def update(self, assessment, now):
        self.count = self.count + 1 if assessment.code == self.code else 1
        self.code = assessment.code
        ready = assessment.code == "ready" and self.count >= self.required
        speak = (self.count >= 2 and now - self.last_spoken >= self.interval
                 and assessment.code not in self.spoken_codes)
        if assessment.code == "ready":
            speak = ready
        if speak:
            self.last_spoken = now
            self.spoken_codes.add(assessment.code)
        return ready, speak


def guided_capture(camera, speak, stop_check, config):
    """Guide briefly, then let full-resolution OCR judge uncertain framing."""
    analyzer = FrameAnalyzer(config)
    gate = GuidanceGate(config)
    stopped = stop_check or (lambda: False)
    speak("Point the camera toward the page. Keep the page still and move the camera when instructed.")
    started = time.monotonic()
    deadline = started + number(config, "ocr_guidance_timeout_s", 10, 300)
    auto_after = min(number(config, "ocr_guidance_auto_capture_s", 2, 30),
                     (deadline - started) * .6)
    quiet_frames = 0
    missing_since = None
    try:
        if not camera.start_preview():
            return None, "OCR guidance unavailable. Camera preview could not start. Please check the camera and try again."
        while time.monotonic() < deadline:
            if stopped():
                return None, "OCR scan cancelled."
            frame = camera.preview_frame()
            if frame is None:
                now = time.monotonic()
                missing_since = now if missing_since is None else missing_since
                if now - missing_since > 8:
                    return None, "OCR guidance unavailable. Camera preview stopped. Please try again."
                time.sleep(.1)
                continue
            missing_since = None
            assessment = analyzer.assess(frame)
            now = time.monotonic()
            ready, announce = gate.update(assessment, now)
            # Even a visible page can fail contour/text-shape heuristics.
            # After a short window, try OCR on a steady, adequately lit frame.
            quiet_frames = quiet_frames + 1 if assessment.code not in ("dark", "moving", "blur") else 0
            fallback = now - started >= auto_after and quiet_frames >= gate.required
            if fallback and not ready:
                speak("Hold steady. I will try reading this image now.")
                announce = False
                ready = True
            if announce:
                speak(assessment.message)
            if stopped():
                return None, "OCR scan cancelled."
            if ready:
                camera.stop_preview()
                from modules.processing_feedback import run_with_feedback
                frame = run_with_feedback(
                    lambda: camera.capture(stop_check=stopped), speak, stopped,
                    message="Still capturing the image. Please hold steady.")
                if stopped():
                    return None, "OCR scan cancelled."
                if frame is None:
                    return None, "Image capture failed. Please try again."
                # Still and preview can have different crops. Check the actual
                # still, independently of motion between different resolutions.
                check = FrameAnalyzer(config).assess(frame)
                if check.readable or check.code == "ready" or (
                    fallback and check.code not in ("dark", "blur", "moving")
                ):
                    if check.code == "clipped":
                        speak("Reading the visible text. Some text may be outside the image.")
                    speak("Reading now.")
                    return frame, ""
                return None, "OCR guidance unavailable. I could not capture clear text. Returning to the menu."
            time.sleep(.15)
        return None, "OCR positioning timed out."
    finally:
        camera.stop_preview()
