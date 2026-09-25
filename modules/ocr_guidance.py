"""Local page framing and stable, rate-limited spoken OCR guidance.

Heuristics deliberately require a visible page boundary and text-like marks;
a missing boundary is not evidence of the direction of an off-screen page.
"""
from dataclasses import dataclass
import time

import cv2
import numpy as np

DEFAULTS = {
    "ocr_guidance_enabled": True,
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
        if page is None:
            margin = number(cfg, "ocr_guidance_edge_margin", .005, .1)
            if len(marks) >= 8 and any(x < w * margin or y < h * margin or
                    x + bw > w * (1-margin) or y + bh > h * (1-margin)
                    for x, y, bw, bh, _ in marks):
                return Assessment("clipped", "Text may be cut off. Move the camera farther away.")
            return Assessment("search", "I cannot find the whole page. Slowly move the camera to locate it.")
        x, y, bw, bh = cv2.boundingRect(page)
        margin = number(cfg, "ocr_guidance_edge_margin", .005, .1)
        if min(x / w, y / h, (w-x-bw) / w, (h-y-bh) / h) < margin:
            return Assessment("clipped", "Page edges may be cut off. Move the camera farther away.")
        dx, dy = (x + bw/2) / w - .5, (y + bh/2) / h - .5
        tolerance = number(cfg, "ocr_guidance_center_tolerance", .03, .3)
        if max(abs(dx), abs(dy)) > tolerance:
            direction = ("right" if dx > 0 else "left") if abs(dx) >= abs(dy) else ("down" if dy > 0 else "up")
            return Assessment(direction, f"Move the camera slightly {direction}.")
        if max(bw/w, bh/h) < number(cfg, "ocr_guidance_min_fill", .2, .9):
            return Assessment("closer", "Move the camera closer.")
        inside = [s for s in marks if x < s[0] < x+bw and y < s[1] < y+bh]
        if len(inside) < 8:
            return Assessment("text", "I cannot see clear text. Check that the camera faces the printed side.")
        if motion > number(cfg, "ocr_guidance_max_motion", 1, 80):
            return Assessment("moving", "Hold the camera steady.")
        # Exclude the strong page boundary from the sharpness estimate.
        roi = gray[y+5:y+bh-5, x+5:x+bw-5]
        if cv2.Laplacian(roi, cv2.CV_64F).var() < number(cfg, "ocr_guidance_min_sharpness", 1, 1000):
            return Assessment("blur", "Text is blurry. Hold steady while the camera focuses.")
        return Assessment("ready", "Page positioned. Hold steady.")


class GuidanceGate:
    def __init__(self, config):
        self.required = int(number(config, "ocr_guidance_stable_frames", 2, 20))
        self.interval = number(config, "ocr_guidance_prompt_interval_s", 1, 15)
        self.code = None
        self.count = 0
        self.last_spoken = -float("inf")

    def update(self, assessment, now):
        self.count = self.count + 1 if assessment.code == self.code else 1
        self.code = assessment.code
        ready = assessment.code == "ready" and self.count >= self.required
        speak = self.count >= 2 and now - self.last_spoken >= self.interval
        if assessment.code == "ready":
            speak = ready
        if speak:
            self.last_spoken = now
        return ready, speak


def guided_capture(camera, speak, stop_check, config):
    """Return (full-resolution frame, status); never OCR a rejected capture."""
    analyzer = FrameAnalyzer(config)
    gate = GuidanceGate(config)
    stopped = stop_check or (lambda: False)
    speak("Point the camera toward the page. Keep the page still and move the camera when instructed.")
    deadline = time.monotonic() + number(config, "ocr_guidance_timeout_s", 10, 300)
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
            ready, announce = gate.update(assessment, time.monotonic())
            if announce:
                speak(assessment.message)
            if stopped():
                return None, "OCR scan cancelled."
            if ready:
                camera.stop_preview()
                frame = camera.capture(stop_check=stopped)
                if stopped():
                    return None, "OCR scan cancelled."
                if frame is None:
                    return None, "Image capture failed. Please try again."
                # Still and preview can have different crops. Check the actual
                # still, independently of motion between different resolutions.
                check = FrameAnalyzer(config).assess(frame)
                if check.code == "ready":
                    speak("Reading now.")
                    return frame, ""
                speak(check.message)
                analyzer = FrameAnalyzer(config)
                gate = GuidanceGate(config)
                if not camera.start_preview():
                    return None, "OCR guidance unavailable. Camera preview could not restart."
            time.sleep(.15)
        return None, "OCR positioning timed out."
    finally:
        camera.stop_preview()
