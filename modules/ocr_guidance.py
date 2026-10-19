"""Local framing guidance for printed pages and phone screens.

Preview analysis never transcribes or uploads text. Capture requires steady,
visible, sharp text; uncertain framing gets a bounded correction window.
"""
from dataclasses import dataclass
from bisect import bisect_left, bisect_right
import time

import cv2
import numpy as np

DEFAULTS = {
    "ocr_guidance_enabled": True,
    "ocr_guidance_timeout_s": 60,
    "ocr_guidance_prompt_interval_s": 3,
    "ocr_guidance_repeat_interval_s": 10,
    "ocr_guidance_stable_frames": 4,
    "ocr_guidance_hold_seconds": 1.0,
    "ocr_guidance_framing_grace_s": 8.0,
    "ocr_guidance_capture_attempts": 2,
    "ocr_guidance_rotation": 0,
    "ocr_guidance_mirror": False,
    "ocr_guidance_center_tolerance": 0.12,
    "ocr_guidance_target_width_cm": 0,
    "ocr_guidance_target_height_cm": 0,
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


def orient_frame(frame, config):
    """Use the same mount correction for preview and the OCR photograph."""
    rotation = int(number(config, "ocr_guidance_rotation", 0, 270))
    frame = np.rot90(frame, -(rotation // 90))
    if config.get("ocr_guidance_mirror", False):
        frame = cv2.flip(frame, 1)
    return np.ascontiguousarray(frame)


def _text_marks(gray):
    """Find small character-like marks in both light and dark screen themes."""
    h, w = gray.shape
    candidates = []
    for source in (gray, 255 - gray):
        ink = cv2.adaptiveThreshold(source, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 12)
        count, _, stats, _ = cv2.connectedComponentsWithStats(ink)
        candidates.extend(s for s in stats[1:count]
                          if 2 <= s[2] <= w * .12 and 3 <= s[3] <= h * .15
                          and 4 <= s[4] <= h * w * .004
                          and .08 <= s[4] / (s[2] * s[3]) <= .95)
    # Desk texture, paper borders and glare also produce small components.
    # Keep marks supported by another similar-sized mark on the same line.
    # Reject duplicate/overlapping detections from the opposite polarity so
    # they cannot certify an isolated speck as a line of text.
    positioned = sorted(
        [(float(s[1]+s[3]/2), int(s[0]), int(s[2]), int(s[3]), s)
         for s in candidates], key=lambda item: item[0])
    centers = [item[0] for item in positioned]
    supported = []
    for cy, x, width, height, mark in positioned:
        # Restrict comparisons to nearby rows to keep preview work bounded
        # when a textured background produces many small components.
        nearby = positioned[bisect_left(centers, cy-height):
                            bisect_right(centers, cy+height)]
        for ocy, ox, ow, oh, _ in nearby:
            if not .5 <= oh / height <= 2:
                continue
            gap = max(ox - (x + width), x - (ox + ow))
            if (0 <= gap <= max(12 * height, w * .04)
                    and abs(cy - ocy) <= max(height, oh) * .5):
                supported.append(mark)
                break
    return supported


def movement_prompt(direction, dx, dy, image_width, image_height, page, config):
    """Estimate a short lateral step only with a known, fully visible target.

    A monocular image has no absolute scale by itself. Configured dimensions
    must describe the actual page/screen in its current upright orientation.
    Reject clipped or strongly tilted outlines rather than invent centimetres.
    """
    width_cm = number(config, "ocr_guidance_target_width_cm", 0, 100)
    height_cm = number(config, "ocr_guidance_target_height_cm", 0, 100)
    fallback = (f"Move the camera one small step {direction}. "
                "Keep it facing the text, then pause for guidance.")
    if page is None or width_cm <= 0 or height_cm <= 0:
        return fallback
    points = np.asarray(page, dtype=np.float32).reshape(4, 2)
    sums, differences = points.sum(axis=1), np.diff(points, axis=1).ravel()
    indices = [np.argmin(sums), np.argmin(differences),
               np.argmax(sums), np.argmax(differences)]
    if len(set(indices)) != 4:
        return fallback
    tl, tr, br, bl = points[indices]
    horizontal = [np.linalg.norm(tr-tl), np.linalg.norm(br-bl)]
    vertical = [np.linalg.norm(bl-tl), np.linalg.norm(br-tr)]
    margin = number(config, "ocr_guidance_edge_margin", .005, .1)
    if (min(horizontal + vertical) < 1
            or min(horizontal)/max(horizontal) < .85
            or min(vertical)/max(vertical) < .85
            or np.any(points[:, 0] <= image_width*margin)
            or np.any(points[:, 0] >= image_width*(1-margin))
            or np.any(points[:, 1] <= image_height*margin)
            or np.any(points[:, 1] >= image_height*(1-margin))):
        return fallback
    scale_x, scale_y = np.mean(horizontal)/width_cm, np.mean(vertical)/height_cm
    if not .85 <= scale_x/scale_y <= 1.18:
        return fallback
    # Reject heavily rotated targets; image left/right must correspond to the
    # physical axes whose dimensions were configured.
    if abs(float(tr[1]-tl[1])) > .2*horizontal[0]:
        return fallback
    distance = (abs(dx)*image_width/scale_x if direction in ("left", "right")
                else abs(dy)*image_height/scale_y)
    # Ask for at most five centimetres, then observe again to limit overshoot.
    step_cm = max(1, min(5, int(float(distance)+.5)))
    unit = "centimetre" if step_cm == 1 else "centimetres"
    return (f"Move the camera about {step_cm} {unit} {direction}. "
            "Keep it facing the text, then pause for guidance.")


class FrameAnalyzer:
    def __init__(self, config=None):
        self.config = config or {}
        self.previous = None

    def assess(self, frame):
        cfg = self.config
        frame = orient_frame(frame, cfg)
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(cv2.resize(frame, (max(1, int(w * 640 / max(h, w))),
                                             max(1, int(h * 640 / max(h, w))))), cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        small = cv2.resize(gray, (160, 120))
        motion = 0 if self.previous is None else float(np.mean(cv2.absdiff(small, self.previous)))
        self.previous = small
        marks = _text_marks(gray)

        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        page, best_score = None, 0
        # Prefer a rectangle containing text rather than the first large object
        # (a desk, keyboard or monitor border can also be rectangular).
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
            area = cv2.contourArea(contour)
            if area < h * w * .04:
                break
            polygon = cv2.approxPolyDP(contour, .025 * cv2.arcLength(contour, True), True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            x, y, bw, bh = cv2.boundingRect(polygon)
            inside = [s for s in marks if cv2.pointPolygonTest(
                polygon, (float(s[0]+s[2]/2), float(s[1]+s[3]/2)), False) > 0]
            # The full frame is not a useful boundary for centering guidance.
            if len(inside) >= 2 and area < h*w*.97:
                score = len(inside) / max(area, 1) ** .5
                if score > best_score:
                    page, best_score = polygon, score

        if page is not None:
            x, y, bw, bh = cv2.boundingRect(page)
            inside = [s for s in marks if cv2.pointPolygonTest(
                page, (float(s[0]+s[2]/2), float(s[1]+s[3]/2)), False) > 0]
        elif marks:
            inside = marks
            x, y = min(s[0] for s in marks), min(s[1] for s in marks)
            bw = max(s[0]+s[2] for s in marks) - x
            bh = max(s[1]+s[3] for s in marks) - y
        else:
            x, y, bw, bh, inside = 0, 0, w, h, []
        roi = gray[y:y+bh, x:x+bw]
        patches = []
        for s in inside[:250]:
            sx, sy, sw, sh = map(int, s[:4])
            patches.append(gray[max(0, sy-2):min(h, sy+sh+2),
                                max(0, sx-2):min(w, sx+sw+2)])
        # A dark background around a lit phone is not a dark document. Also
        # allow white text on a dark screen when its local contrast is usable.
        contrast = float(np.percentile(roi, 95) - np.percentile(roi, 5))
        if patches:
            # A single bright label can occupy less than five percent of a
            # dark screen; measure the letters as well as the whole screen.
            contrast = max(contrast, float(np.median([
                np.percentile(patch, 90) - np.percentile(patch, 10) for patch in patches])))
        dark = float(np.mean(roi)) < number(cfg, "ocr_guidance_min_brightness", 1, 200)
        if dark and (len(inside) < 2 or contrast < 45):
            return Assessment("dark", "I need more light on the text. Add light to the page, or brighten the phone screen.")
        if motion > number(cfg, "ocr_guidance_max_motion", 1, 80):
            return Assessment("moving", "Hold the camera and the text still. Stop scrolling if using a phone.")
        if len(inside) < 2:
            return Assessment("search", "I have not found text yet. Aim at the page or lit phone screen. Move slowly.")

        # Every readable assessment, including a framing warning, must pass
        # focus first. A framing correction must never conceal blurry text.
        detail = [cv2.Laplacian(patch, cv2.CV_64F).var() for patch in patches]
        if detail and np.median(detail) < number(cfg, "ocr_guidance_min_sharpness", 1, 1000):
            return Assessment("blur", "Text looks blurry. Hold still and keep the camera parallel to the page or screen.")

        margin = number(cfg, "ocr_guidance_edge_margin", .005, .1)
        # A complete page outline inside the image rules out camera-edge
        # clipping of that page. Background marks must not veto its capture.
        complete_page = (page is not None and x > 2 and y > 2
                         and x+bw < w-2 and y+bh < h-2)
        # A margin is useful for positioning, but being near an edge does not
        # mean a glyph is cut off. Require marks to touch the image boundary.
        edge = max(1, min(2, int(min(w, h) * margin)))
        sides = {
            "left": sum(s[0] <= edge for s in inside),
            "right": sum(s[0]+s[2] >= w-edge for s in inside),
            "top": sum(s[1] <= edge for s in inside),
            "bottom": sum(s[1]+s[3] >= h-edge for s in inside),
        }
        clipped = [side for side, count in sides.items()
                   if not complete_page and count >= max(3, len(inside)*.06)]
        if clipped:
            return Assessment("clipped", "Text may be cut off at the " + " and ".join(clipped) +
                              ". Move the camera a little farther away.", True)

        dx, dy = (x + bw/2) / w - .5, (y + bh/2) / h - .5
        tolerance = number(cfg, "ocr_guidance_center_tolerance", .03, .3)
        if max(abs(dx), abs(dy)) > tolerance:
            direction = ("right" if dx > 0 else "left") if abs(dx) >= abs(dy) else ("down" if dy > 0 else "up")
            return Assessment(direction, movement_prompt(direction, dx, dy, w, h, page, cfg), True)
        if page is not None and max(bw/w, bh/h) < number(cfg, "ocr_guidance_min_fill", .2, .9):
            return Assessment("closer", "Text is too far away. Bring the camera a little closer.", True)
        return Assessment("ready", "Text is in position.", True)


class GuidanceGate:
    def __init__(self, config):
        self.required = int(number(config, "ocr_guidance_stable_frames", 2, 20))
        self.hold = number(config, "ocr_guidance_hold_seconds", .3, 5)
        self.framing_grace = number(config, "ocr_guidance_framing_grace_s", 3, 30)
        self.interval = number(config, "ocr_guidance_prompt_interval_s", 1, 15)
        self.repeat = number(config, "ocr_guidance_repeat_interval_s", 5, 30)
        self.code = None
        self.count = 0
        self.since = 0
        self.last_spoken = -float("inf")
        self.last_code = None
        self.repeats = 0
        self.readable_since = None
        self.readable_count = 0

    def update(self, assessment, now):
        if assessment.readable:
            if self.readable_since is None:
                self.readable_since = now
            self.readable_count += 1
        else:
            self.readable_since = None
            self.readable_count = 0
        if assessment.code == self.code:
            self.count += 1
        else:
            self.code, self.count, self.since = assessment.code, 1, now
        ready = (assessment.code == "ready" and self.count >= self.required
                 and now - self.since >= self.hold)
        # Direction estimates can alternate even while usable text is steady.
        # Give the user time to improve framing, then read the visible text.
        # Darkness, motion, blur or missing text reset this window immediately.
        if (assessment.readable and self.readable_count >= self.required
                and now - self.readable_since >= self.framing_grace):
            ready = True
        same = assessment.code == self.last_code
        speak = (not ready and assessment.code != "ready" and self.count >= 2
                 and now - self.last_spoken >= (self.repeat if same else self.interval)
                 and (not same or self.repeats < 3))
        if speak:
            self.repeats = self.repeats + 1 if same else 1
            self.last_spoken, self.last_code = now, assessment.code
        return ready, speak


def guided_capture(camera, speak, stop_check, config):
    """Guide, capture, and allow a bounded recapture without leaving OCR mode."""
    analyzer, gate = FrameAnalyzer(config), GuidanceGate(config)
    stopped = stop_check or (lambda: False)
    speak("Keep the page or phone still and move the camera when I guide you. "
          "Left and right mean sliding sideways. Up and down mean sliding toward "
          "the top or bottom of the text, not moving closer or farther away. "
          "For a phone, turn on the screen and stop scrolling. I will take the picture automatically.")
    deadline = time.monotonic() + number(config, "ocr_guidance_timeout_s", 10, 300)
    attempts = int(number(config, "ocr_guidance_capture_attempts", 1, 3))
    missing_since = None
    last_assessment = Assessment("search", "Aim at the page or lit phone screen.")
    try:
        if stopped():
            return None, "OCR scan cancelled."
        if not camera.start_preview():
            return None, "OCR guidance unavailable. Camera preview could not start. Please try again."
        while time.monotonic() < deadline:
            if stopped():
                return None, "OCR scan cancelled."
            frame = camera.preview_frame()
            now = time.monotonic()
            if frame is None:
                missing_since = now if missing_since is None else missing_since
                if now - missing_since > 8:
                    return None, "OCR guidance unavailable. Camera preview stopped. Please try again."
                time.sleep(.1)
                continue
            missing_since = None
            assessment = last_assessment = analyzer.assess(frame)
            ready, announce = gate.update(assessment, now)
            if announce:
                message = assessment.message
                if gate.repeats > 1 and assessment.code in ("search", "blur"):
                    message = ("Keep the camera facing the text straight on. If using a phone, "
                               "tilt the phone slightly to move reflections away, then hold still.")
                speak(message)
                # Reassess fresh frames after speech; pre-prompt stability must
                # not trigger capture while the user is following a correction.
                analyzer.previous = None
                gate.count = 0
                gate.since = time.monotonic()
            if stopped():
                return None, "OCR scan cancelled."
            if ready:
                if assessment.code == "ready":
                    speak("Text is in position. Hold still while I take the picture.")
                else:
                    speak("I can see steady text, but I cannot confirm the whole page fits. "
                          "Hold still. I will photograph the visible text now.")
                if stopped():
                    return None, "OCR scan cancelled."
                camera.stop_preview()
                # Keep camera ownership on this thread. A cancelled background
                # capture must never overlap release or the next scan.
                frame = camera.capture(stop_check=stopped)
                if stopped():
                    return None, "OCR scan cancelled."
                attempts -= 1
                check = FrameAnalyzer(config).assess(frame) if frame is not None else None
                if check and check.readable:
                    if check.code == "clipped":
                        speak("Some text at the edge may be missing from this picture.")
                    speak("Picture taken. You can relax your hand while I read.")
                    return frame, ""
                if attempts <= 0:
                    return None, "OCR guidance unavailable. I could not capture clear text. Please select scan to try again."
                speak("That picture was not clear enough. Let us try once more. " +
                      (check.message if check else "Keep the camera aimed at the text."))
                if stopped():
                    return None, "OCR scan cancelled."
                # Give the user a fresh positioning window for the recapture.
                deadline = time.monotonic() + number(config, "ocr_guidance_timeout_s", 10, 300)
                analyzer, gate = FrameAnalyzer(config), GuidanceGate(config)
                if not camera.start_preview():
                    return None, "OCR guidance unavailable. Camera preview could not restart."
                missing_since = None
            time.sleep(.15)
        return None, "OCR positioning timed out. " + last_assessment.message + " Select scan to try again."
    finally:
        camera.stop_preview()
