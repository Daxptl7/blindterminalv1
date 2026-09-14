"""Focused regressions for full-page and small-print OCR behavior."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from modules import ocr


def _tesseract_data(words, confidence, image_height=100):
    count = len(words)
    return {
        "text": list(words),
        "conf": [str(confidence)] * count,
        "block_num": [1] * count,
        "par_num": [1] * count,
        "line_num": [index // 4 + 1 for index in range(count)],
        "top": [min(image_height - 5, 5 + index * 4) for index in range(count)],
        "height": [4] * count,
    }


class OCRSmallTextTests(unittest.TestCase):
    def setUp(self):
        self.original_config = dict(ocr._config)

    def tearDown(self):
        ocr._config.clear()
        ocr._config.update(self.original_config)

    def test_default_capture_uses_native_resolution_and_page_wide_focus(self):
        ocr._config.update({
            "ocr_capture_width": 4608,
            "ocr_capture_height": 2592,
            "ocr_capture_quality": 95,
            "ocr_autofocus_range": "full",
            "ocr_autofocus_window": "0.05,0.05,0.90,0.90",
        })
        frame = np.zeros((2592, 4608, 3), dtype=np.uint8)

        with mock.patch.object(ocr, "RPICAM_BIN", "/usr/bin/rpicam-still"), \
             mock.patch.object(ocr.os.path, "exists", return_value=True), \
             mock.patch.object(ocr.cv2, "imread", return_value=frame), \
             mock.patch.object(ocr.os, "remove"), \
             mock.patch.object(ocr.subprocess, "run") as run:
            captured = ocr._rpicam_capture()

        self.assertIs(captured, frame)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--width") + 1], "4608")
        self.assertEqual(command[command.index("--height") + 1], "2592")
        self.assertEqual(command[command.index("--autofocus-range") + 1], "full")
        self.assertEqual(
            command[command.index("--autofocus-window") + 1],
            "0.05,0.05,0.90,0.90",
        )

    def test_native_page_is_split_into_overlapping_tiles(self):
        ocr._config.update({"ocr_tile_size": 1800, "ocr_tile_overlap": 180})
        page = Image.new("RGB", (4608, 2592), "white")

        with mock.patch.object(ocr.TesseractOCREngine, "_document_crop", return_value=None):
            tiles = ocr.TesseractOCREngine._tile_images(page)

        self.assertEqual(len(tiles), 6)
        self.assertEqual(tiles[0][0], "tile-r0-c0")
        self.assertEqual(tiles[-1][0], "tile-r1-c2")
        self.assertTrue(all(tile.size == (1800, 1800) for _, tile in tiles))

    def test_orientation_probe_selects_upside_down_page(self):
        engine = ocr.TesseractOCREngine()
        images = [
            Image.new("RGB", (120, 80), (value, value, value))
            for value in (0, 60, 180, 240)
        ]
        rotations = list(zip(("rot0", "rot90", "rot180", "rot270"), images))
        good_words = "this upside down textbook page is now read in correct order".split()

        def image_to_data(image, **_kwargs):
            marker = image.convert("L").getpixel((0, 0))
            if marker == 180:
                return _tesseract_data(good_words, 88, image.height)
            return _tesseract_data(["noise", "mark"], 18, image.height)

        fake_tesseract = SimpleNamespace(
            Output=SimpleNamespace(DICT="DICT"),
            image_to_data=image_to_data,
        )
        ocr._config["ocr_tesseract_psm_modes"] = "6"

        with mock.patch.object(engine, "_rotation_images", return_value=rotations), \
             mock.patch.object(
                 engine, "_variant_images", side_effect=lambda image: [("raw", image)]
             ), mock.patch.object(engine, "_tile_images", return_value=[]):
            candidate = engine._best_candidate(fake_tesseract, images[0], "eng")

        self.assertTrue(candidate.variant.startswith("rot180-"))
        self.assertIn("correct order", candidate.text)

    def test_coverage_score_prefers_dense_body_over_short_heading(self):
        heading = ocr.TesseractCandidate(
            "VERY CLEAR HEADING ONLY",
            avg_confidence=94,
            word_count=4,
            variant="heading",
            psm=6,
            line_count=1,
            vertical_coverage=0.1,
        )
        body = ocr.TesseractCandidate(
            "body " * 150,
            avg_confidence=76,
            word_count=150,
            variant="body",
            psm=3,
            line_count=24,
            vertical_coverage=0.9,
        )

        self.assertGreater(body.score, heading.score)

    def test_low_confidence_gibberish_is_rejected(self):
        engine = ocr.TesseractOCREngine()
        weak = ocr.TesseractCandidate(
            "xqz 11l O0O",
            avg_confidence=22,
            word_count=3,
            variant="weak",
            psm=6,
        )
        ocr._config["ocr_tesseract_min_confidence"] = 45

        with mock.patch.object(engine, "_check_available", return_value=True), \
             mock.patch.object(engine, "_resolve_lang", return_value="eng"), \
             mock.patch.object(engine, "_best_candidate", return_value=weak):
            result = engine.extract_text(Image.new("RGB", (100, 100)), "eng")

        self.assertEqual(result, "No text detected in the image.")

    def test_sparse_cloud_result_on_full_page_runs_local_coverage_check(self):
        frame = np.zeros((2592, 4608, 3), dtype=np.uint8)
        local_text = " ".join(f"body{index}" for index in range(40))
        ocr._config.update({
            "ocr_engine": "auto",
            "ocr_gemini_first": True,
            "gemini_api_key": "configured",
            "ocr_gemini_min_words": 12,
            "ocr_gemini_coverage_check_min_pixels": 2000000,
            "ocr_surya_fallback": False,
        })

        with mock.patch.object(ocr._camera, "open", return_value=True), \
             mock.patch.object(ocr._camera, "capture", return_value=frame), \
             mock.patch.object(
                 ocr._gemini_engine, "extract_text", return_value="Chapter One Introduction"
             ), mock.patch.object(
                 ocr._tesseract_engine, "extract_text", return_value=local_text
             ) as tesseract:
            result = ocr.scan_and_read("eng")

        self.assertEqual(result, local_text)
        tesseract.assert_called_once()


if __name__ == "__main__":
    unittest.main()
