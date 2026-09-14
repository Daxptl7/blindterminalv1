# OCR validation

BlindAssist must be evaluated on photographs captured by the real device. A
Tesseract or Gemini confidence value is not an accuracy percentage.

Create a frozen set of at least 50 representative pages. Include 8–12 point
text, one- and two-column layouts, page curvature, headings, footnotes,
numbers, English, Hindi, and Gujarati. Keep the original camera images; do not
replace difficult pages after measuring them.

For each image, manually create an exact UTF-8 transcription. Put the images,
transcriptions, and a manifest in one directory:

```json
{
  "pages": [
    {
      "image": "page-001.jpg",
      "ground_truth": "page-001.txt",
      "lang": "eng"
    }
  ]
}
```

Run the local OCR release gate:

```bash
python3 scripts/benchmark_ocr.py data/ocr_benchmark/manifest.json --engine tesseract
```

The default gate requires character error rate at or below 5%, word recall at
or above 95%, and numeric accuracy at or above 99%. Also inspect every failed
page for omitted paragraphs or incorrect reading order. Record the p95 latency
on the Raspberry Pi, not only on a development laptop.

Camera setup is part of the test: use a rigid overhead mount, diffuse light on
both sides, keep the whole page inside the focus window, and make the page fill
most of the frame. If those conditions are not met consistently, software-only
tuning will not produce a dependable reader.
