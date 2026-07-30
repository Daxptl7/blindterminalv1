---
status: awaiting_human_verify
trigger: "Investigate and help fix the OCR slowness/functionality issue in /Users/daxpatel/Desktop/blindterminalv1. Summary: User says OCR takes too much time to load and wants all functionality checked so it works properly."
created: 2026-07-28T17:28:25Z
updated: 2026-07-28T17:39:56Z
---

## Current Focus
<!-- OVERWRITE on each update - reflects NOW -->

hypothesis: Root cause is confirmed; current worktree fix direction is fast-default OCR routing with Surya opt-in, and tests pass.
test: Human/user or main agent should verify Mode 1 OCR on Raspberry Pi hardware after the modules/ocr.py implementation changes land.
expecting: Mode 1 should avoid Surya by default and complete using Gemini or Tesseract in seconds; Surya should only appear in logs when ocr_engine is "surya" or ocr_surya_fallback is true.
next_action: await Pi hardware verification or main-agent confirmation

## Symptoms
<!-- Written during gathering, then IMMUTABLE -->

expected: OCR scan should respond quickly and reliably for a Raspberry Pi BlindAssist app.
actual: Logs show Surya OCR cold starts and inference take ~98-111 seconds; user experiences very slow load.
errors: Recent logs include Surya API errors ('list' object has no attribute 'bboxes'), llama-server binary missing, model download/startup, then long Surya inference.
reproduction: Use Mode 1 OCR scan in main.py; scan_and_read in modules/ocr.py captures camera and runs Gemini if configured, otherwise Surya.
started: Started after OCR module moved to Surya OCR 2 based on recent logs around 2026-07-26.

## Eliminated
<!-- APPEND only - prevents re-investigating -->

## Evidence
<!-- APPEND only - facts discovered -->

- timestamp: 2026-07-28T17:29:28Z
  checked: .planning/debug/knowledge-base.md
  found: No knowledge base file exists in this workspace.
  implication: No prior known-pattern hypothesis is available; investigate from repository evidence.
- timestamp: 2026-07-28T17:30:15Z
  checked: repository-wide OCR search
  found: Broad search is dominated by vendored llama.cpp and historical logs; app-level hits include modules/ocr.py, main.py, config/settings.json, tests/test_core_regressions.py, and logs/main.log.
  implication: Narrow inspection is required to avoid anchoring on unrelated vendored code.
- timestamp: 2026-07-28T17:32:44Z
  checked: modules/ocr.py, main.py, config/settings.json, tests/test_core_regressions.py, logs/main.log
  found: scan_and_read captures a frame, tries Gemini only when gemini_api_key is present, catches any Gemini exception, and then synchronously calls SuryaOCREngine.extract_text. Config defaults include ocr_engine and ocr_surya_fallback, but settings.json does not set them. GeminiOCREngine stores gemini_timeout_s but does not pass it to generate_content. Existing tests have no OCR regression coverage.
  implication: A Gemini package/API/network/key failure can put the user into a slow local Surya path with no clear runtime switch or timeout guard.
- timestamp: 2026-07-28T17:32:44Z
  checked: logs/main.log around 2026-07-26
  found: Earlier OCR on 2026-07-21 completed in about 2325 ms. After the Surya path, logs show Surya load, model download/server attach, and Surya OCR inference taking 98418 ms and 111360 ms.
  implication: The observed user slowness correlates directly with Surya local inference, not camera capture alone.
- timestamp: 2026-07-28T17:34:10Z
  checked: current modules/ocr.py search results
  found: The file now contains a TesseractOCREngine and references to _surya_enabled, ocr_engine, and ocr_surya_fallback at later lines that were not present in the earlier read.
  implication: Another agent likely changed modules/ocr.py during this investigation; review must account for the current version without reverting it.
- timestamp: 2026-07-28T17:35:26Z
  checked: current modules/ocr.py lines 542-616
  found: The current scan_and_read routes auto/gemini through Gemini first, then Tesseract, and only calls Surya after a Tesseract failure/status when ocr_surya_fallback is true. Explicit engine_name "surya" still calls Surya.
  implication: The current worktree likely fixes the main latency root cause, but needs regression tests for default no-Surya behavior.
- timestamp: 2026-07-28T17:37:40Z
  checked: tests/test_core_regressions.py
  found: Added regression coverage for auto mode with Gemini disabled and Tesseract unavailable, asserting Surya is not called when ocr_surya_fallback is false.
  implication: Default interactive OCR should not silently fall into the multi-minute Surya path when fast local OCR is unavailable.
- timestamp: 2026-07-28T17:38:22Z
  checked: python -m unittest tests.test_core_regressions
  found: Failed before running tests because tests is not an importable package: ModuleNotFoundError: No module named 'tests.test_core_regressions'.
  implication: Use direct file execution or add package discovery configuration separately.
- timestamp: 2026-07-28T17:39:56Z
  checked: python tests/test_core_regressions.py
  found: All 10 tests passed. The run emitted environment warnings for Matplotlib/font caches, pygame mixer, and missing serial, but no test failures.
  implication: The added OCR routing regression is passing in this environment.

## Resolution
<!-- OVERWRITE as understanding evolves -->

root_cause: After OCR moved to Surya OCR 2, the interactive Mode 1 scan path could enter local Surya/llama-server inference after Gemini was unavailable or failed. Logs prove this path took 98418-111360 ms on the Pi, compared with earlier sub-second to low-second OCR. The earlier Surya API mismatch also caused "'list' object has no attribute 'bboxes'", and missing llama-server/model startup added functional failures.
fix: Current modules/ocr.py worktree changes by another agent route auto mode through Gemini first, then fast local Tesseract, and only call Surya when explicitly selected or ocr_surya_fallback is true. This investigation added a regression test ensuring auto mode does not invoke Surya when Tesseract is unavailable and fallback is disabled.
verification: python tests/test_core_regressions.py passed 10 tests locally. Still needs Raspberry Pi Mode 1 hardware verification because camera/audio/network conditions cannot be fully exercised here.
files_changed: [tests/test_core_regressions.py, .planning/debug/ocr-slow-load.md]
