---
status: fixing
trigger: "Raspberry Pi translation en->hi rejects Google and MyMemory outputs because protected-token validation reports missing protected tokens: m, m for source text containing i'm i'm."
created: 2026-09-14T21:30:00+05:30
updated: 2026-09-14T21:41:00+05:30
---

## Current Focus

hypothesis: confirmed — `_UNIT_RE` treats the `m` in English contractions as a standalone metre unit because an apostrophe satisfies both non-word boundaries
test: replace standalone-unit extraction with measurement-context extraction, then rerun the exact failing test and adjacent safety tests
expecting: contractions produce no protected `m`, while numeric suffix units and currency-prefix units remain protected
next_action: apply the minimal contextual-unit extraction fix

## Symptoms

expected: English sentences containing contractions such as "I'm" translate to Hindi; unit preservation applies only to real measurements.
actual: Google and MyMemory translations are rejected and the original English text is returned.
errors: "Rejected google translation: missing protected tokens: m, m"; same for MyMemory. IndicTrans2 missing is expected. Audio playback on plughw:3,0 fails separately.
reproduction: Run Mode 4 voice input English to Hindi and speak a sentence containing "I'm" twice.
started: Began after commit 29e4db0 added protected-token validation.

## Eliminated

## Evidence

- timestamp: 2026-09-14T21:33:00+05:30
  checked: `.planning/debug/knowledge-base.md`
  found: No debug knowledge base exists yet.
  implication: There is no prior known-pattern diagnosis to test first.

- timestamp: 2026-09-14T21:33:30+05:30
  checked: Complete protected-token and translation validation path in `modules/translator.py`
  found: `_UNIT_RE` allows any non-word character before the unit and includes the bare unit `m`; `_protected_tokens` applies it to the full source text.
  implication: In `I'm`, the apostrophe before `m` is not a word character and the end after `m` is also a valid boundary, so `m` is collected once per contraction.

- timestamp: 2026-09-14T21:35:00+05:30
  checked: Focused test invocation with `python`
  found: The current laptop shell has no `python` executable on PATH.
  implication: Use the repository virtual environment or `python3`; this is a test-environment issue, not evidence about the validator hypothesis.

- timestamp: 2026-09-14T21:36:00+05:30
  checked: Repository `.venv` test tooling
  found: `.venv/bin/python` exists but pytest is not installed.
  implication: The test file is written with `unittest`, so run it using `python3 -m unittest` without adding dependencies.

- timestamp: 2026-09-14T21:37:00+05:30
  checked: Dotted unittest invocation
  found: `tests/` has no package initializer, so `tests.test_translation_product` is not importable.
  implication: Execute `tests/test_translation_product.py` directly to run its unittest suite.

- timestamp: 2026-09-14T21:38:00+05:30
  checked: Direct execution of the translation test file
  found: Python places `tests/`, rather than the repository root, first on `sys.path`, so imports of `modules` and `main` fail.
  implication: Set `PYTHONPATH=.` for the repository's direct-file test convention.

- timestamp: 2026-09-14T21:40:00+05:30
  checked: New regression test against the unmodified production validator
  found: The test fails with `['missing protected tokens: m, m']`, exactly matching the Raspberry Pi log.
  implication: The hypothesis is confirmed and the failure is independent of Google, MyMemory, networking, or IndicTrans2 availability.

## Resolution

root_cause: The unit regex accepted any standalone unit abbreviation bounded by non-word characters. In `I'm`, the apostrophe and string/word boundary make the contraction suffix `m` look like the metre abbreviation; two contractions therefore require two nonexistent `m` tokens in the Hindi output.
fix: pending
verification:
files_changed: []
