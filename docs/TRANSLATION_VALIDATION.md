# Translation validation and 95% release gate

“95% accuracy” is not established by a handful of phrases. Before calling the
module product-ready, create a blind-user-centered evaluation set with at least
500 independently reviewed sentences for each of the six directions (3,000
total). Keep train/tuning examples out of this set.

Include textbook paragraphs, navigation instructions, menus, dates, decimal
numbers, percentages, units, names, mixed English/Indic terms, punctuation,
poor OCR spacing, and safety/medical phrases. Use at least two native reviewers
for ambiguous examples and allow multiple valid reference translations.

Each JSONL row for `scripts/benchmark_translation.py` has this form:

```json
{"id":"en-hi-0001","src":"en","dest":"hi","source":"The dose is 25 mg.","references":["खुराक 25 मिलीग्राम है।"],"human_adequate":true}
```

Run locally/private by default:

```bash
python scripts/benchmark_translation.py data/translation-eval.jsonl
```

Use `--allow-cloud` only for a non-confidential dataset when cloud comparison
is intentional. The script reports reference chrF, deterministic safety
validation, provider, and latency. chrF is a regression signal, not proof of
meaning; the human score is the semantic release metric.

Release requires, for every direction:

- at least 500 human-scored sentences;
- at least 95% human adequacy and at least 95% automatic pass rate;
- 100% preservation of names, dates, numbers, percentages, and units in the
  safety-critical subset;
- zero wrong-script outputs accepted for speech;
- no cloud or persistent-cache access in private-mode tests;
- p95 latency measured separately on the target Raspberry Pi, not inferred
  from laptop timing.

The runtime validator rejects empty, severely truncated/expanded, unchanged,
wrong-script, or protected-token-losing results before they are cached or
spoken. It cannot prove semantic correctness, so it complements rather than
replaces native-speaker review.
