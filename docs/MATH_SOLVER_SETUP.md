# BlindAssist Math Solver

Mode 9 uses a verified offline symbolic solver before it considers an AI
answer. This prevents simple arithmetic and algebra from changing with the
network provider or model temperature.

## Supported verified problems

- arithmetic with parentheses, decimals, powers and square roots;
- common spoken forms such as “two x plus three equals seven”;
- percentages, percentage increases/decreases, and averages;
- one-variable polynomial equations through degree four;
- determinants of square matrices up to four by four;
- exact rational and radical results rather than rounded guesses.

The parser constructs SymPy expressions from a strict allow-list. It never
runs `eval`, `sympify` on raw user input, or arbitrary Python names/functions.
Input length, AST size, exponent, polynomial degree, and matrix size are
bounded to prevent accidental resource exhaustion on the Raspberry Pi.

## Input and output

Voice and Morse input can spell operations naturally:

```text
TWO PLUS THREE TIMES FOUR
SQUARE ROOT OF SIXTEEN PLUS TWO POINT FIVE
TWO X PLUS THREE EQUALS SEVEN
X SQUARED MINUS FIVE X PLUS SIX EQUALS ZERO
MATRIX TWO BY TWO ROW ONE 1 2 ROW TWO 3 4 FIND DETERMINANT
```

The answer first says how the problem was interpreted, then gives short spoken
steps and a final answer without relying on visual notation. Equation roots
are substituted back before they are announced.

## Word problems

Open-ended word problems that cannot be converted safely are sent to the AI
tutor only when an AI provider is available. The device announces that this
fallback is not automatically verified. RAG retrieval is disabled for this
call so unrelated textbook passages cannot alter the calculation.

## Laptop and Pi validation

Install the dependency and run the tests:

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -p 'test_math_solver_product.py' -v
python3 selftest.py
```

No separate model download is required. The same deterministic solver runs on
the laptop and Raspberry Pi CPU.
