"""Exact, offline-first mathematics for BlindAssist Mode 9.

The parser accepts symbols or common spoken forms, builds SymPy objects from a
strict Python AST allow-list, and never evaluates user input as Python code.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Optional

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except Exception:  # pragma: no cover - exercised on incomplete installations
    sp = None
    SYMPY_AVAILABLE = False


MAX_INPUT_CHARS = 500
MAX_AST_NODES = 80
MAX_MATRIX_SIDE = 4


@dataclass(frozen=True)
class MathSolution:
    ok: bool
    spoken: str
    normalized: str = ""
    exact: str = ""
    kind: str = ""
    error_code: str = ""


_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUMBER_WORDS = set(_UNITS) | set(_TENS) | {"hundred", "thousand", "point"}


class MathParseError(ValueError):
    def __init__(self, message: str, code: str = "INVALID"):
        super().__init__(message)
        self.code = code


def diagnostics() -> dict:
    return {
        "sympy_ready": SYMPY_AVAILABLE,
        "supported": [
            "arithmetic", "square roots", "percentages", "averages",
            "one-variable polynomial equations", "matrix determinants",
        ],
        "limits": {
            "input_characters": MAX_INPUT_CHARS,
            "equation_degree": 4,
            "matrix_side": MAX_MATRIX_SIDE,
        },
    }


def _parse_number_words(words: list[str]) -> Optional[str]:
    if not words or any(word not in _NUMBER_WORDS for word in words):
        return None
    if "point" in words:
        point = words.index("point")
        whole = _parse_number_words(words[:point]) if point else "0"
        decimals = words[point + 1:]
        if not decimals or any(word not in _UNITS or _UNITS[word] > 9
                               for word in decimals):
            return None
        return whole + "." + "".join(str(_UNITS[word]) for word in decimals)

    total = current = 0
    for word in words:
        if word in _UNITS:
            current += _UNITS[word]
        elif word in _TENS:
            current += _TENS[word]
        elif word == "hundred":
            current = max(current, 1) * 100
        elif word == "thousand":
            total += max(current, 1) * 1000
            current = 0
    return str(total + current)


def _convert_number_words(text: str) -> str:
    text = re.sub(r"\b(hundred|thousand)\s+and\b", r"\1", text.lower())
    tokens = re.findall(r"\d+(?:\.\d+)?|[a-z]+|[(),%+*/^=\-]", text)
    output = []
    index = 0
    while index < len(tokens):
        if tokens[index] not in _NUMBER_WORDS:
            output.append(tokens[index])
            index += 1
            continue
        end = index
        while end < len(tokens) and tokens[end] in _NUMBER_WORDS:
            end += 1
        converted = _parse_number_words(tokens[index:end])
        output.append(converted if converted is not None else tokens[index])
        index = end if converted is not None else index + 1
    return " ".join(output)


def _preclean(problem: str) -> str:
    text = problem.lower().strip()
    text = text.replace("’", "'")
    text = text.translate(str.maketrans({"×": "*", "÷": "/", "−": "-",
                                        "–": "-", "√": "sqrt ", "²": "^ 2",
                                        "³": "^ 3"}))
    text = re.sub(r"[?.!]+$", "", text)
    text = re.sub(
        r"^\s*(?:please\s+)?(?:what(?:'s|\s+is)|calculate|compute|evaluate|simplify|solve)\s+",
        "", text,
    )
    return re.sub(r"\s+", " ", text).strip()


def _special_expression(text: str) -> Optional[tuple[str, str]]:
    prepared = _convert_number_words(_preclean(text))
    prepared = re.sub(r"\bnegative\s+(?=\d)", "-", prepared)
    percent = re.fullmatch(
        r"(-?\d+(?:\.\d+)?)\s*(?:percent|%)\s+of\s+(-?\d+(?:\.\d+)?)",
        prepared,
    )
    if percent:
        return f"({percent.group(1)} / 100) * {percent.group(2)}", "percentage"

    change = re.fullmatch(
        r"(?:increase|decrease)\s+(-?\d+(?:\.\d+)?)\s+by\s+"
        r"(-?\d+(?:\.\d+)?)\s*(?:percent|%)",
        prepared,
    )
    if change:
        operator = "+" if prepared.startswith("increase") else "-"
        base, rate = change.group(1), change.group(2)
        return f"{base} {operator} ({base} * {rate} / 100)", "percentage"

    if prepared.startswith("average of ") or prepared.startswith("mean of "):
        body = prepared.split(" of ", 1)[1]
        values = re.findall(r"-?\d+(?:\.\d+)?", body)
        residue = re.sub(r"-?\d+(?:\.\d+)?|,|\band\b|\s+", "", body)
        if len(values) >= 2 and not residue:
            return f"({' + '.join(values)}) / {len(values)}", "average"
    return None


def normalize_expression(problem: str) -> str:
    """Convert common spoken mathematics to a small symbolic language."""
    text = _convert_number_words(_preclean(problem))
    replacements = (
        (r"\bis\s+equal\s+to\b|\bequals?\b", "="),
        (r"\braised\s+to\s+the\s+power(?:\s+of)?\b|"
         r"\bto\s+the\s+power(?:\s+of)?\b|\braised\s+to\b|"
         r"\bpower(?:\s+of)?\b", "^"),
        (r"\bmultiplied\s+by\b|\btimes\b", "*"),
        (r"\bdivided\s+by\b|\bover\b", "/"),
        (r"\bplus\b", "+"),
        (r"\bminus\b", "-"),
        (r"\bnegative\b", "-"),
        (r"\bopen\s+(?:bracket|parenthesis)\b", "("),
        (r"\bclose\s+(?:bracket|parenthesis)\b", ")"),
    )
    text = re.sub(r"\bsquare\s+root\s+(?:of\s+)?", "sqrt ", text)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s+halves?\b", r"(\1 / 2)", text)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s+quarters?\b", r"(\1 / 4)", text)
    text = re.sub(r"\b(?:a\s+)?half\b", r"(1 / 2)", text)
    text = re.sub(r"\b(?:a\s+)?quarter\b", r"(1 / 4)", text)
    for pattern, replacement in replacements:
        text = re.sub(pattern, f" {replacement} ", text)
    text = re.sub(r"\b([a-z]|\d+(?:\.\d+)?|\))\s+(?:squared|square)\b",
                  r"\1 ^ 2", text)
    text = re.sub(r"\b([a-z]|\d+(?:\.\d+)?|\))\s+(?:cubed|cube)\b",
                  r"\1 ^ 3", text)
    text = re.sub(r"\bsqrt\s+(\([^()]+\)|-?\d+(?:\.\d+)?|[a-z])\b",
                  r"sqrt(\1)", text)
    text = re.sub(r"\bthe\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Explicit multiplication is needed by Python's AST: 2x, 2(x+1), x(y).
    text = re.sub(r"(?<=\d)\s*(?=[a-z(])", " * ", text)
    text = re.sub(r"(?<=\))\s*(?=[a-z\d(])", " * ", text)
    text = re.sub(r"(?<=[a-z])\s+(?=[a-z\d(])", " * ", text)
    text = text.replace("sqrt * (", "sqrt(").replace("sqrt(", "sqrt(")
    return re.sub(r"\s+", " ", text).strip()


def _safe_expression(text: str):
    python_text = text.strip().replace("^", "**")
    try:
        tree = ast.parse(python_text, mode="eval")
    except SyntaxError as exc:
        raise MathParseError("the expression is incomplete or ambiguous") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise MathParseError("the expression is too complex", "TOO_COMPLEX")

    def convert(node):
        if isinstance(node, ast.Expression):
            return convert(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            value = str(node.value)
            if len(value.replace(".", "").replace("-", "")) > 40:
                raise MathParseError("a number is too large", "TOO_COMPLEX")
            return sp.Rational(value)
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return sp.pi
            if not re.fullmatch(r"[a-z]", node.id):
                raise MathParseError(f"unsupported name {node.id!r}")
            return sp.Symbol(node.id, real=True)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = convert(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = convert(node.left), convert(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                if not right.is_number or abs(float(right)) > 1000:
                    raise MathParseError("the exponent is unsupported", "TOO_COMPLEX")
                return left ** right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "sqrt" and len(node.args) == 1 \
                and not node.keywords:
            return sp.sqrt(convert(node.args[0]))
        raise MathParseError("the problem contains an unsupported operation")

    value = convert(tree)
    if value.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise MathParseError("division by zero is undefined", "DIVISION_BY_ZERO")
    return value


def _spoken(value) -> str:
    text = sp.sstr(value) if SYMPY_AVAILABLE and not isinstance(value, str) else str(value)
    text = re.sub(r"sqrt\(([^()]*)\)", r"the square root of \1", text)
    text = re.sub(r"\*\*2\b", " squared", text)
    text = re.sub(r"\*\*3\b", " cubed", text)
    text = text.replace("**", " to the power of ")
    text = text.replace("^", " to the power of ")
    text = text.replace("*", " times ").replace("/", " divided by ")
    text = text.replace("+", " plus ").replace("-", " minus ")
    text = text.replace("(", " open parenthesis ").replace(")", " close parenthesis ")
    text = re.sub(r"\bI\b", "the imaginary unit", text)
    text = re.sub(r"^\s*minus\s+", "negative ", text)
    return re.sub(r"\s+", " ", text).strip()


def _matrix_solution(problem: str) -> Optional[MathSolution]:
    prepared = _convert_number_words(_preclean(problem))
    if "matrix" not in prepared and "determinant" not in prepared:
        return None
    match = re.search(r"\bmatrix\s+(\d+)\s+by\s+(\d+)\b", prepared)
    if not match or "determinant" not in prepared:
        raise MathParseError(
            "say matrix, its row and column size, each row, and find determinant"
        )
    rows, columns = int(match.group(1)), int(match.group(2))
    if rows != columns:
        raise MathParseError("a determinant requires a square matrix")
    if rows < 1 or rows > MAX_MATRIX_SIDE:
        raise MathParseError("matrices larger than four by four are not supported",
                             "TOO_COMPLEX")

    body = prepared[match.end():]
    body = body[:body.find("determinant")]
    markers = list(re.finditer(r"\brow\s*(\d+)\b", body))
    values = []
    if markers:
        if len(markers) != rows:
            raise MathParseError(f"I expected {rows} matrix rows")
        for index, marker in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(body)
            row_values = re.findall(r"-?\d+(?:\.\d+)?", body[marker.end():end])
            if len(row_values) != columns:
                raise MathParseError(f"row {index + 1} needs {columns} values")
            values.extend(row_values)
    else:
        values = re.findall(r"-?\d+(?:\.\d+)?", body)
    if len(values) != rows * columns:
        raise MathParseError(f"I expected {rows * columns} matrix values")

    matrix = sp.Matrix(rows, columns, [sp.Rational(value) for value in values])
    determinant = sp.simplify(matrix.det())
    if rows == 2:
        step = ("For a two by two matrix, I multiplied the top left by the "
                "bottom right, then subtracted the other diagonal product. ")
    else:
        step = "I expanded the determinant and simplified each term. "
    spoken = (f"I read a {rows} by {columns} matrix. {step}"
              f"Final answer: the determinant is {_spoken(determinant)}.")
    return MathSolution(True, spoken, prepared, str(determinant), "determinant")


def solve(problem: str) -> MathSolution:
    """Solve supported mathematics exactly and return a TTS-friendly result."""
    if not SYMPY_AVAILABLE:
        return MathSolution(False, "Math Solver needs SymPy, but it is not installed.",
                            error_code="DEPENDENCY_MISSING")
    if not problem or not problem.strip():
        return MathSolution(False, "I did not receive a math problem.",
                            error_code="EMPTY")
    if len(problem) > MAX_INPUT_CHARS:
        return MathSolution(False, "That problem is too long for safe local solving.",
                            error_code="TOO_COMPLEX")
    try:
        matrix = _matrix_solution(problem)
        if matrix is not None:
            return matrix

        special = _special_expression(problem)
        if special:
            normalized, kind = special
        else:
            normalized, kind = normalize_expression(problem), "expression"
        if not normalized:
            raise MathParseError("the problem was empty")
        if re.search(r"[a-z]{2,}", normalized.replace("sqrt", "")):
            raise MathParseError("this looks like a word problem", "UNSUPPORTED")

        if normalized.count("=") > 1:
            raise MathParseError("only one equality may be solved at a time")
        if "=" in normalized:
            left_text, right_text = normalized.split("=", 1)
            left, right = _safe_expression(left_text), _safe_expression(right_text)
            difference = sp.together(left - right)
            interpretation = f"{_spoken(left_text)} equals {_spoken(right_text)}"
            source_variable_names = set(re.findall(r"\b([a-z])\b", normalized))
            variables = sorted(left.free_symbols | right.free_symbols,
                               key=lambda item: item.name)
            if sp.simplify(difference) == 0 and (variables or source_variable_names):
                names = ", ".join(
                    str(item) for item in variables
                ) or ", ".join(sorted(source_variable_names))
                spoken = (
                    f"I interpreted the equation as {interpretation}. "
                    f"It is true for every value of {names} where the original "
                    "expression is defined."
                )
                return MathSolution(True, spoken, normalized, "identity", "equation")
            if len(variables) > 1:
                raise MathParseError(
                    "equations with more than one unknown are not supported yet",
                    "UNSUPPORTED",
                )
            if not variables:
                true = sp.simplify(difference) == 0
                answer = "The equality is true." if true else "The equality is false."
                return MathSolution(True, f"I interpreted this as {interpretation}. {answer}",
                                    normalized, str(true), "equality")

            variable = variables[0]
            numerator, denominator = sp.fraction(difference)
            try:
                degree = sp.Poly(numerator, variable).degree()
            except sp.PolynomialError as exc:
                raise MathParseError(
                    "this equation is outside the verified local solver",
                    "UNSUPPORTED",
                ) from exc
            if degree > 4:
                raise MathParseError(
                    "equations above degree four are not supported", "TOO_COMPLEX"
                )
            candidates = sp.solve(sp.Eq(left, right), variable)
            solutions = []
            for candidate in candidates:
                if sp.simplify(denominator.subs(variable, candidate)) == 0:
                    continue
                if sp.simplify(difference.subs(variable, candidate)) == 0:
                    solutions.append(candidate)
            solutions = sorted(set(solutions), key=sp.default_sort_key)
            if not solutions:
                final = f"there is no real solution for {variable}"
                exact = "[]"
            elif len(solutions) == 1:
                final = f"{variable} equals {_spoken(solutions[0])}"
                exact = str(solutions[0])
            else:
                joined = ", or ".join(
                    f"{variable} equals {_spoken(value)}" for value in solutions
                )
                final, exact = joined, str(solutions)
            spoken = (
                f"I interpreted the equation as {interpretation}. "
                "I moved all terms to one side and solved it. "
                "I checked each result by substituting it back. "
                f"Final answer: {final}."
            )
            return MathSolution(True, spoken, normalized, exact, "equation")

        expression = _safe_expression(normalized)
        result = sp.simplify(expression)
        if len(str(result)) > 250:
            raise MathParseError("the exact result is too large to speak", "TOO_COMPLEX")
        interpretation = _spoken(normalized)
        final = _spoken(result)
        if expression.free_symbols:
            spoken = (f"I interpreted this as {interpretation}. I combined like terms "
                      f"and simplified it. Final answer: {final}.")
            kind = "simplification"
        else:
            spoken = (f"I interpreted this as {interpretation}. Using the order of "
                      f"operations, the final answer is {final}.")
        return MathSolution(True, spoken, normalized, str(result), kind)
    except MathParseError as exc:
        prefix = ("I cannot safely solve that locally. " if exc.code == "UNSUPPORTED"
                  else "I could not solve that problem. ")
        return MathSolution(False, prefix + str(exc) + ".", error_code=exc.code)
    except Exception:
        return MathSolution(False, "I could not solve that problem safely.",
                            error_code="INVALID")


def sanitize_ai_answer(answer: str) -> str:
    """Make an unverified AI fallback less hostile to text-to-speech."""
    text = str(answer or "").replace("$", " ")
    text = re.sub(r"\\(?:left|right)", "", text)
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"the square root of \1", text)
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}",
                  r"\1 divided by \2", text)
    text = text.replace("^2", " squared").replace("^3", " cubed")
    text = text.replace("×", " times ").replace("÷", " divided by ")
    text = text.replace("=", " equals ")
    text = re.sub(r"[*_`#]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
