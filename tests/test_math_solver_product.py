import importlib
import unittest
from unittest import mock


class MathSolverProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = importlib.import_module("modules.math_solver")

    def assertSolved(self, problem, exact):
        result = self.solver.solve(problem)
        self.assertTrue(result.ok, result.spoken)
        self.assertEqual(result.exact, exact)
        return result

    def test_arithmetic_preserves_order_of_operations(self):
        result = self.assertSolved("two plus three times four", "14")
        self.assertIn("2 plus 3 times 4", result.spoken)

    def test_square_root_and_spoken_decimal(self):
        self.assertSolved("square root of sixteen plus two point five", "13/2")

    def test_common_spoken_fractions(self):
        self.assertSolved("one half plus three quarters", "5/4")

    def test_linear_equation_is_solved_and_checked(self):
        result = self.assertSolved("two x plus three equals seven", "2")
        self.assertIn("substituting it back", result.spoken)

    def test_quadratic_returns_both_real_roots(self):
        self.assertSolved("x squared minus five x plus six equals zero", "[2, 3]")

    def test_no_real_solution_is_stated_honestly(self):
        result = self.assertSolved("x squared plus one equals zero", "[]")
        self.assertIn("no real solution", result.spoken)

    def test_cancelled_identity_preserves_domain_warning(self):
        result = self.assertSolved("two x divided by x equals two", "identity")
        self.assertIn("where the original expression is defined", result.spoken)
        self.assertSolved("x equals x", "identity")

    def test_percent_and_average(self):
        self.assertSolved("what is twenty percent of fifty", "10")
        self.assertSolved("average of 2, 4, and 6", "4")
        self.assertSolved("average of negative 2, 4, and 10", "4")
        self.assertSolved("increase 80 by 25 percent", "100")

    def test_spoken_variants_cover_school_arithmetic(self):
        cases = {
            "negative five plus two": "-3",
            "open bracket 2 plus 3 close bracket times 4": "20",
            "two power ten": "1024",
            "three squared plus four squared": "25",
            "one hundred and five plus twenty five": "130",
            "one point two five times four": "5",
        }
        for problem, exact in cases.items():
            with self.subTest(problem=problem):
                self.assertSolved(problem, exact)

    def test_matrix_determinant_uses_declared_rows(self):
        self.assertSolved(
            "matrix two by two row one 1 2 row two 3 4 find determinant", "-2"
        )

    def test_undefined_and_unsafe_input_are_rejected(self):
        zero = self.solver.solve("two divided by zero")
        self.assertFalse(zero.ok)
        self.assertEqual(zero.error_code, "DIVISION_BY_ZERO")

        malicious = self.solver.solve("__import__('os').system('echo unsafe')")
        self.assertFalse(malicious.ok)
        self.assertNotIn("unsafe\n", malicious.spoken)

    def test_word_problem_is_marked_for_unverified_fallback(self):
        result = self.solver.solve("John has five apples and buys three more")
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "UNSUPPORTED")

    def test_ai_answer_is_cleaned_for_speech(self):
        cleaned = self.solver.sanitize_ai_answer(
            "**Answer:** $x=\\frac{3}{2}$ and $y^2=4$."
        )
        for token in ("*", "$", "\\frac", "="):
            self.assertNotIn(token, cleaned)
        self.assertIn("3 divided by 2", cleaned)

    def test_mode_uses_verified_local_solver_before_ai(self):
        main = importlib.import_module("main")
        spoken = []
        ai = mock.Mock()
        with mock.patch.dict(main._modules, {
            "mathsolver": self.solver,
            "ai_query": ai,
        }), mock.patch.object(main, "_keyboard_input", return_value="2"), \
             mock.patch.object(main, "_morse_type_sentence",
                               return_value="two plus two"), \
             mock.patch.object(main, "_speak",
                               side_effect=lambda text, **kwargs: spoken.append(text)):
            main.mode_math_solver()
        ai.ask_ai.assert_not_called()
        self.assertTrue(any("final answer is 4" in text for text in spoken))

    def test_mode_disables_rag_for_unverified_word_problem(self):
        main = importlib.import_module("main")
        fallback = mock.Mock()
        fallback.solve.return_value = self.solver.MathSolution(
            False, "unsupported", error_code="UNSUPPORTED"
        )
        fallback.sanitize_ai_answer.side_effect = lambda value: value
        ai = mock.Mock()
        ai.ask_ai.return_value = "The final answer is eight."
        with mock.patch.dict(main._modules, {
            "mathsolver": fallback,
            "ai_query": ai,
        }), mock.patch.object(main, "_keyboard_input", return_value="2"), \
             mock.patch.object(main, "_morse_type_sentence",
                               return_value="a short word problem"), \
             mock.patch.object(main, "_speak"):
            main.mode_math_solver()
        self.assertTrue(ai.ask_ai.call_args.kwargs["context"])


if __name__ == "__main__":
    unittest.main()
