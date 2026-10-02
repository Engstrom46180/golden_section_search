import math
import unittest

from golden_section_search import golden_section_search, GoldenSectionResult
from golden_section_search.core import ConvergenceError


class TestGoldenSectionSearch(unittest.TestCase):
    def test_finds_minimum_of_quadratic(self):
        # f(x) = (x - 2)^2 has its minimum at x = 2.
        f = lambda x: (x - 2.0) ** 2
        result = golden_section_search(f, 0.0, 5.0, tol=1e-10)
        self.assertIsInstance(result, GoldenSectionResult)
        self.assertAlmostEqual(result.x, 2.0, places=6)
        self.assertAlmostEqual(result.fx, 0.0, places=6)
        self.assertLessEqual(result.b - result.a, 1e-10)
        self.assertGreaterEqual(result.iterations, 1)

    def test_finds_minimum_of_cosine(self):
        # cos(x) on [0, pi] is unimodal with minimum at pi.
        result = golden_section_search(math.cos, 0.0, math.pi, tol=1e-10)
        self.assertAlmostEqual(result.x, math.pi, places=6)
        self.assertAlmostEqual(result.fx, -1.0, places=6)

    def test_minimum_at_left_endpoint(self):
        # f(x) = x^2 on [-1, 1] is unimodal with minimum at 0, interior.
        # Use a strictly increasing function to test endpoint behaviour.
        f = lambda x: x
        result = golden_section_search(f, 1.0, 3.0, tol=1e-10)
        self.assertAlmostEqual(result.x, 1.0, places=6)

    def test_minimum_at_right_endpoint(self):
        f = lambda x: -x
        result = golden_section_search(f, 1.0, 3.0, tol=1e-10)
        self.assertAlmostEqual(result.x, 3.0, places=6)

    def test_result_fields_are_consistent(self):
        f = lambda x: (x - 1.0) ** 2 + 3.0
        result = golden_section_search(f, -2.0, 4.0, tol=1e-9)
        self.assertLessEqual(result.a, result.x)
        self.assertLessEqual(result.x, result.b)
        self.assertEqual(result.evaluations, result.iterations + 3)
        self.assertAlmostEqual(result.fx, f(result.x), places=10)

    def test_rejects_degenerate_bracket(self):
        with self.assertRaises(ValueError):
            golden_section_search(lambda x: x, 1.0, 1.0)
        with self.assertRaises(ValueError):
            golden_section_search(lambda x: x, 2.0, 1.0)

    def test_rejects_non_positive_tol(self):
        with self.assertRaises(ValueError):
            golden_section_search(lambda x: x, 0.0, 1.0, tol=0.0)
        with self.assertRaises(ValueError):
            golden_section_search(lambda x: x, 0.0, 1.0, tol=-1e-6)

    def test_rejects_zero_max_iter(self):
        with self.assertRaises(ValueError):
            golden_section_search(lambda x: x, 0.0, 1.0, max_iter=0)

    def test_raises_on_non_finite_value(self):
        # Returns NaN for x near 0.5; should fail fast.
        def f(x):
            if abs(x - 0.5) < 0.01:
                return float("nan")
            return (x - 0.5) ** 2
        with self.assertRaises(ConvergenceError):
            golden_section_search(f, 0.0, 1.0, tol=1e-10)

    def test_raises_on_infinite_value(self):
        def f(x):
            if abs(x - 0.5) < 0.01:
                return float("inf")
            return (x - 0.5) ** 2
        with self.assertRaises(ConvergenceError):
            golden_section_search(f, 0.0, 1.0, tol=1e-10)

    def test_max_iter_cap_is_respected(self):
        # A tiny max_iter should either converge or raise ConvergenceError.
        # With tol very small and max_iter=1, it must raise.
        with self.assertRaises(ConvergenceError):
            golden_section_search(
                lambda x: (x - 2.0) ** 2,
                0.0,
                5.0,
                tol=1e-12,
                max_iter=1,
            )

    def test_converges_with_default_tolerance(self):
        f = lambda x: (x - 7.0) ** 2
        result = golden_section_search(f, 0.0, 20.0)
        self.assertAlmostEqual(result.x, 7.0, places=4)
        self.assertLessEqual(result.b - result.a, 1e-8)

    def test_handles_asymmetric_bracket(self):
        # Minimum not at the centre of the initial interval.
        f = lambda x: (x + 3.0) ** 2
        result = golden_section_search(f, -10.0, 10.0, tol=1e-10)
        self.assertAlmostEqual(result.x, -3.0, places=6)

    def test_evaluation_count_is_logged(self):
        calls = [0]
        def f(x):
            calls[0] += 1
            return (x - 1.0) ** 2
        result = golden_section_search(f, 0.0, 2.0, tol=1e-10)
        self.assertEqual(result.evaluations, calls[0])


if __name__ == "__main__":
    unittest.main()
