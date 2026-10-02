"""Golden-section search for minimising a unimodal scalar function.

The golden-section search is a derivative-free bracketing method. It narrows
an interval [a, b] known to contain a single minimum of a unimodal function
by comparing two interior points placed at the golden ratio. Each iteration
reduces the interval by a factor of ~0.618, so the number of evaluations is
logarithmic in the desired tolerance.

We return a small dataclass rather than a bare float so callers can inspect
the final bracket and the number of evaluations. This matters in practice:
the bracket width is the only honest measure of how well the minimum is
pinned down, and the evaluation count lets you budget expensive functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

# The golden ratio conjugate. Using the exact algebraic form keeps the two
# interior points symmetric throughout the iteration and avoids drift from
# repeated floating multiplication.
_GOLDEN = (5.0 ** 0.5 - 1.0) / 2.0


class ConvergenceError(RuntimeError):
    """Raised when the search cannot make progress, e.g. a non-finite value."""


@dataclass(frozen=True)
class GoldenSectionResult:
    """Outcome of a golden-section search.

    Attributes:
        x: The minimiser estimate (midpoint of the final bracket).
        fx: The function value at x.
        a, b: The final bracket endpoints, with a <= x <= b.
        iterations: Number of bracket-narrowing iterations performed.
        evaluations: Number of function evaluations (each iteration costs one).
    """

    x: float
    fx: float
    a: float
    b: float
    iterations: int
    evaluations: int


def golden_section_search(
    func: Callable[[float], float],
    a: float,
    b: float,
    *,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> GoldenSectionResult:
    """Minimise a unimodal scalar function on [a, b] without derivatives.

    The function must be unimodal on the interval: strictly decreasing then
    strictly increasing, with a single minimum. If it is not, the method may
    converge to a non-minimum point. We do not attempt to detect multimodality;
    that is the caller's responsibility.

    Args:
        func: A unimodal function of one float returning a float.
        a: Left endpoint of the bracket.
        b: Right endpoint of the bracket. Must satisfy b > a.
        tol: Desired bracket width. Iteration stops when b - a <= tol.
        max_iter: Hard cap on iterations to bound cost on expensive functions.

    Returns:
        A GoldenSectionResult describing the final bracket and estimate.

    Raises:
        ValueError: If the bracket is degenerate or tol is non-positive.
        ConvergenceError: If a function value is non-finite, or max_iter is
            reached without the bracket narrowing to tol.
    """
    if b <= a:
        raise ValueError(f"bracket must satisfy b > a; got a={a!r}, b={b!r}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive; got {tol!r}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1; got {max_iter!r}")

    def _eval(x: float) -> float:
        fx = func(x)
        # Non-finite values break the ordering assumptions of the method.
        # Failing fast is better than silently returning a nonsense bracket.
        if fx != fx or fx in (float("inf"), float("-inf")):
            raise ConvergenceError(
                f"function returned non-finite value {fx!r} at x={x!r}"
            )
        return fx

    lo, hi = a, b
    # First iteration needs two evaluations; subsequent iterations reuse one.
    c = hi - _GOLDEN * (hi - lo)
    d = lo + _GOLDEN * (hi - lo)
    fc = _eval(c)
    fd = _eval(d)
    evaluations = 2
    iterations = 0

    while hi - lo > tol:
        if iterations >= max_iter:
            raise ConvergenceError(
                f"failed to converge in {max_iter} iterations; "
                f"bracket width {hi - lo!r} exceeds tol {tol!r}"
            )
        if fc < fd:
            hi = d
            d = c
            fd = fc
            c = hi - _GOLDEN * (hi - lo)
            fc = _eval(c)
        else:
            lo = c
            c = d
            fc = fd
            d = lo + _GOLDEN * (hi - lo)
            fd = _eval(d)
        iterations += 1
        evaluations += 1

    x = 0.5 * (lo + hi)
    return GoldenSectionResult(
        x=x,
        fx=_eval(x),
        a=lo,
        b=hi,
        iterations=iterations,
        evaluations=evaluations + 1,
    )
