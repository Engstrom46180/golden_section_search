# golden_section_search

A small, dependency-free Python library for minimising a unimodal scalar function on a closed interval using the golden-section search. No derivatives required.

## Usage

```python
from golden_section_search import golden_section_search

f = lambda x: (x - 2.0) ** 2
result = golden_section_search(f, 0.0, 5.0, tol=1e-10)
print(result.x)          # ~2.0
print(result.fx)         # ~0.0
print(result.iterations) # number of bracket-narrowing steps
print(result.evaluations)# total function calls
```

The exported names are `golden_section_search` (the search function) and `GoldenSectionResult` (the returned dataclass). The module `golden_section_search.core` also exposes `ConvergenceError`, raised on non-finite values or failure to converge within `max_iter`.

## Why this exists

When you have a unimodal function and cannot (or will not) compute its derivative, golden-section search gives a guaranteed linear contraction of the bracket per iteration at the cost of one function evaluation. It is slower than Brent's method but simpler, fully deterministic, and easy to audit. This library is for cases where that simplicity matters more than the last word in speed.

The trade-off: the function must be unimodal on the interval. If it is not, the method may converge to a point that is not a minimum. We do not attempt to detect multimodality; that is the caller's responsibility.

## The awkward edge

The method compares function values by ordering. If your function returns `NaN` or `inf` anywhere inside the bracket, the comparison is meaningless and the result is garbage. This library detects non-finite values and raises `ConvergenceError` immediately rather than returning a misleading answer. If your function has singularities, either bracket around them or handle them before calling.

## Parameters

- `func`: callable taking a float, returning a float. Must be unimodal on `[a, b]`.
- `a`, `b`: bracket endpoints. Must satisfy `b > a`.
- `tol`: desired final bracket width. Iteration stops when `b - a <= tol`. Must be positive.
- `max_iter`: hard cap on iterations. Defaults to 100. If the bracket has not narrowed to `tol` by then, `ConvergenceError` is raised.

## Return value

A `GoldenSectionResult` with fields `x` (minimiser estimate, the bracket midpoint), `fx` (function value at `x`), `a` and `b` (final bracket), `iterations`, and `evaluations`.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

