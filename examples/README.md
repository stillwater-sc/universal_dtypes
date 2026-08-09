# Examples

Worked problems showing where each `universal_dtypes` number system earns its
keep, using plain NumPy. Every script is self-contained, prints a short
narrative, and **asserts its result** — so it doubles as a regression test
(`tests/test_examples.py` runs them all in CI).

Examples are organized into a subdirectory per number-system family. Run any of
them with the package installed (`pip install universal_dtypes`):

```bash
python examples/cascades/catastrophic_cancellation.py
```

| example | family | what it shows |
|---------|--------|---------------|
| [`cascades/catastrophic_cancellation.py`](cascades/catastrophic_cancellation.py) | dd/td/qd_cascade | a sum float64 gets wrong, the cascades get exact |
| [`bfloat16/dynamic_range.py`](bfloat16/dynamic_range.py) | bfloat16 | float32-class range vs float16 (why ML training uses it) |
| [`posit/tapered_precision.py`](posit/tapered_precision.py) | posit | more accuracy near ±1 than float16; NaR (no inf) |
| [`cfloat/precision_tradeoff.py`](cfloat/precision_tradeoff.py) | cfloat (fp16, fp8e5m2) | halving storage for a bounded precision cost |
| [`lns/power_of_two_scaling.py`](lns/power_of_two_scaling.py) | lns | multiply/divide in the log domain; exact power-of-two scaling |

One example per family today; this set is meant to grow. To add one, drop a
script into the relevant family directory (`examples/<family>/`) with a clear
docstring stating the problem and an `assert` on the "universal wins" outcome —
the CI runner (`tests/test_examples.py`) picks it up automatically.

Examples that need MTL5's accelerated linear-algebra solvers belong in
[`mtl5-python`](https://github.com/stillwater-sc/mtl5-python), not here — this set
stays pure NumPy + `universal_dtypes`.
