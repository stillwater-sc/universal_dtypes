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

## Application studies

Where the per-family demos isolate one type, these solve a real problem and
compare **two or more** number systems on it — organized by application domain
under [`applications/`](applications/).

| study | domain | number systems | what it shows |
|-------|--------|----------------|---------------|
| [`applications/math/rump.py`](applications/math/rump.py) | math | float32/64, dd/td/qd | Rump's example: float64 *and* double-double are wrong; triple-double is the first precision that solves it |
| [`applications/ml/quantized_mlp.py`](applications/ml/quantized_mlp.py) | ml | float32, bf16/fp16/posit16, fp8e5m2/posit8 | quantized MLP inference: 16-bit is ~lossless at half the memory; 8-bit trades accuracy for 4× size (posit8 > fp8e5m2) |

More application studies (DSP, control) are being added the same way — a
subdirectory per domain, each script self-asserting and CI-run.

Examples that need MTL5's accelerated linear-algebra solvers belong in
[`mtl5-python`](https://github.com/stillwater-sc/mtl5-python), not here — this set
stays pure NumPy + `universal_dtypes`.
