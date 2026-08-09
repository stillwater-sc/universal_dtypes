"""DSP integrator: fixed-point adds exactly where floating point drifts (and stalls).

Accumulators and integrators are everywhere in DSP — moving averages, DC/offset
estimation, the integral term of a PID loop. They add a stream of small samples
into a growing running sum, and that is exactly where number-system choice bites:

  * Fixed-point (fixpnt) addition is EXACT within range — it is just integer
    addition of the scaled values — so the running sum is bit-exact.
  * Floating point loses low bits every step once the accumulator is large
    relative to a sample. The error drifts, and in a low-mantissa format the
    accumulator can *stall completely*: when a sample is smaller than one ULP of
    the running sum, `sum + sample == sum` and integration stops (swamping).

Here we integrate 2000 small samples (each an exact multiple of 2^-8, so exactly
representable in fixpnt16's Q8.8) and compare each format's running sum against a
float64 reference.

    fixpnt16 : max error 0.00     — exact
    fp16     : max error ~0.37    — drifts
    posit16  : max error ~0.43    — drifts
    bfloat16 : max error ~28      — STALLS (only 7 mantissa bits; sum freezes ~8)

    python examples/applications/dsp/fixed_point_integrator.py
"""

import numpy as np

import universal_dtypes as ud

_rng = np.random.default_rng(0)
N = 2000
STEP = 2.0**-8  # fixpnt16 (Q8.8) resolution — samples land exactly on the grid
SAMPLES = _rng.integers(1, 9, size=N) * STEP  # each in [2^-8, 8·2^-8], partial sum < 128
REF = np.cumsum(SAMPLES.astype(np.float64))


def integrate(dtype):
    """Run the sample-by-sample running sum entirely in `dtype`."""
    s = SAMPLES.astype(dtype)
    acc = np.array([0.0], dtype=dtype)
    traj = np.empty(N, dtype=np.float64)
    for i in range(N):
        acc = acc + s[i : i + 1]
        traj[i] = float(acc[0])
    return traj


def main():
    print(f"integrating {N} samples; true final sum = {REF[-1]:.4f}\n")
    formats = [
        ("fixpnt16", ud.fixpnt16),
        ("fp16", ud.fp16),
        ("posit16", ud.posit16),
        ("bfloat16", ud.bfloat16),
    ]
    err = {}
    for name, dt in formats:
        traj = integrate(dt)
        err[name] = float(np.max(np.abs(traj - REF)))
        print(f"  {name:9} max running-sum error = {err[name]:8.4f}   final = {traj[-1]:8.4f}")

    # fixed-point integrates exactly (adds are exact within range)
    assert err["fixpnt16"] == 0.0
    # every floating format drifts, and fixpnt beats all of them
    for name in ("fp16", "posit16", "bfloat16"):
        assert err[name] > err["fixpnt16"]
    # bfloat16 doesn't just drift — it stalls (swamping), so its error is large
    assert err["bfloat16"] > 10.0
    print("\nOK: fixpnt16 integrates exactly; floating point drifts and bfloat16 stalls.")


if __name__ == "__main__":
    main()
