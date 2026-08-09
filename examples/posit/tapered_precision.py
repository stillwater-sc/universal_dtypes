"""Posit tapered precision: more accuracy near +-1, and NaR instead of inf/nan.

Posits spend their bits unevenly: values near magnitude 1.0 get more fraction
bits than a same-width IEEE float, at the cost of precision far from 1. For
numbers clustered around 1 (normalized weights, probabilities, ratios) posit16 is
noticeably more accurate than IEEE float16.

Posits also have a single exceptional value, NaR ("Not a Real") — there is no
+-inf and no signed nan. 1/0 is NaR, which ``np.isnan`` reports (and ``np.isinf``
never fires).

    python examples/posit/tapered_precision.py
"""

import numpy as np

import universal_dtypes as ud


def mean_relerr(dtype, ref):
    q = np.array(ref, dtype=dtype).astype(np.float64)
    return float(np.mean(np.abs(q - ref) / ref))


def main():
    grid = np.linspace(1.0, 2.0, 20001)[:-1]  # values in [1, 2)
    p = mean_relerr(ud.posit16, grid)
    f = mean_relerr(ud.fp16, grid)
    print("mean relative error over [1, 2):")
    print(f"  posit16 = {p:.3e}")
    print(f"  fp16    = {f:.3e}   (IEEE half)")
    print(f"posit16 is {f / p:.2f}x more accurate near 1.0")

    # a value posit16 represents exactly but fp16 cannot resolve
    x = 1.0 + 2.0**-11
    px = float(np.array([x], dtype=ud.posit16)[0])
    fx = float(np.array([x], dtype=ud.fp16)[0])
    print(f"\n1 + 2^-11 = {x}")
    print(f"  posit16 -> {px}   (exact: {px == x})")
    print(f"  fp16    -> {fx}   (rounded away)")

    # NaR: posit has no infinity; 1/0 is NaR, reported by isnan
    nar = np.array([1.0], dtype=ud.posit16) / np.array([0.0], dtype=ud.posit16)
    print(f"\nposit16 1/0 -> NaR:  isnan={bool(np.isnan(nar)[0])}  isinf={bool(np.isinf(nar)[0])}")

    assert p < f, "posit16 should be more accurate than fp16 near 1.0"
    assert px == x and fx != x
    assert bool(np.isnan(nar)[0]) and not bool(np.isinf(nar)[0])
    print("\nOK: tapered precision near 1.0, and NaR (no inf) semantics.")


if __name__ == "__main__":
    main()
