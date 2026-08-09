"""cfloat (configurable float): the fp16 vs fp8e5m2 precision/size trade-off.

The cfloat family exposes IEEE-754-style floats at configurable widths. fp16 is
IEEE half; fp8e5m2 is an 8-bit float with the same 5-bit exponent (so a similar
dynamic range) but only 2 mantissa bits — a 2x storage saving for much coarser
precision. fp16 is bit-compatible with numpy.float16 and fp8e5m2 with
ml_dtypes.float8_e5m2.

This example quantizes a signal to each and compares the round-trip error — the
kind of measurement you make when choosing an activation format for on-device ML.

    python examples/cfloat/precision_tradeoff.py
"""

import numpy as np

import universal_dtypes as ud


def rms_relerr(dtype, ref):
    q = np.array(ref, dtype=dtype).astype(np.float64)
    return float(np.sqrt(np.mean(((q - ref) / ref) ** 2)))


def main():
    # a spread of positive magnitudes both formats can represent (no overflow)
    rng = np.linspace(0.01, 100.0, 5000)
    e16 = rms_relerr(ud.fp16, rng)
    e8 = rms_relerr(ud.fp8e5m2, rng)
    print("RMS relative quantization error over [0.01, 100]:")
    print(f"  fp16    (2 bytes) = {e16:.3e}")
    print(f"  fp8e5m2 (1 byte)  = {e8:.3e}   ({e8 / e16:.0f}x coarser, half the size)")

    print(
        f"\npi rounds to:  fp16 ={float(np.array([np.pi], dtype=ud.fp16)[0])}"
        f"   fp8e5m2 ={float(np.array([np.pi], dtype=ud.fp8e5m2)[0])}"
    )

    # both are IEEE-style: inf and nan behave
    sv = np.array([np.inf, np.nan], dtype=ud.fp8e5m2)
    print(f"fp8e5m2 specials: isinf={np.isinf(sv).tolist()} isnan={np.isnan(sv).tolist()}")

    assert e16 < e8, "fp16 should be finer than fp8e5m2"
    assert bool(np.isinf(sv)[0]) and bool(np.isnan(sv)[1])
    print("\nOK: fp8e5m2 halves storage for a bounded precision cost; both IEEE-style.")


if __name__ == "__main__":
    main()
