"""bfloat16 vs float16: same 16 bits, very different dynamic range.

bfloat16 and IEEE float16 are both 16-bit, but they split the bits differently:

    bfloat16 : 1 sign / 8 exponent / 7 mantissa   -> float32's range, ~3.4e38
    float16  : 1 sign / 5 exponent / 10 mantissa  -> range only to 65504

bfloat16 trades mantissa precision for exponent range. That is exactly why it is
the default for deep-learning training: activations and gradients that overflow
float16 stay finite in bfloat16 (you lose precision, not the value).

    python examples/bfloat16/dynamic_range.py
"""

import numpy as np

import universal_dtypes as ud


def main():
    big = 1e20
    fp16_big = float(np.array([big], dtype=ud.fp16)[0])
    bf16_big = float(np.array([big], dtype=ud.bfloat16)[0])
    print(f"representing {big:g}:")
    print(f"  fp16     -> {fp16_big}   (overflow: float16 tops out at 65504)")
    print(f"  bfloat16 -> {bf16_big:g}   (finite: keeps float32-like range)")

    # the flip side: near 1.0, float16's extra mantissa bits win
    x = 1.0 + 2.0**-9
    fp16_x = float(np.array([x], dtype=ud.fp16)[0])
    bf16_x = float(np.array([x], dtype=ud.bfloat16)[0])
    print(f"\nrepresenting 1 + 2^-9 = {x}:")
    print(f"  fp16     -> {fp16_x}   (10 mantissa bits: resolves it)")
    print(f"  bfloat16 -> {bf16_x}   (7 mantissa bits: rounds to 1.0)")

    assert np.isinf(np.array([big], dtype=ud.fp16))[0], "fp16 should overflow at 1e20"
    assert np.isfinite(np.array([big], dtype=ud.bfloat16))[0], "bfloat16 should stay finite"
    assert abs(fp16_x - x) < abs(bf16_x - x), "fp16 should be finer near 1.0"
    print("\nOK: bfloat16 trades mantissa precision for float32-class range.")


if __name__ == "__main__":
    main()
