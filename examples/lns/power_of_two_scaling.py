"""LNS: multiply and divide are add/subtract in the log domain — and power-of-two
scaling is exact.

The logarithmic number system stores a sign and the base-2 log of the magnitude.
So a * b becomes log(a) + log(b) and a / b becomes log(a) - log(b): multiply and
divide are cheap and never lose bits to a mantissa product. In particular,
scaling by a power of two (log2 is an integer, exactly representable) is *exact* —
scale a signal up by 2^k and back down and you recover it bit-for-bit, which is
the bread and butter of fixed-point DSP gain staging.

(The trade-off is the other direction: add/subtract are the hard, approximate
operations in LNS — see the tolerances in tests/test_lns.py.)

    python examples/lns/power_of_two_scaling.py
"""

import numpy as np

import universal_dtypes as ud


def main():
    signal = np.array([1.3, -2.7, 0.4, 5.1, 123.5, 0.001], dtype=ud.lns16)

    print("scale by 2^k then by 2^-k, check exact round-trip (lns16):")
    all_exact = True
    for k in range(-4, 8):
        factor = np.array([2.0**k] * signal.size, dtype=ud.lns16)
        back = (signal * factor) / factor
        exact = bool((back == signal).all())
        all_exact = all_exact and exact
        print(f"  k={k:+d}: exact={exact}")

    # multiply of powers of two is exact too (log2 is an integer)
    a = np.array([2.0, 0.25, 8.0], dtype=ud.lns16)
    b = np.array([16.0, 4.0, 0.5], dtype=ud.lns16)
    prod = (a * b).astype(np.float64)
    print(f"\n2*16, 0.25*4, 8*0.5 = {prod.tolist()}  (exact)")

    assert all_exact, "power-of-two scaling must round-trip exactly in LNS"
    np.testing.assert_array_equal(prod, [32.0, 1.0, 4.0])
    print("\nOK: power-of-two scaling is exact; multiply/divide live in the log domain.")


if __name__ == "__main__":
    main()
