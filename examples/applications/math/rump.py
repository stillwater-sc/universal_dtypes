"""Rump's example: when even double-double isn't enough.

In 1988 Siegfried Rump constructed an expression whose IEEE evaluation is
catastrophically wrong — not just imprecise, but the wrong sign and ~21 orders of
magnitude off:

    f(a, b) = 333.75 b^6 + a^2 (11 a^2 b^2 - b^6 - 121 b^4 - 2) + 5.5 b^8 + a/(2b)
    a = 77617,  b = 33096

The true value is f = -0.827396059946821368141165... The enormous terms (5.5 b^8
is ~8e36) nearly cancel, so all the significance lives in bits far below what
float64 carries. This example evaluates f across five precisions and shows the
answer only stabilizes at triple-double:

    float32   : ~ -6.3e29     (garbage)
    float64   : ~ -1.18e21    (garbage — the "double is fine" trap)
    dd_cascade: ~ +1.172604   (~106 bits: still WRONG — wrong sign!)
    td_cascade: ~ -0.827396   (~159 bits: correct)
    qd_cascade: ~ -0.827396   (~212 bits: correct)

The lesson: "use double" — or even "use double-double" — is not a precision
strategy. Know how many bits your problem actually needs.

    python examples/applications/math/rump.py
"""

import numpy as np

import universal_dtypes as ud

TRUE = -0.8273960599468214  # Rump's exact value, rounded to double
A, B = 77617.0, 33096.0


def rump(dtype):
    """Evaluate Rump's f(a, b) entirely in `dtype`."""

    def c(x):  # a scalar constant as a 1-element array in this dtype
        return np.array([x], dtype=dtype)

    a, b = c(A), c(B)
    b2 = b * b
    b4 = b2 * b2
    b6 = b4 * b2
    b8 = b4 * b4
    a2 = a * a
    f = (
        c(333.75) * b6
        + a2 * (c(11.0) * a2 * b2 - b6 - c(121.0) * b4 - c(2.0))
        + c(5.5) * b8
        + a / (c(2.0) * b)
    )
    return float(f[0])


def main():
    rows = [
        ("float32", np.float32),
        ("float64", np.float64),
        ("dd_cascade", ud.dd_cascade),
        ("td_cascade", ud.td_cascade),
        ("qd_cascade", ud.qd_cascade),
    ]
    print(f"Rump's expression, true value = {TRUE}\n")
    results = {}
    for name, dt in rows:
        val = rump(dt)
        results[name] = val
        ok = "OK" if abs(val - TRUE) < 1e-3 else "WRONG"
        print(f"  {name:11} = {val: .6g}   [{ok}]")

    # float64 is not just imprecise — it's astronomically wrong.
    assert abs(results["float64"] - TRUE) > 1e6
    # double-double (~106 bits) is STILL wrong — wrong sign, off by ~2.
    assert abs(results["dd_cascade"] - TRUE) > 1.0
    # triple-double and quad-double recover the true value.
    assert abs(results["td_cascade"] - TRUE) < 1e-3
    assert abs(results["qd_cascade"] - TRUE) < 1e-3
    print("\nOK: float64 AND double-double are wrong here; triple-double is the")
    print("    first precision that solves Rump's example.")


if __name__ == "__main__":
    main()
