"""IIR limit cycles: when a recursive filter refuses to decay to zero.

A recursive (IIR) filter feeds its own quantized output back in. After the input
stops, the response should ring down to exactly zero — but in fixed point the
feedback term can round back to the *same* stored value, trapping the filter in a
small self-sustaining oscillation, or a stuck nonzero output, that never dies.
This "limit cycle" (a dead-band effect) is a classic fixed-point hazard the FIR
study can't show, because FIR filters have no feedback.

We drive a resonant 2nd-order section (poles at radius 0.97) with a single
impulse, then feed it zeros for thousands of samples and look at the tail:

    float64  : ~1e-33   — rings down to zero
    q31      : ~4e-9    — a tiny limit cycle (31 fractional bits -> small dead-band)
    q15      : ~2e-4    — a pronounced limit cycle: stuck, never reaches zero
    bfloat16 : ~1e-33   — decays (a float exponent lets small values keep shrinking)

Two lessons: (1) fixed point limit-cycles, and the dead-band shrinks with
fractional bits (q15 -> q31); (2) it isn't about bit width — bfloat16 is 16-bit
but its exponent lets the ring-down continue, where q15's uniform grid traps it.

    python examples/applications/dsp/iir_limit_cycles.py
"""

import numpy as np

import universal_dtypes as ud

R = 0.97  # pole radius (< 1: stable, should decay)
THETA = 2 * np.pi * 0.05  # resonant frequency
A1 = 2 * R * np.cos(THETA)
A2 = R * R
N = 3000
TAIL = 500  # samples at the end to measure the limit-cycle amplitude


def run(dtype):
    """y[n] = a1 y[n-1] - a2 y[n-2] + x[n], impulse then zeros, entirely in dtype."""

    def a(v):
        return np.array([float(v)], dtype=dtype)

    y1, y2 = a(0.0), a(0.0)
    a1, a2 = a(A1), a(A2)
    zero = a(0.0)
    impulse = a(0.5)
    out = np.empty(N)
    for i in range(N):
        x = impulse if i == 0 else zero
        y = a1 * y1 - a2 * y2 + x
        out[i] = float(y[0])
        y2, y1 = y1, y
    return out


def main():
    print(f"resonant biquad, poles at radius {R}; impulse then {N} samples\n")
    print(f"{'format':9} {'tail max|y|':>14}   behavior")
    tail = {}
    for name, dt in [
        ("float64", np.float64),
        ("q31", ud.q31),
        ("q15", ud.q15),
        ("bfloat16", ud.bfloat16),
    ]:
        out = run(dt)
        tail[name] = float(np.max(np.abs(out[-TAIL:])))
        behavior = "limit cycle" if tail[name] > 1e-6 else "decays to ~0"
        print(f"{name:9} {tail[name]:14.3e}   {behavior}")

    # float64 rings down to (essentially) zero
    assert tail["float64"] < 1e-20
    # fixed-point q15 is trapped in a pronounced limit cycle
    assert tail["q15"] > 1e-5
    # more fractional bits shrink the dead-band, but a fixed-point limit cycle
    # remains (still far above float64's clean decay)
    assert tail["q31"] < tail["q15"]
    assert tail["q31"] > tail["float64"]
    # bfloat16's exponent lets the ring-down continue — not a bit-width story
    assert tail["bfloat16"] < 1e-6
    print("\nOK: fixed point limit-cycles (dead-band shrinks q15 -> q31); a float")
    print("    exponent (bfloat16/float64) lets the ring-down reach zero.")


if __name__ == "__main__":
    main()
