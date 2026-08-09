"""TI IQmath-style control loop in iq24 (Q8.24).

TI's C2000 real-time control MCUs run without a hardware FPU; their IQmath library
does control math in fixed-point IQ formats, with IQ24 (Q8.24) the usual global
default. IQ24 is the sweet spot for control: 24 fractional bits for precision plus
8 integer bits of range (±128) for setpoints, error, and integrator state that a
purely *fractional* format (Q1.15, range ±1) simply cannot hold.

This runs a PI controller regulating a first-order plant to a setpoint of 10 —
a value already outside a Q1.15 range — in several formats:

    float64 : y -> 10.000  (reference)
    iq24    : y -> 10.000  (Q8.24: range AND precision — matches float64)
    q15     : y ->  0.571  (Q1.15: ±1 range can't represent the setpoint; stuck)

iq24 tracking float64 while q15 fails is the concrete reason IQmath uses IQ-format
fixed-point for control instead of a fractional format.

    python examples/applications/control/iqmath_pid.py
"""

import numpy as np

import universal_dtypes as ud

SETPOINT = 10.0
STEPS = 300
KP, KI = 0.4, 0.4
PLANT_A, PLANT_B = 0.9, 0.1  # y[n+1] = 0.9 y[n] + 0.1 u[n]  (steady state: y = u)


def run_loop(dtype):
    """PI control loop run entirely in `dtype`; returns (final_output, peak_integrator)."""

    def a(x):
        return np.array([float(x)], dtype=dtype)

    y, integ = a(0.0), a(0.0)
    kp, ki, r, pa, pb = a(KP), a(KI), a(SETPOINT), a(PLANT_A), a(PLANT_B)
    peak = 0.0
    for _ in range(STEPS):
        e = r - y
        integ = integ + e
        u = kp * e + ki * integ
        y = pa * y + pb * u
        peak = max(peak, abs(float(integ[0])))
    return float(y[0]), peak


def main():
    print(f"PI control to setpoint {SETPOINT} (a value outside a Q1.15 ±1 range)\n")
    print(f"{'format':8} {'final y':>10} {'error':>10} {'integ peak':>12}")
    out = {}
    for name, dt in [("float64", np.float64), ("iq24", ud.iq24), ("q15", ud.q15)]:
        y, peak = run_loop(dt)
        out[name] = y
        print(f"{name:8} {y:10.4f} {abs(y - SETPOINT):10.3g} {peak:12.1f}")

    # iq24 regulates to the setpoint and matches float64 (range + precision)
    assert abs(out["iq24"] - SETPOINT) < 0.01
    assert abs(out["iq24"] - out["float64"]) < 0.01
    # q15's ±1 range cannot hold a setpoint of 10 — the loop cannot regulate
    assert abs(out["q15"] - SETPOINT) > 1.0
    print("\nOK: iq24 (Q8.24) has the range and precision for control; q15 (Q1.15) does not.")


if __name__ == "__main__":
    main()
