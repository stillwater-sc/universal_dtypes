"""Catastrophic cancellation: where float64 loses digits and the cascades keep them.

Summing values that span many orders of magnitude is the classic failure mode of
fixed-precision floating point: the small terms fall off the end of the
accumulator and vanish. The double-double / triple-double / quad-double *cascade*
dtypes carry ~106 / ~159 / ~212 bits of significand, so they retain what float64
throws away.

Here we sum ``[1e16] + [1.0]*100 + [-1e16]``. The exact answer is 100 (the two
1e16 terms cancel), but in float64 each ``+ 1.0`` is lost against the 1e16 running
sum, so the result is badly wrong. Run:

    python examples/cascades/catastrophic_cancellation.py
"""

import numpy as np

import universal_dtypes as ud

TERMS = [1e16] + [1.0] * 100 + [-1e16]
EXACT = 100.0


def sum_in(dtype):
    return float(np.sum(np.array(TERMS, dtype=dtype)))


def main():
    f64 = float(np.sum(np.array(TERMS, dtype=np.float64)))
    dd = sum_in(ud.dd_cascade)
    td = sum_in(ud.td_cascade)
    qd = sum_in(ud.qd_cascade)

    print(f"exact sum                = {EXACT}")
    print(f"float64   np.sum         = {f64}   (error {abs(f64 - EXACT):g})")
    print(f"dd_cascade np.sum        = {dd}")
    print(f"td_cascade np.sum        = {td}")
    print(f"qd_cascade np.sum        = {qd}")

    # float64 gets it wrong; every cascade tier recovers the exact answer.
    assert f64 != EXACT, "expected float64 to lose precision here"
    assert dd == EXACT and td == EXACT and qd == EXACT
    print("\nOK: the cascades retain the small terms that float64 dropped.")


if __name__ == "__main__":
    main()
