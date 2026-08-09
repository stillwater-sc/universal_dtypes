"""The danger of running a Kalman filter in low-precision floating point.

A Kalman filter carries an error-covariance matrix P and updates it every step
with P = (I - K H) P. That recursion subtracts nearly-equal quantities, so it is
exquisitely sensitive to rounding: lose enough mantissa bits and P stops being a
valid covariance, the gain goes wrong, and the estimate diverges. This is a
classic trap when someone "quantizes the filter to fp16/fp8 to save cycles".

Here we track a constant-velocity target from noisy position measurements and run
the *same* filter in several formats, reporting the position RMSE against ground
truth. The measurement-only RMSE is the "did the filter even help?" baseline.

Typical result:

    measurement-only  ~ 0.46      (raw sensor)
    float64           ~ 0.14      (filter works)
    fp16              ~ 0.14      (10 mantissa bits: safe)
    posit16           ~ 0.14      (tapered: safe)
    bfloat16          ~ 0.92      (7 mantissa bits: WORSE than not filtering)
    fp8e5m2           ~ 44        (2 mantissa bits: diverges catastrophically)

The danger isn't bit width — it's mantissa bits in the covariance recursion.
bfloat16 is 16 bits yet degrades the estimate below the raw sensor; fp8 blows up.

    python examples/applications/control/kalman_precision.py
"""

import numpy as np

import universal_dtypes as ud

DT = 1.0
STEPS = 300
MEAS_NOISE = 0.5
Q = 1e-4  # process-noise variance
R = MEAS_NOISE**2  # measurement-noise variance

_rng = np.random.default_rng(1)
_truth = np.cumsum(np.full(STEPS, DT))  # constant velocity 1.0
_meas = _truth + _rng.standard_normal(STEPS) * MEAS_NOISE


def kalman_rmse(dtype):
    """Run the constant-velocity Kalman filter entirely in `dtype`."""

    def a(x):  # shape-(1,) array keeps every op in the registered ufuncs
        return np.array([float(x)], dtype=dtype)

    x_pos, x_vel = a(0.0), a(0.0)
    p00, p01, p10, p11 = a(1.0), a(0.0), a(0.0), a(1.0)
    q, r, one, d = a(Q), a(R), a(1.0), a(DT)

    est = []
    for zk in _meas:
        z = a(zk)
        # predict: x = F x  (F = [[1, dt], [0, 1]]);  P = F P Fᵀ + Q
        x_pos = x_pos + x_vel * d
        n00 = p00 + d * p10 + d * (p01 + d * p11) + q
        n01 = p01 + d * p11
        n10 = p10 + d * p11
        n11 = p11 + q
        p00, p01, p10, p11 = n00, n01, n10, n11
        # update: scalar measurement, H = [1, 0]
        s = p00 + r
        k0 = p00 / s
        k1 = p10 / s
        y = z - x_pos
        x_pos = x_pos + k0 * y
        x_vel = x_vel + k1 * y
        # P = (I - K H) P  — the cancellation-prone step
        m00 = (one - k0) * p00
        m01 = (one - k0) * p01
        m10 = p10 - k1 * p00
        m11 = p11 - k1 * p01
        p00, p01, p10, p11 = m00, m01, m10, m11
        est.append(float(x_pos[0]))

    return float(np.sqrt(np.mean((np.array(est) - _truth) ** 2)))


def main():
    meas_rmse = float(np.sqrt(np.mean((_meas - _truth) ** 2)))
    formats = [
        ("float64", np.float64),
        ("fp16", ud.fp16),
        ("posit16", ud.posit16),
        ("bfloat16", ud.bfloat16),
        ("fp8e5m2", ud.fp8e5m2),
    ]
    print(f"measurement-only RMSE = {meas_rmse:.3f}  (baseline: filter must beat this)\n")
    rmse = {}
    for name, dt in formats:
        rmse[name] = kalman_rmse(dt)
        verdict = (
            "diverged"
            if rmse[name] > meas_rmse * 3
            else ("degraded" if rmse[name] > meas_rmse else "safe")
        )
        print(f"  {name:9} RMSE = {rmse[name]:10.3f}   [{verdict}]")

    # float64, fp16, posit16: the filter helps (beats the raw sensor)
    assert rmse["float64"] < meas_rmse
    assert rmse["fp16"] < meas_rmse
    assert rmse["posit16"] < meas_rmse
    # bfloat16 (16-bit but only 7 mantissa bits): worse than not filtering
    assert rmse["bfloat16"] > meas_rmse
    # fp8e5m2: catastrophic divergence
    assert rmse["fp8e5m2"] > meas_rmse * 10
    print("\nOK: fp8 diverges and bfloat16 degrades below the raw sensor — the")
    print("    covariance recursion needs mantissa bits fp16/posit16 have and they don't.")


if __name__ == "__main__":
    main()
