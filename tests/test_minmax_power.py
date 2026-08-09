"""power (**), minimum/maximum/fmin/fmax, and clip ufuncs — issues #41 and #40.

Before this, `init_ufuncs` registered no `power`, `minimum`, `maximum`, `fmin`,
`fmax`, or `clip` loop, so `a ** b`, `np.min`/`np.max`, `np.clip`, and
`np.nanmin`/`np.nanmax` had no in-type implementation. These verify the loops and
their NaN semantics.
"""

import numpy as np
import pytest

import universal_dtypes as ud

ALL = list(ud.dtypes.items())

# Values inside every format's range (the fractional DSP formats saturate ~1).
POS = [0.25, 0.5, 0.75]  # positive, so integer/real powers stay well defined
PAIR_A = [0.5, -0.25, 0.75, -0.6]
PAIR_B = [0.25, -0.5, 0.6, -0.75]

# Types that can represent a NaN (posit NaR, IEEE NaN, ...). Determined per type.
HAS_NAN = [(n, T) for n, T in ALL if np.isnan(float(np.array([np.nan], dtype=T)[0]))]


def _arr(v, T):
    return np.array(v, dtype=T)


# ---- power (#41) ------------------------------------------------------------


@pytest.mark.parametrize("name,T", ALL)
def test_power_matches_double_path(name, T):
    """np.power in-type == compute-in-double-then-round (its documented semantics)."""
    a = _arr(POS, T)
    b = _arr([2.0, 2.0, 2.0], T)
    got = np.power(a, b)
    ref = np.power(a.astype(np.float64), b.astype(np.float64)).astype(T)
    assert got.dtype == np.dtype(T)
    assert got.tobytes() == ref.tobytes()


@pytest.mark.parametrize("name,T", ALL)
def test_power_operator_scalar_exponent(name, T):
    """`a ** 2` works and stays in-type."""
    a = _arr(POS, T)
    got = a**2
    assert got.dtype == np.dtype(T)
    ref = (a.astype(np.float64) ** 2).astype(T)
    assert got.tobytes() == ref.tobytes()


# ---- minimum / maximum (#40) ------------------------------------------------


@pytest.mark.parametrize("name,T", ALL)
def test_minimum_maximum_elementwise(name, T):
    a, b = _arr(PAIR_A, T), _arr(PAIR_B, T)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    assert lo.dtype == np.dtype(T) and hi.dtype == np.dtype(T)
    af, bf = a.astype(np.float64), b.astype(np.float64)
    assert np.array_equal(lo.astype(np.float64), np.minimum(af, bf))
    assert np.array_equal(hi.astype(np.float64), np.maximum(af, bf))


@pytest.mark.parametrize("name,T", ALL)
def test_min_max_reductions(name, T):
    a = _arr(PAIR_A, T)
    af = a.astype(np.float64)
    assert float(np.min(a)) == af.min()
    assert float(np.max(a)) == af.max()


@pytest.mark.parametrize("name,T", ALL)
def test_clip(name, T):
    a = _arr(PAIR_A, T)
    lo, hi = T(-0.4), T(0.6)
    out = np.clip(a, lo, hi)
    assert out.dtype == np.dtype(T)
    lof, hif = float(lo), float(hi)
    of = out.astype(np.float64)
    assert (of >= lof).all() and (of <= hif).all()
    # elements already inside [lo, hi] are unchanged
    af = a.astype(np.float64)
    inside = (af >= lof) & (af <= hif)
    assert np.array_equal(of[inside], af[inside])


def test_minmax_full_precision_on_cascade():
    """min/max compare at the cascade's full precision, not through float64."""
    one = _arr([1.0], ud.dd_cascade)
    hi = one + _arr([2.0**-80], ud.dd_cascade)  # 1 + 2^-80, invisible to float64
    assert np.minimum(hi, one).tobytes() == one.tobytes()
    assert np.maximum(hi, one).tobytes() == hi.tobytes()


# ---- NaN semantics ----------------------------------------------------------


@pytest.mark.parametrize("name,T", HAS_NAN)
def test_minimum_maximum_propagate_nan(name, T):
    nan, v = T(float("nan")), T(0.5)
    assert np.isnan(float(np.maximum(nan, v)))
    assert np.isnan(float(np.minimum(nan, v)))
    assert np.isnan(float(np.maximum(v, nan)))


@pytest.mark.parametrize("name,T", HAS_NAN)
def test_fmin_fmax_suppress_nan(name, T):
    nan, v = T(float("nan")), T(0.5)
    assert float(np.fmax(nan, v)) == 0.5
    assert float(np.fmin(nan, v)) == 0.5
    assert float(np.fmax(v, nan)) == 0.5


# ---- mean: documents the accumulation contract ------------------------------


@pytest.mark.parametrize("name,T", ALL[:4])
def test_mean_contract_via_float64(name, T):
    """np.mean divides the in-type sum by an integer count, for which there is no
    in-type loop by design (the reduction/accumulation dtype is the caller's
    choice — see the v2.0.0 reduction-contract work). The supported path is an
    explicit accumulation dtype."""
    a = _arr(PAIR_A, T)
    with pytest.raises(TypeError):
        np.mean(a)  # no divide(T, int64) loop — deliberate
    # supported: accumulate in float64
    assert np.isclose(a.astype(np.float64).mean(), np.mean(a.astype(np.float64)))
