"""Reduction / accumulation contract — issue #48.

Pins the frozen v2.0.0 semantics: in-type accumulation (no hidden wider
accumulator), empty-input identities, and the "cast first" rule for wider
accumulation and `mean`. See docs/dtypes.md "Reductions and the accumulation
contract".
"""

import numpy as np
import pytest

import universal_dtypes as ud

# Several cases here intentionally use values the ±1 fractional formats cannot hold
# (e.g. the scalar 2), which now reports saturation — see #60 and
# test_saturation_warning.py. That signal is not what these tests are about.
pytestmark = pytest.mark.filterwarnings(
    "ignore:(value .* out of range|overflow encountered in cast)"
)

ALL = list(ud.dtypes.items())
GRID = [0.5, 0.25, 0.75, -0.5, 0.125, 0.375]


def _arr(v, T):
    return np.array(v, dtype=T)


# ---- in-type sum / prod -----------------------------------------------------


@pytest.mark.parametrize("name,T", ALL)
def test_sum_prod_stay_in_type(name, T):
    a = _arr(GRID, T)
    s, p = np.sum(a), np.prod(a)
    # the reduction stays in-type: it returns a scalar of this dtype's scalar type
    assert type(s) is T and type(p) is T
    # not NaN (the values are ordinary); actual value depends on the type's range
    assert not np.isnan(float(s)) and not np.isnan(float(p))


def test_sum_value_in_type():
    # a type with enough range holds the sum exactly for this grid
    a = _arr(GRID, ud.posit16)
    assert np.isclose(float(np.sum(a)), 1.5, atol=1e-2)


# ---- empty-input identities -------------------------------------------------


@pytest.mark.parametrize("name,T", ALL)
def test_empty_sum_prod_identities(name, T):
    empty = _arr([], T)
    # the identity is the type's representation of 0 / 1 (the fractional DSP
    # formats can't hold 1.0 exactly, so prod([]) is their nearest to 1).
    assert float(np.sum(empty)) == float(T(0.0))
    assert float(np.prod(empty)) == float(T(1.0))


@pytest.mark.parametrize("name,T", ALL)
def test_empty_min_max_raise(name, T):
    empty = _arr([], T)
    with pytest.raises(ValueError):
        np.min(empty)
    with pytest.raises(ValueError):
        np.max(empty)


# ---- mean and dtype= : cast-first contract ----------------------------------


@pytest.mark.parametrize("name,T", ALL)
def test_mean_requires_explicit_accumulation(name, T):
    a = _arr(GRID, T)
    # in-type mean is unsupported by design (no divide-by-count loop)
    with pytest.raises(TypeError):
        np.mean(a)
    with pytest.raises(TypeError):
        np.mean(a, dtype=np.float64)
    # the supported path works and returns a finite float
    assert np.isfinite(a.astype(np.float64).mean())


@pytest.mark.parametrize("name,T", ALL)
def test_reduction_dtype_override_unsupported(name, T):
    a = _arr(GRID, T)
    # a wider accumulation dtype is not selected implicitly
    with pytest.raises(TypeError):
        np.sum(a, dtype=np.float64)
    # supported: cast first (equals reducing the cast array directly)
    assert a.astype(np.float64).sum() == np.sum(a.astype(np.float64))


# ---- naive accumulation swamps (documents "no hidden wider accumulator") ----


def test_in_type_accumulation_swamps():
    a = np.array([100.0] + [0.01] * 50, dtype=ud.posit16)
    in_type = float(np.sum(a))
    wide = float(np.sum(a.astype(np.float64)))
    assert in_type == 100.0  # the small addends vanish in posit16
    assert wide > 100.4  # accumulating wider keeps them


# ---- full-precision min/max reduction on a cascade --------------------------


def test_min_max_reduction_full_precision_on_cascade():
    a = _arr([1.0], ud.dd_cascade)
    hi = a + _arr([2.0**-80], ud.dd_cascade)  # 1 + 2^-80, invisible to float64
    both = np.concatenate([a, hi])
    mn = np.array([np.min(both)], dtype=ud.dd_cascade)
    mx = np.array([np.max(both)], dtype=ud.dd_cascade)
    assert mn.tobytes() == a.tobytes()
    assert mx.tobytes() == hi.tobytes()
