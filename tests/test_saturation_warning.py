"""Saturating conversions are reported, not silent — issue #60.

`q15_arr * 2` scales by ~0.99997 instead of doubling, because the scalar 2
saturates to maxpos on the way into the type. The value semantics are correct
and deliberate (see test_scalar_promotion), but doing it silently hid the
mistake: the result looks plausible.

Conversions that clamp now report. The two conversion paths use different
mechanisms, because only one of them survives:

- array -> array (`astype`, array operands) sets NumPy's floating-point error
  state, exactly as NumPy does for float16 overflow. np.errstate(over=...)
  selects ignore/warn/raise, and it aggregates to one report per cast.
- scalar -> element (the weak-scalar operand of `arr * 2`, np.array(x, dtype=T),
  np.full, the scalar constructor) emits a RuntimeWarning directly. NumPy clears
  the float status before running a ufunc loop, and the scalar operand is
  converted *before* that clear, so a status set there would be wiped.

Types with an infinity (cfloat, bfloat16) overflow to inf rather than saturating
and must stay silent.
"""

import warnings

import numpy as np
import pytest

import universal_dtypes as ud

SATURATION_MSG = "out of range"


def maxpos(T):
    """The type's saturation target, computed without tripping the warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.array([1e300], dtype=T)[0])


# The formats where a small scalar realistically saturates: range ±1.
FRACTIONAL = [(n, ud.dtypes[n]) for n in ("q7", "q15", "q31")]
# Types that carry an infinity, so out-of-range goes to inf, not maxpos.
HAS_INF = [(n, T) for n, T in ud.dtypes.items() if np.isinf(maxpos(T))]


# ---- the reported case ------------------------------------------------------


@pytest.mark.parametrize("name,T", FRACTIONAL)
def test_saturating_scalar_operand_warns(name, T):
    """The headline case from #60: `q15_arr * 2` must not be silent."""
    a = np.array([0.5, 0.25], dtype=T)
    with pytest.warns(RuntimeWarning, match=SATURATION_MSG):
        got = a * 2
    # The value semantics are unchanged — still saturate-and-multiply.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert got.tobytes() == (a * np.array(2.0, dtype=T)).tobytes()


@pytest.mark.parametrize("name,T", FRACTIONAL)
def test_saturating_scalar_warns_on_every_scalar_entry_point(name, T):
    """NumPy warns on every conversion entry point for float16 overflow, not
    only the operator; match that."""
    a = np.array([0.5], dtype=T)
    for label, fn in [
        ("operator", lambda: a * 2),
        ("reflected", lambda: 2 * a),
        ("comparison", lambda: a > 2),
        ("np.array", lambda: np.array(2.0, dtype=T)),
        ("np.full", lambda: np.full(2, 2.0, dtype=T)),
        ("scalar ctor", lambda: T(2)),
        ("clip bound", lambda: np.clip(a, 0, 2)),
    ]:
        with pytest.warns(RuntimeWarning, match=SATURATION_MSG):
            fn()


@pytest.mark.parametrize("name,T", FRACTIONAL)
def test_astype_reports_through_errstate(name, T):
    """The array path uses NumPy's float error state, so it is np.errstate
    controllable — including raise, which #60 listed as the stricter option."""
    src = np.array([2.0, 3.0])
    with pytest.warns(RuntimeWarning, match="overflow"):
        with np.errstate(over="warn"):
            src.astype(T)
    with np.errstate(over="ignore"):
        warnings.simplefilter("error")  # would fail if anything warned
        src.astype(T)
        warnings.resetwarnings()
    with np.errstate(over="raise"):
        with pytest.raises(FloatingPointError):
            src.astype(T)


@pytest.mark.parametrize("name,T", FRACTIONAL)
def test_arange_saturation_warns(name, T):
    """The other surface #60 covered."""
    with pytest.warns(RuntimeWarning, match=SATURATION_MSG):
        np.arange(3, dtype=T)


# ---- no false positives -----------------------------------------------------


@pytest.mark.parametrize("name,T", FRACTIONAL)
def test_in_range_scalar_is_silent(name, T):
    a = np.array([0.5, 0.25], dtype=T)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a * 0.5
        a + 0.25
        a > 0.5
        np.array(0.5, dtype=T)
        np.clip(a, -0.5, 0.5)


@pytest.mark.parametrize("name,T", FRACTIONAL)
def test_ordinary_rounding_is_not_saturation(name, T):
    """0.1 is not representable in these formats but is well in range — rounding
    is normal and must stay silent. Only clamping reports."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = np.array(0.1, dtype=T)
    assert float(got) != 0.1  # it really did round
    assert abs(float(got) - 0.1) < 0.01


@pytest.mark.parametrize("name,T", HAS_INF)
def test_types_with_infinity_never_warn(name, T):
    """cfloat/bfloat16 overflow to inf rather than saturating, so there is no
    clamping to report."""
    a = np.array([1.0], dtype=T)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a * 1e10
        np.array(1e300, dtype=T)


def test_wide_range_types_do_not_warn_for_small_scalars():
    """posit16 and iq24 hold 2 comfortably; only the genuinely bounded formats
    should ever report for an everyday scalar."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.array([1.0], dtype=ud.posit16) * 2
        np.array([1.0], dtype=ud.iq24) * 2  # range ±128
        np.array([1.0], dtype=ud.fixpnt8) * 2  # range ±8


# ---- the warning is informative --------------------------------------------


@pytest.mark.parametrize("name,T", FRACTIONAL)
def test_warning_names_the_dtype_and_bound(name, T):
    with pytest.warns(RuntimeWarning) as rec:
        np.array(2.0, dtype=T)
    msg = str(rec[0].message)
    assert name in msg, msg
    assert "saturated" in msg, msg


@pytest.mark.parametrize("name,T", FRACTIONAL)
def test_suppressible_by_the_warnings_module(name, T):
    """Users who know their data saturates can silence it the standard way."""
    a = np.array([0.5], dtype=T)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("ignore")
        got = a * 2
        assert len(w) == 0
    assert float(got[0]) == pytest.approx(0.5 * maxpos(T), rel=1e-2)
