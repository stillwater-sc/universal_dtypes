"""Python-scalar promotion in ufuncs — issue #55.

Before this, NumPy dispatched every ufunc on the *exact* DType signature of its
operands. A Python scalar arrives as one of the abstract weak DTypes (NEP 50),
which matched no registered loop, so `a * 2` raised UFuncTypeError even though
`np.promote_types(posit16, float64)` already said posit16 wins. Two things fixed
it: `common_dtype` naming those abstract DTypes (which also repairs
`np.result_type(a, 2)` and everything built on it, e.g. `np.where`), and a
promoter per ufunc rewriting the mixed signature to the all-this-type one.

Python `bool` needed one extra piece: it is *not* weak — NumPy maps `True`
straight to the concrete BoolDType — so `a * True` performs a real bool -> this
type cast, which a ufunc checks at same_kind. That single cast level was
relaxed; float and int inbound casts remain UNSAFE deliberately.

These pin what absorbs, what deliberately does not, and the value semantics of
converting the scalar into the type first.
"""

import numpy as np
import pytest

import universal_dtypes as ud

ALL = list(ud.dtypes.items())

# In range for every format, including the fractional DSP types (range ±1).
GRID = [0.5, 0.25, -0.75]

BINARY_ARITH = [
    "add",
    "subtract",
    "multiply",
    "true_divide",
    "power",
    "minimum",
    "maximum",
    "fmin",
    "fmax",
]
COMPARISONS = ["equal", "not_equal", "less", "less_equal", "greater", "greater_equal"]


def _arr(v, T):
    return np.array(v, dtype=T)


# ---- the reported bug -------------------------------------------------------


@pytest.mark.parametrize("name,T", ALL)
def test_multiply_by_python_int(name, T):
    """`a * 2` — the exact expression from #55."""
    a = _arr(GRID, T)
    got = a * 2
    assert got.dtype == np.dtype(T)
    # Equivalent to converting the scalar into the type, then multiplying.
    assert got.tobytes() == (a * _arr(2.0, T)).tobytes()


@pytest.mark.parametrize("name,T", ALL)
@pytest.mark.parametrize("scalar", [2, 2.0, True])
def test_binary_arith_with_scalar(name, T, scalar):
    """Every registered binary arithmetic ufunc absorbs an absorbable scalar."""
    a = _arr(GRID, T)
    typed = _arr(float(scalar), T)
    for uname in BINARY_ARITH:
        uf = getattr(np, uname)
        got = uf(a, scalar)
        assert got.dtype == np.dtype(T), f"{uname} returned {got.dtype}"
        assert got.tobytes() == uf(a, typed).tobytes(), f"{uname} value mismatch"


@pytest.mark.parametrize("name,T", ALL)
@pytest.mark.parametrize("scalar", [2, 2.0, True])
def test_comparisons_with_scalar(name, T, scalar):
    a = _arr(GRID, T)
    typed = _arr(float(scalar), T)
    for uname in COMPARISONS:
        uf = getattr(np, uname)
        got = uf(a, scalar)
        assert got.dtype == np.dtype(bool), f"{uname} returned {got.dtype}"
        np.testing.assert_array_equal(got, uf(a, typed))


@pytest.mark.parametrize("name,T", ALL)
def test_reflected_operand_order(name, T):
    """The scalar may be on either side."""
    a = _arr(GRID, T)
    assert (2 * a).tobytes() == (a * 2).tobytes()
    np.testing.assert_array_equal(2 < a, a > 2)
    assert (2 - a).dtype == np.dtype(T)
    assert (2 / _arr([0.5, 0.25, 0.5], T)).dtype == np.dtype(T)


@pytest.mark.parametrize("name,T", ALL)
def test_clip_with_scalar_bounds(name, T):
    """clip is a 3-operand ufunc: any mix of bounds may be scalars."""
    a = _arr(GRID, T)
    lo, hi = _arr(-0.5, T), _arr(0.5, T)
    assert np.clip(a, -0.5, 0.5).tobytes() == np.clip(a, lo, hi).tobytes()
    assert np.clip(a, -0.5, hi).tobytes() == np.clip(a, lo, hi).tobytes()
    assert np.clip(a, lo, 0.5).tobytes() == np.clip(a, lo, hi).tobytes()
    assert np.clip(a, -0.5, 0.5).dtype == np.dtype(T)


@pytest.mark.parametrize("name,T", ALL)
def test_inplace_and_zero_dim(name, T):
    a = _arr(GRID, T)
    b = a.copy()
    b *= 2
    assert b.dtype == np.dtype(T)
    assert b.tobytes() == (a * 2).tobytes()
    zero_d = np.array(0.5, dtype=T)
    assert np.asarray(zero_d * 2).dtype == np.dtype(T)


# ---- promotion level (np.result_type and everything built on it) ------------


@pytest.mark.parametrize("name,T", ALL)
def test_result_type_with_python_scalar(name, T):
    """common_dtype must name the abstract weak DTypes; np.where et al. need it."""
    a = _arr(GRID, T)
    assert np.result_type(a, 2) == np.dtype(T)
    assert np.result_type(a, 2.0) == np.dtype(T)


@pytest.mark.parametrize("name,T", ALL)
def test_where_with_scalar(name, T):
    a = _arr(GRID, T)
    got = np.where(a > 0, a, 0.25)
    assert got.dtype == np.dtype(T)


# ---- what deliberately does NOT absorb --------------------------------------


@pytest.mark.parametrize("name,T", ALL)
def test_complex_scalar_refused(name, T):
    """Absorbing complex would silently drop the imaginary part."""
    a = _arr(GRID, T)
    with pytest.raises(TypeError):
        a * 2j
    with pytest.raises(TypeError):
        np.result_type(a, 2j)


@pytest.mark.parametrize("name,T", ALL)
def test_concrete_builtin_operand_refused(name, T):
    """A concrete builtin operand still raises — rounding a float64 array into a
    low-precision type is a data-loss decision the caller makes with .astype()."""
    a = _arr(GRID, T)
    with pytest.raises(TypeError):
        a * np.float64(2)
    with pytest.raises(TypeError):
        a * np.array([2.0, 2.0, 2.0])
    with pytest.raises(TypeError):
        a * np.arange(3)


@pytest.mark.parametrize("name,T", ALL)
def test_casting_kwarg_matches_numpy_weak_scalar_behavior(name, T):
    """A weak scalar involves no cast at all — common_dtype resolves it and NumPy
    builds it directly in this dtype — so every casting= level accepts it, just
    as it does for NumPy's own float16. (Pinned because the obvious guess, that
    casting='safe' rejects it, is wrong.)"""
    a = _arr(GRID, T)
    ref = np.array([1.0], dtype=np.float16)
    for level in ("safe", "same_kind", "unsafe", "no"):
        np.add(ref, 2, casting=level)  # NumPy's own behavior: accepted
        got = np.add(a, 2, casting=level)
        assert got.dtype == np.dtype(T)


def test_promoters_do_not_hijack_other_dtypes():
    """Promoters are registered globally on each ufunc, so they must only fire
    when one operand is a universal dtype — never for plain NumPy work."""
    assert 2 * 3 == 6
    assert np.float32(1) * 2 == np.float32(2)
    np.testing.assert_array_equal(np.arange(3) * 2, [0, 2, 4])
    assert (np.arange(3) * 2).dtype == np.dtype(np.arange(3).dtype)
    assert np.float16(1) + 1.5 == np.float16(2.5)
    assert np.result_type(np.int8, 2) == np.int8


# ---- value semantics: the scalar is converted into the type first -----------


def test_bounded_format_saturates_not_doubles():
    """Pinned: for a bounded format the scalar is converted into the type before
    the operation, so it saturates rather than scaling. `q15 * 2` multiplies by
    maxpos (~0.99997) — it does NOT double, and it does NOT raise. This matches
    the explicit path np.array(2.0, dtype=q15), which saturates identically."""
    for n in ("q7", "q15", "q31"):
        T = ud.dtypes[n]
        a = _arr([0.5], T)
        maxpos = float(_arr([1e30], T)[0])  # saturation target
        assert maxpos < 1.0
        got = a * 2
        assert got.tobytes() == (a * _arr(2.0, T)).tobytes()
        assert float(got[0]) == pytest.approx(0.5 * maxpos, rel=1e-2)


@pytest.mark.parametrize("name,T", ALL)
def test_scalar_rounds_into_type(name, T):
    """A scalar not representable in the type rounds/saturates by the type's own
    rules — identical to converting it explicitly."""
    a = _arr([0.5], T)
    for scalar in (0.1, 1 / 3):
        assert (a * scalar).tobytes() == (a * _arr(scalar, T)).tobytes()


def test_nan_survives_promotion():
    """A type's NaN/NaR propagates through a scalar operation."""
    for name, T in ALL:
        nan_arr = _arr([np.nan], T)
        if not np.isnan(float(nan_arr[0])):
            continue  # type has no NaN encoding
        assert np.isnan(nan_arr * 2)[0]


# ---- the reduction contract must survive (issues #48/#50) -------------------


@pytest.mark.parametrize("name,T", ALL)
def test_reduction_contract_unaffected(name, T):
    """mean/std and an explicit wider accumulator still raise: their divisor and
    accumulator are *concrete* dtypes, not weak scalars, so promotion does not
    quietly re-enable what the reduction contract froze out."""
    a = _arr(GRID, T)
    with pytest.raises(TypeError):
        np.mean(a)
    with pytest.raises(TypeError):
        np.sum(a, dtype=np.float64)
    # the supported route still works
    assert a.astype(np.float64).mean() == pytest.approx(np.mean([float(x) for x in a]))
