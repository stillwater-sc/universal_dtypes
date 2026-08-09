"""posit<nbits,2> NumPy dtype family — tests (issue #7).

The posit family is registered through the reusable NEP-42 harness. Conversion is
cross-validated against Universal's own ``posit16_roundtrip`` (the C++ oracle);
arithmetic is checked on exactly representable operands and for NaR semantics.
"""

import numpy as np
import pytest

import universal_dtypes as ud

POSITS = [
    ("posit8", ud.posit8, 1),
    ("posit16", ud.posit16, 2),
    ("posit32", ud.posit32, 4),
    ("posit64", ud.posit64, 8),
]

# The full shipped set (issue #16): es variants + non-power-of-two widths. The
# itemsize is the smallest containing uint (12->2B, 20/24/28->4B, 40/48->8B).
POSITS_ALL = [
    ("posit8", ud.posit8, 1),
    ("posit16", ud.posit16, 2),
    ("posit32", ud.posit32, 4),
    ("posit64", ud.posit64, 8),
    ("posit8e0", ud.posit8e0, 1),
    ("posit8e1", ud.posit8e1, 1),
    ("posit16e1", ud.posit16e1, 2),
    ("posit12", ud.posit12, 2),
    ("posit20", ud.posit20, 4),
    ("posit24", ud.posit24, 4),
    ("posit28", ud.posit28, 4),
    ("posit40", ud.posit40, 8),
    ("posit48", ud.posit48, 8),
]


@pytest.mark.parametrize("name,scalar,itemsize", POSITS)
def test_dtype_resolves(name, scalar, itemsize):
    dt = np.dtype(scalar)
    assert dt.itemsize == itemsize
    # string-name resolution (acceptance criterion)
    assert np.dtype(name) == dt


@pytest.mark.parametrize("name,scalar,itemsize", POSITS)
def test_array_creation_and_roundtrip(name, scalar, itemsize):
    # small integers and halves are exactly representable in every posit<n,2>
    vals = [1.0, 2.0, 3.0, 0.5, -2.0, 0.25]
    a = np.array(vals, dtype=scalar)
    assert a.dtype == np.dtype(scalar)
    np.testing.assert_array_equal(a.astype(np.float64), np.array(vals))


@pytest.mark.parametrize("name,scalar,itemsize", POSITS)
def test_exact_arithmetic(name, scalar, itemsize):
    a = np.array([1.0, 2.0, 3.0, 0.5], dtype=scalar)
    b = np.array([1.0, 2.0, 1.0, 0.5], dtype=scalar)
    np.testing.assert_array_equal((a + b).astype(np.float64), [2.0, 4.0, 4.0, 1.0])
    np.testing.assert_array_equal((a * b).astype(np.float64), [1.0, 4.0, 3.0, 0.25])
    np.testing.assert_array_equal((a - b).astype(np.float64), [0.0, 0.0, 2.0, 0.0])
    np.testing.assert_array_equal((-a).astype(np.float64), [-1.0, -2.0, -3.0, -0.5])


@pytest.mark.parametrize("name,scalar,itemsize", POSITS)
def test_comparisons(name, scalar, itemsize):
    a = np.array([1.0, 2.0, 3.0], dtype=scalar)
    b = np.array([1.0, 5.0, 1.0], dtype=scalar)
    np.testing.assert_array_equal(a == b, [True, False, False])
    np.testing.assert_array_equal(a < b, [False, True, False])


@pytest.mark.parametrize("name,scalar,itemsize", POSITS)
def test_reduction_and_sort(name, scalar, itemsize):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=scalar)
    assert float(np.sum(a)) == 10.0  # 1+2+3+4 exact in posit<n,2>
    c = np.array([3.0, 1.0, 2.0], dtype=scalar)
    np.testing.assert_array_equal(np.sort(c).astype(np.float64), [1.0, 2.0, 3.0])


@pytest.mark.parametrize("name,scalar,itemsize", POSITS)
def test_nar_semantics(name, scalar, itemsize):
    # non-finite float -> NaR; isnan detects it; NaR -> float is nan
    n = np.array([np.inf, -np.inf, np.nan, 1.0], dtype=scalar)
    np.testing.assert_array_equal(np.isnan(n), [True, True, True, False])
    out = n.astype(np.float64)
    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isnan(out[2])
    assert out[3] == 1.0


@pytest.mark.parametrize("name,scalar,itemsize", POSITS)
def test_casts_from_int_and_double(name, scalar, itemsize):
    a = np.array([1, 2, 3], dtype=np.int64).astype(scalar)
    np.testing.assert_array_equal(a.astype(np.float64), [1.0, 2.0, 3.0])


def test_conversion_matches_universal_oracle():
    # posit16 dtype conversion must match Universal's posit16_roundtrip bit-for-bit.
    grid = np.linspace(-50, 50, 2001, dtype=np.float64)
    via_dtype = np.array(grid, dtype=ud.posit16).astype(np.float64)
    via_universal = np.array([ud.posit16_roundtrip(x) for x in grid])
    np.testing.assert_array_equal(via_dtype, via_universal)


def test_tapered_precision():
    # posit16 has more precision near +-1 than far away; a value near 1 round-trips
    # more accurately than a large value. (Sanity check on tapered precision.)
    near = np.array([1.0 + 1.0 / 512], dtype=ud.posit16).astype(np.float64)[0]
    assert abs(near - (1.0 + 1.0 / 512)) < 1e-3


@pytest.mark.parametrize("name,scalar,itemsize", POSITS)
def test_argsort(name, scalar, itemsize):
    a = np.array([3.0, 1.0, 2.0, 0.5], dtype=scalar)
    np.testing.assert_array_equal(np.argsort(a), [3, 1, 2, 0])


@pytest.mark.parametrize("name,scalar,itemsize", POSITS)
def test_pickle_roundtrip(name, scalar, itemsize):
    import pickle

    a = np.array([1.0, 2.0, 3.0, 0.5], dtype=scalar)
    b = pickle.loads(pickle.dumps(a))
    assert b.dtype == np.dtype(scalar)
    np.testing.assert_array_equal(b.astype(np.float64), a.astype(np.float64))
    # scalar pickle
    s = scalar(2.5)
    assert float(pickle.loads(pickle.dumps(s))) == 2.5


@pytest.mark.parametrize("name,scalar,itemsize", POSITS)
def test_math_ufuncs_exact(name, scalar, itemsize):
    # perfect squares are exactly representable in posit<n,2> for n >= 8
    a = np.array([1.0, 4.0, 0.25], dtype=scalar)
    np.testing.assert_array_equal(np.sqrt(a).astype(np.float64), [1.0, 2.0, 0.5])
    np.testing.assert_array_equal(
        np.square(np.array([2.0, 3.0], dtype=scalar)).astype(np.float64), [4.0, 9.0]
    )


def test_math_ufuncs_approx():
    # transcendental funcs: posit32 has enough precision to be close to the true value
    p32 = ud.posit32
    x = np.array([1.0, 2.0], dtype=p32)
    np.testing.assert_allclose(np.exp(x).astype(np.float64), np.exp([1.0, 2.0]), rtol=1e-5)
    np.testing.assert_allclose(
        np.log(np.array([1.0, np.e], dtype=p32)).astype(np.float64), [0.0, 1.0], atol=1e-5
    )


@pytest.mark.parametrize("name,scalar,itemsize", POSITS_ALL)
def test_all_configs(name, scalar, itemsize):
    # One end-to-end check per shipped config (es variants + non-power-of-two
    # widths). Uses only values exactly representable in every posit (near +-1).
    import pickle

    dt = np.dtype(scalar)
    assert dt.itemsize == itemsize
    assert np.dtype(name) == dt  # string-name resolution

    vals = [1.0, 2.0, 0.5, -1.0, 0.25]
    a = np.array(vals, dtype=scalar)
    np.testing.assert_array_equal(a.astype(np.float64), vals)

    # arithmetic (all operands/results exactly representable near +-1)
    b = np.array([0.0, 0.0, 0.5, 1.0, 0.25], dtype=scalar)
    np.testing.assert_array_equal((a + b).astype(np.float64), [1.0, 2.0, 1.0, 0.0, 0.5])

    # NaR + pickle
    assert bool(np.isnan(np.array([np.inf], dtype=scalar).astype(np.float64))[0])
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(a)).astype(np.float64), vals)
