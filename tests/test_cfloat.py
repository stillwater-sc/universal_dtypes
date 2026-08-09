"""cfloat NumPy dtype family — tests (issue #8).

The cfloat configs are registered through the reusable NEP-42 harness. Both
shipped configs are chosen for exact parity with a reference:

- ``fp16``    (cfloat<16,5>) == ``numpy.float16``
- ``fp8e5m2`` (cfloat<8,5>)  == ``ml_dtypes.float8_e5m2``
"""

import numpy as np
import pytest

import universal_dtypes as ud

try:
    import ml_dtypes

    HAVE_ML = True
except ImportError:
    HAVE_ML = False

CFLOATS = [
    ("fp16", ud.fp16, 2),
    ("fp8e5m2", ud.fp8e5m2, 1),
]


@pytest.mark.parametrize("name,scalar,itemsize", CFLOATS)
def test_dtype_resolves(name, scalar, itemsize):
    dt = np.dtype(scalar)
    assert dt.itemsize == itemsize
    assert np.dtype(name) == dt  # string-name resolution


@pytest.mark.parametrize("name,scalar,itemsize", CFLOATS)
def test_arithmetic(name, scalar, itemsize):
    a = np.array([1.0, 2.0, 3.0, 0.5], dtype=scalar)
    b = np.array([1.0, 2.0, 1.0, 0.5], dtype=scalar)
    np.testing.assert_array_equal((a + b).astype(np.float64), [2.0, 4.0, 4.0, 1.0])
    np.testing.assert_array_equal((a * b).astype(np.float64), [1.0, 4.0, 3.0, 0.25])
    np.testing.assert_array_equal((-a).astype(np.float64), [-1.0, -2.0, -3.0, -0.5])


@pytest.mark.parametrize("name,scalar,itemsize", CFLOATS)
def test_special_values(name, scalar, itemsize):
    # cfloat is IEEE-style: has both inf and NaN.
    v = np.array([np.inf, -np.inf, np.nan, 1.0], dtype=scalar)
    np.testing.assert_array_equal(np.isinf(v), [True, True, False, False])
    np.testing.assert_array_equal(np.isnan(v), [False, False, True, False])
    np.testing.assert_array_equal(np.isfinite(v), [False, False, False, True])
    out = v.astype(np.float64)
    assert np.isinf(out[0]) and out[0] > 0
    assert np.isinf(out[1]) and out[1] < 0
    assert np.isnan(out[2]) and out[3] == 1.0


@pytest.mark.parametrize("name,scalar,itemsize", CFLOATS)
def test_reduction_and_sort(name, scalar, itemsize):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=scalar)
    assert float(np.sum(a)) == 10.0
    c = np.array([3.0, 1.0, 2.0], dtype=scalar)
    np.testing.assert_array_equal(np.sort(c).astype(np.float64), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(np.argsort(c), [1, 2, 0])


@pytest.mark.parametrize("name,scalar,itemsize", CFLOATS)
def test_pickle_roundtrip(name, scalar, itemsize):
    import pickle

    a = np.array([1.0, 2.0, 3.0, 0.5], dtype=scalar)
    b = pickle.loads(pickle.dumps(a))
    assert b.dtype == np.dtype(scalar)
    np.testing.assert_array_equal(b.astype(np.float64), a.astype(np.float64))
    assert float(pickle.loads(pickle.dumps(scalar(2.5)))) == 2.5


@pytest.mark.parametrize("name,scalar,itemsize", CFLOATS)
def test_math_ufuncs(name, scalar, itemsize):
    a = np.array([1.0, 4.0, 0.25], dtype=scalar)
    np.testing.assert_array_equal(np.sqrt(a).astype(np.float64), [1.0, 2.0, 0.5])


def test_fp16_matches_numpy_float16():
    # fp16 conversion is bit-exact with numpy.float16 across the range.
    grid = np.linspace(-70000, 70000, 5001, dtype=np.float64)
    ours = np.array(grid, dtype=ud.fp16).astype(np.float64)
    with np.errstate(over="ignore"):
        theirs = grid.astype(np.float16).astype(np.float64)
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.skipif(not HAVE_ML, reason="ml_dtypes not installed")
def test_fp8e5m2_matches_ml_dtypes():
    # fp8e5m2 conversion is bit-exact with ml_dtypes.float8_e5m2 over the domain.
    grid = np.linspace(-70000, 70000, 6001, dtype=np.float64)
    ours = np.array(grid, dtype=ud.fp8e5m2).astype(np.float64)
    with np.errstate(over="ignore"):
        theirs = grid.astype(ml_dtypes.float8_e5m2).astype(np.float64)
    np.testing.assert_array_equal(ours, theirs)


def test_cfloat_registry():
    assert set(ud.cfloat_dtypes) == {name for name, _, _ in CFLOATS}
    for name, scalar, _ in CFLOATS:
        assert ud.cfloat_dtypes[name] is scalar
    # rolled into the top-level registry too
    assert set(ud.cfloat_dtypes) <= set(ud.dtypes)
