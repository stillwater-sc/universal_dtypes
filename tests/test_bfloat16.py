"""bfloat16 NumPy dtype — MVP tests (issue #3).

Cross-validated against ml_dtypes.bfloat16 where available (the reference oracle).
"""

import numpy as np
import pytest

import universal_dtypes as ud

bf16 = ud.bfloat16

try:
    import ml_dtypes

    HAVE_ML = True
except ImportError:
    HAVE_ML = False


def test_dtype_resolves():
    dt = np.dtype(bf16)
    assert dt.itemsize == 2


def test_array_creation_and_roundtrip():
    a = np.array([1.0, 2.0, 3.0, 0.5], dtype=bf16)
    assert a.dtype == np.dtype(bf16)
    # exactly representable values round-trip through float
    np.testing.assert_array_equal(a.astype(np.float32), np.array([1, 2, 3, 0.5], np.float32))


def test_arithmetic():
    a = np.array([1.0, 2.0, 3.0], dtype=bf16)
    b = np.array([0.5, 0.5, 0.5], dtype=bf16)
    np.testing.assert_array_equal((a + b).astype(np.float32), np.array([1.5, 2.5, 3.5], np.float32))
    np.testing.assert_array_equal((a * b).astype(np.float32), np.array([0.5, 1.0, 1.5], np.float32))
    np.testing.assert_array_equal((a - b).astype(np.float32), np.array([0.5, 1.5, 2.5], np.float32))
    np.testing.assert_array_equal((-a).astype(np.float32), np.array([-1, -2, -3], np.float32))


def test_comparisons():
    a = np.array([1.0, 2.0, 3.0], dtype=bf16)
    b = np.array([1.0, 5.0, 1.0], dtype=bf16)
    np.testing.assert_array_equal(a == b, [True, False, False])
    np.testing.assert_array_equal(a < b, [False, True, False])


def test_reduction_sum():
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=bf16)
    assert float(np.sum(a)) == 10.0  # np.sum returns a bfloat16 scalar


def test_casts_from_int_and_double():
    a = np.array([1, 2, 3], dtype=np.int64).astype(bf16)
    np.testing.assert_array_equal(a.astype(np.float64), np.array([1.0, 2.0, 3.0]))


def test_pickle_roundtrip():
    import pickle

    a = np.array([1.0, 2.0, 3.0, 0.5], dtype=bf16)
    b = pickle.loads(pickle.dumps(a))
    assert b.dtype == np.dtype(bf16)
    np.testing.assert_array_equal(b.astype(np.float32), a.astype(np.float32))
    assert float(pickle.loads(pickle.dumps(bf16(2.5)))) == 2.5


@pytest.mark.skipif(HAVE_ML, reason="ml_dtypes also owns the 'bfloat16' name")
def test_string_name_resolves():
    # np.dtype("bfloat16") resolves to ours only when ml_dtypes isn't installed;
    # both packages share the name, and we deliberately don't clobber an existing
    # owner. Pickling doesn't rely on the name (it goes through the scalar type).
    assert np.dtype("bfloat16") == np.dtype(bf16)


@pytest.mark.skipif(not HAVE_ML, reason="ml_dtypes not installed")
def test_matches_ml_dtypes_rounding():
    # A grid of float32 values; bf16 rounding must match ml_dtypes bit-for-bit.
    rng = np.linspace(-10, 10, 501, dtype=np.float32)
    ours = rng.astype(bf16).astype(np.float32)
    theirs = rng.astype(ml_dtypes.bfloat16).astype(np.float32)
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.skipif(not HAVE_ML, reason="ml_dtypes not installed")
def test_matches_ml_dtypes_arithmetic():
    rng = np.linspace(-5, 5, 51, dtype=np.float32)
    a_ours, b_ours = rng.astype(bf16), rng[::-1].copy().astype(bf16)
    a_ml, b_ml = rng.astype(ml_dtypes.bfloat16), rng[::-1].copy().astype(ml_dtypes.bfloat16)
    for op in (np.add, np.subtract, np.multiply):
        np.testing.assert_array_equal(
            op(a_ours, b_ours).astype(np.float32),
            op(a_ml, b_ml).astype(np.float32),
        )
