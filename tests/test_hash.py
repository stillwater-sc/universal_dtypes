"""Scalar __hash__ — issue #42.

Scalars define __eq__ (via richcompare) but had no tp_hash, so they were
unhashable / hashed inconsistently with equality — unusable as dict/set keys. The
hash is defined on the scalar's value (the same value __eq__ compares), so it is
consistent with equality and matches Python's float/int hashing.
"""

import numpy as np
import pytest

import universal_dtypes as ud

ALL = list(ud.dtypes.items())
HAS_NAN = [(n, T) for n, T in ALL if np.isnan(float(np.array([np.nan], dtype=T)[0]))]


@pytest.mark.parametrize("name,T", ALL)
def test_scalar_is_hashable(name, T):
    assert isinstance(hash(T(0.5)), int)


@pytest.mark.parametrize("name,T", ALL)
def test_hash_consistent_with_eq(name, T):
    a, b = T(0.5), T(0.5)
    assert a == b
    assert hash(a) == hash(b)


@pytest.mark.parametrize("name,T", ALL)
def test_hash_matches_python_float_and_int(name, T):
    # 0.5 is exactly representable in every format; hashes like the float.
    assert hash(T(0.5)) == hash(0.5)
    # an integral value hashes like the int, too (Python: hash(0.0) == hash(0)).
    assert hash(T(0.0)) == hash(0.0) == hash(0)


@pytest.mark.parametrize("name,T", ALL)
def test_usable_as_set_and_dict_keys(name, T):
    s = {T(0.5), T(0.5), T(0.25)}
    assert len(s) == 2  # equal values collapse
    d = {T(0.5): "a"}
    d[T(0.5)] = "b"
    assert len(d) == 1 and d[T(0.5)] == "b"


@pytest.mark.parametrize("name,T", HAS_NAN)
def test_nan_is_hashable(name, T):
    assert isinstance(hash(T(float("nan"))), int)  # must not raise
