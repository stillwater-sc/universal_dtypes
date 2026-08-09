"""Persistence & byte-order contract — issue #49.

Pins the frozen v2.0.0 behavior: arrays round-trip via pickle / np.save / raw
bytes **on the same platform**, and the dtypes are **native-endian only** (no
byte-order tag; NumPy 2.x new-style DTypes support neither `newbyteorder` nor
`byteswap`). See docs/dtypes.md "Persistence and byte order".
"""

import io
import pickle
import warnings

import numpy as np
import pytest

import universal_dtypes as ud

ALL = list(ud.dtypes.items())
GRID = [0.5, -0.25, 1.5, 0.125, -0.75]


def _arr(v, T):
    return np.array(v, dtype=T)


# ---- supported round-trips (same platform) ----------------------------------


@pytest.mark.parametrize("name,T", ALL)
def test_pickle_array_roundtrip(name, T):
    a = _arr(GRID, T)
    assert pickle.loads(pickle.dumps(a)).tobytes() == a.tobytes()


@pytest.mark.parametrize("name,T", ALL)
def test_npy_save_load_roundtrip(name, T):
    a = _arr(GRID, T)
    buf = io.BytesIO()
    with warnings.catch_warnings():  # numpy warns that custom dtypes save via pickle
        warnings.simplefilter("ignore")
        np.save(buf, a)
        buf.seek(0)
        b = np.load(buf, allow_pickle=True)
    assert b.tobytes() == a.tobytes()
    assert type(b[0]) is T


@pytest.mark.parametrize("name,T", ALL)
def test_raw_bytes_roundtrip(name, T):
    a = _arr(GRID, T)
    b = np.frombuffer(a.tobytes(), dtype=T)
    assert b.tobytes() == a.tobytes()
    assert np.array_equal(a.astype(np.float64), b.astype(np.float64))


@pytest.mark.parametrize("name,T", ALL)
def test_scalar_pickle_by_value(name, T):
    # a single scalar pickles by value (portable), unlike the raw-bytes array
    x = T(0.5)
    assert float(pickle.loads(pickle.dumps(x))) == float(x)


def test_multiword_cascade_roundtrips():
    a = _arr([1.0, 2.5, -3.25], ud.dd_cascade)
    assert pickle.loads(pickle.dumps(a)).tobytes() == a.tobytes()
    assert np.frombuffer(a.tobytes(), dtype=ud.dd_cascade).tobytes() == a.tobytes()


# ---- byte order: native-only contract ---------------------------------------


@pytest.mark.parametrize("name,T", ALL)
def test_dtype_has_no_byteorder(name, T):
    # '|' == "not applicable": these dtypes carry no endianness tag
    assert np.dtype(T).byteorder == "|"


@pytest.mark.parametrize("name,T", ALL)
def test_byteswap_and_newbyteorder_unsupported(name, T):
    a = _arr(GRID, T)
    with pytest.raises(TypeError):
        a.byteswap()
    with pytest.raises(TypeError):
        np.dtype(T).newbyteorder(">")
