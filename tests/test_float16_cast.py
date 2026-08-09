"""float16 casts — issue #43.

Before this, `build_casts` wired float32/float64/int/bool but no float16, so the
ML-ecosystem-common half type had no direct cast. These verify to/from float16 in
both directions, via the value domain (npy_half helpers, since npy_half is a bit
pattern, not a C float type).
"""

import numpy as np
import pytest

import universal_dtypes as ud

ALL = list(ud.dtypes.items())
# powers of two -> exact in every format (so the assertions are bit-exact)
GRID = [0.5, 0.25, -0.75, 0.0, 0.125, -0.375]


@pytest.mark.parametrize("name,T", ALL)
def test_cast_to_float16(name, T):
    a = np.array(GRID, dtype=T)
    h = a.astype(np.float16)
    assert h.dtype == np.float16
    # value-domain: same as going through float64 first
    ref = a.astype(np.float64).astype(np.float16)
    assert h.tobytes() == ref.tobytes()


@pytest.mark.parametrize("name,T", ALL)
def test_cast_from_float16(name, T):
    h = np.array(GRID, dtype=np.float16)
    got = h.astype(T)
    assert got.dtype == np.dtype(T)
    ref = h.astype(np.float64).astype(T)
    assert got.tobytes() == ref.tobytes()


@pytest.mark.parametrize("name,T", ALL)
def test_float16_can_cast_unsafe(name, T):
    assert np.can_cast(np.float16, np.dtype(T), "unsafe")
    assert np.can_cast(np.dtype(T), np.float16, "unsafe")


@pytest.mark.parametrize("name,T", ALL)
def test_float16_roundtrip_where_representable(name, T):
    # pure powers of two are exact in every format (incl. LNS, which only
    # represents powers of two exactly), so they round-trip through float16.
    h = np.array([0.5, -0.25, 0.125], dtype=np.float16)
    assert np.array_equal(h.astype(T).astype(np.float16), h)
