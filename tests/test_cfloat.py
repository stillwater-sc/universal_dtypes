"""cfloat NumPy dtype family — tests (issue #8).

The cfloat configs are registered through the reusable NEP-42 harness.

IEEE AND ml_dtypes: from 3.0.0 (Universal v5.18.0), `fp16` is bit-identical to
`numpy.float16` and `fp8e5m2` to `ml_dtypes.float8_e5m2`, `±inf` and `NaN`
included. Universal v5.15.0 made cfloat's inf the encoding right after maxpos,
with every NaN above it (universal#1895, #1897), which is IEEE-754's rule for
configs without max-exponent values. Before that, `±inf` was `s.11..11.11..10`,
and the four `±inf` patterns meant something else (#57). The exhaustive tests
below pin the new encodings.
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


def test_fp16_rounding_matches_ieee_half_on_finite_values():
    # Same layout + round-to-nearest-even, so the finite range must agree.
    grid = np.linspace(-70000, 70000, 5001, dtype=np.float64)
    ours = np.array(grid, dtype=ud.fp16).astype(np.float64)
    with np.errstate(over="ignore"):
        theirs = grid.astype(np.float16).astype(np.float64)
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.skipif(not HAVE_ML, reason="ml_dtypes not installed")
def test_fp8e5m2_rounding_matches_reference_on_finite_values():
    grid = np.linspace(-70000, 70000, 6001, dtype=np.float64)
    ours = np.array(grid, dtype=ud.fp8e5m2).astype(np.float64)
    with np.errstate(over="ignore"):
        theirs = grid.astype(ml_dtypes.float8_e5m2).astype(np.float64)
    np.testing.assert_array_equal(ours, theirs)


# ---- signaling-NaN conversion --------------------------------------------------
#
# Universal v4.7.9's cfloat converted a NaN whose quiet bit is clear into
# infinity when narrowing from a wider IEEE type: the quiet bit was used as the
# discriminator instead of "exponent all ones and mantissa nonzero". NaN silently
# becoming +-inf turns a value that would have propagated and been detectable
# into a plausible one.
#
# Fixed upstream (stillwater-sc/universal#1303) and picked up by the v5.x pin.
# These ran as strict xfails until then; they now assert the required semantics.


def _ieee_half_nan_encodings():
    """Every IEEE binary16 NaN encoding, as a float16 array.

    This is the source to use, not a signaling NaN built in float64. A
    constructed sNaN is not a portable way to reach the conversion: MSVC quiets
    it before it gets there, so the same test XPASSed on Windows while failing on
    Linux and macOS. These go through cast_from_half -> npy_half_to_double, which
    is bit manipulation, so the payload arrives intact on every platform.
    """
    half = np.frombuffer(np.arange(1 << 16, dtype=np.uint16).tobytes(), dtype=np.float16)
    return half[np.isnan(half)]


@pytest.mark.parametrize("name,scalar,itemsize", CFLOATS)
def test_every_ieee_half_nan_encoding_stays_nan(name, scalar, itemsize):
    """Every IEEE binary16 NaN encoding must land on a NaN, whatever its payload.

    Under Universal v4.7.9, 2040 of the 2046 became infinity — the ones whose
    quiet bit is clear.
    """
    nan_in = _ieee_half_nan_encodings()
    with np.errstate(all="ignore"):  # a signaling NaN raises IEEE invalid by definition
        converted = nan_in.astype(scalar)
    lost = int((~np.isnan(converted)).sum())
    assert lost == 0, f"{lost} of {len(nan_in)} IEEE half NaN encodings did not stay NaN"


def test_ieee_half_nan_encodings_stay_nan_for_non_cfloat_dtypes():
    """The contrast that scopes the bug: posit, bfloat16 and takum carry all 2046
    encodings across correctly, so this is a cfloat conversion problem rather
    than something in the shared dtype harness. Guards against a future 'fix'
    that breaks these instead."""
    nan_in = _ieee_half_nan_encodings()
    for name in ("posit16", "bfloat16", "takum16"):
        with np.errstate(all="ignore"):
            converted = nan_in.astype(ud.dtypes[name])
        lost = int((~np.isnan(converted)).sum())
        assert lost == 0, f"{name} lost NaN on {lost} of {len(nan_in)} encodings"


def test_quiet_nan_survives_cast():
    """The quiet-NaN path works today and must keep working."""
    for _, scalar, _ in CFLOATS:
        with np.errstate(all="ignore"):
            got = np.array([np.nan], dtype=np.float64).astype(scalar)
        assert np.isnan(got)[0]
        assert not np.isinf(got)[0]


def test_cfloat_registry():
    assert set(ud.cfloat_dtypes) == {name for name, _, _ in CFLOATS}
    for name, scalar, _ in CFLOATS:
        assert ud.cfloat_dtypes[name] is scalar
    # rolled into the top-level registry too
    assert set(ud.cfloat_dtypes) <= set(ud.dtypes)


# ---- IEEE-754 encodings (3.0.0, Universal v5.15.0+) -------------------------------
#
# Each of these fails against 2.x, whose cfloat encoded +-inf as s.11..11.11..10:
# there 0x7c00 decoded as NaN and float('inf') encoded as 0x7ffe.


def _same_values(ours, theirs):
    """Equal as values, with NaN equal to NaN and the sign of every non-NaN kept."""
    both_nan = np.isnan(ours) & np.isnan(theirs)
    same = (ours == theirs) & (np.signbit(ours) == np.signbit(theirs))
    return both_nan | same


def test_fp16_decodes_every_pattern_as_ieee_half():
    bits = np.arange(1 << 16, dtype=np.uint64).astype(np.uint16)
    ours = bits.view(ud.fp16).astype(np.float64)
    theirs = bits.view(np.float16).astype(np.float64)
    bad = np.nonzero(~_same_values(ours, theirs))[0]
    assert bad.size == 0, f"{bad.size} patterns differ, first {hex(int(bits[bad[0]]))}"


def test_fp16_encodes_as_ieee_half():
    rng = np.random.default_rng(57)
    grid = np.concatenate(
        [
            [0.0, -0.0, np.inf, -np.inf, np.nan, 65504.0, 65520.0, -65520.0, 1e30],
            [2.0**-24, 2.0**-25],
            rng.standard_normal(20000) * np.exp(rng.uniform(-20, 12, 20000)),
        ]
    )
    with np.errstate(over="ignore"):
        theirs = grid.astype(np.float16).view(np.uint16)
    ours = grid.astype(ud.fp16).view(np.uint16)
    bad = np.nonzero(ours != theirs)[0]
    assert bad.size == 0, f"{bad.size} encodings differ, first x={grid[bad[0]]!r}"


def test_fp16_buffer_reinterprets_as_float16():
    """Zero-copy aliasing, which 2.x documented as unsupported, now round-trips."""
    a = np.array([1.5, -np.inf, np.inf, np.nan, -0.0, 65504.0], dtype=ud.fp16)
    h = np.frombuffer(a.tobytes(), dtype=np.float16)
    assert _same_values(h.astype(np.float64), a.astype(np.float64)).all()
    back = np.frombuffer(h.tobytes(), dtype=ud.fp16)
    assert _same_values(back.astype(np.float64), a.astype(np.float64)).all()


def test_fp8e5m2_special_encodings_are_ieee():
    bits = np.array([0x7C, 0xFC, 0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF], dtype=np.uint8)
    got = bits.view(ud.fp8e5m2).astype(np.float64)
    assert got[0] == np.inf and got[1] == -np.inf
    assert np.isnan(got[2:]).all()
    enc = np.array([np.inf, -np.inf], dtype=np.float64).astype(ud.fp8e5m2).view(np.uint8)
    assert list(enc) == [0x7C, 0xFC]


@pytest.mark.skipif(not HAVE_ML, reason="ml_dtypes not installed")
def test_fp8e5m2_decodes_every_pattern_as_ml_dtypes():
    bits = np.arange(256, dtype=np.uint8)
    ours = bits.view(ud.fp8e5m2).astype(np.float64)
    theirs = bits.view(ml_dtypes.float8_e5m2).astype(np.float64)
    bad = np.nonzero(~_same_values(ours, theirs))[0]
    assert bad.size == 0, f"{bad.size} patterns differ, first {hex(int(bits[bad[0]]))}"
