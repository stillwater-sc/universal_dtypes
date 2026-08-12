"""cfloat NumPy dtype family — tests (issue #8).

The cfloat configs are registered through the reusable NEP-42 harness.

NOTE ON THE IEEE / ml_dtypes COMPARISONS BELOW (#57): they are *rounding
correctness* checks, not compatibility guarantees. `cfloat` applies one encoding
rule across its whole parameter range and is the definition this package
implements; `fp16` is deliberately **not** a drop-in for `numpy.float16`, nor
`fp8e5m2` for `ml_dtypes.float8_e5m2`, and zero-copy aliasing across them is
unsupported. Same field layout and round-to-nearest-even means the finite range
must agree, which makes those types a convenient oracle for catching a rounding
bug — and nothing more. The `±inf` encodings differ by design.
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
    # Same layout + round-to-nearest-even, so the finite range must agree. This
    # catches a rounding bug; it is not a compatibility guarantee (see #57).
    grid = np.linspace(-70000, 70000, 5001, dtype=np.float64)
    ours = np.array(grid, dtype=ud.fp16).astype(np.float64)
    with np.errstate(over="ignore"):
        theirs = grid.astype(np.float16).astype(np.float64)
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.skipif(not HAVE_ML, reason="ml_dtypes not installed")
def test_fp8e5m2_rounding_matches_reference_on_finite_values():
    # As above: an oracle for rounding, not a supported interchange format.
    grid = np.linspace(-70000, 70000, 6001, dtype=np.float64)
    ours = np.array(grid, dtype=ud.fp8e5m2).astype(np.float64)
    with np.errstate(over="ignore"):
        theirs = grid.astype(ml_dtypes.float8_e5m2).astype(np.float64)
    np.testing.assert_array_equal(ours, theirs)


# ---- signaling-NaN conversion (upstream bug tripwire) -----------------------
#
# Universal's cfloat converts a *signaling* NaN to infinity when narrowing from a
# wider IEEE type — the quiet bit is used as the discriminator instead of
# "exponent all ones and mantissa nonzero". NaN silently becoming ±inf turns a
# value that would have propagated and been detectable into a plausible one.
#
# Filed as stillwater-sc/universal#1303. There is no fix available here: it lands
# when the pinned Universal (CMakeLists.txt, issue #66) moves past it.
#
# These are marked strict xfail rather than asserting today's wrong behavior, so
# they document the *required* semantics. When the pin moves and the bug is
# fixed, strict xfail turns the unexpected pass into a failure — which is the
# signal to delete the markers, not a regression.


def _signaling_nan64():
    """A float64 NaN with the quiet bit clear."""
    raw = np.array([0x7FF0000000000001], dtype=np.uint64)
    return np.frombuffer(raw.tobytes(), dtype=np.float64)


@pytest.mark.xfail(strict=True, reason="stillwater-sc/universal#1303: cfloat sNaN -> inf")
@pytest.mark.parametrize("name,scalar,itemsize", CFLOATS)
def test_signaling_nan_survives_cast(name, scalar, itemsize):
    """A NaN must stay a NaN across the cast, whatever its payload."""
    with np.errstate(all="ignore"):  # sNaN raises IEEE invalid by definition
        got = _signaling_nan64().astype(scalar)
    assert np.isnan(got)[0], "signaling NaN did not survive the cast"
    assert not np.isinf(got)[0], "signaling NaN became infinity"


@pytest.mark.xfail(strict=True, reason="stillwater-sc/universal#1303: cfloat sNaN -> inf")
def test_every_ieee_half_nan_encoding_stays_nan_in_fp16():
    """Sweep: every IEEE binary16 NaN encoding must land on a NaN in fp16.

    2040 of the 2046 NaN encodings currently become infinity.
    """
    half = np.frombuffer(np.arange(1 << 16, dtype=np.uint16).tobytes(), dtype=np.float16)
    is_nan_in = np.isnan(half)
    with np.errstate(all="ignore"):
        converted = half.astype(ud.fp16)
    lost = int((is_nan_in & ~np.isnan(converted)).sum())
    assert lost == 0, f"{lost} IEEE half NaN encodings did not stay NaN"


def test_signaling_nan_survives_cast_for_non_cfloat_dtypes():
    """The contrast that scopes the bug: posit, bfloat16 and takum handle the
    same input correctly, so this is a cfloat conversion problem rather than
    something in the shared dtype harness. Guards against a future 'fix' that
    breaks these instead."""
    snan = _signaling_nan64()
    for name in ("posit16", "bfloat16", "takum16"):
        with np.errstate(all="ignore"):
            got = snan.astype(ud.dtypes[name])
        assert np.isnan(got)[0], f"{name} lost a signaling NaN"
        assert not np.isinf(got)[0], f"{name} turned a signaling NaN into infinity"


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
