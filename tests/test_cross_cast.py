"""Cross-dtype casts: value-domain casts between every pair of universal dtypes.

Before #39 NumPy could not cast between two custom dtypes at all (it does not
chain custom->custom through a builtin). This exercises the shared cross-cast
machinery:

  * every ordered pair casts (and ``np.can_cast(..., "unsafe")`` agrees);
  * the conversion is value-preserving to each type's own precision, verified by
    a bit-exact round-trip through ``qd_cascade`` (which is wide enough to hold
    every registered type);
  * for the wide configs (``posit64`` and the cascades) the compensated
    expansion beats a plain ``double`` intermediate, which is the whole point;
  * a narrow source agrees bit-for-bit with the trusted ``->float64->`` path;
  * NaN / inf propagate.
"""

import numpy as np
import pytest

import universal_dtypes as ud

ALL = list(ud.dtypes.items())  # (name, scalar type)

# Types carrying more significand than double's 53 bits. Determined empirically
# by a value-grid probe (a value built via the type's own arithmetic that a
# float64 round-trip loses); see the PR for #39. `lns32` as currently configured
# fits in double, so it is NOT here — but the machinery covers it identically.
WIDE = {"posit64", "dd_cascade", "td_cascade", "qd_cascade"}

# Small dyadic values inside the range of every format (the fractional DSP
# formats saturate at |x| ~ 1), so rounding into any type is well defined.
GRID = [0.0, 0.5, -0.5, 0.25, -0.25, 0.75, -0.75, 0.125, -0.375, 0.9, -0.6]


def _arr(values, T):
    return np.array(values, dtype=T)


@pytest.mark.parametrize("dst_name,dst", ALL)
@pytest.mark.parametrize("src_name,src", ALL)
def test_every_ordered_pair_casts(src_name, src, dst_name, dst):
    """astype works for every ordered pair, with the right output dtype/shape."""
    a = _arr(GRID, src)
    b = a.astype(dst)
    assert b.dtype == np.dtype(dst)
    assert b.shape == a.shape
    if src_name != dst_name:
        assert np.can_cast(np.dtype(src), np.dtype(dst), "unsafe")


@pytest.mark.parametrize("name,T", ALL)
def test_roundtrip_through_qd_is_bit_exact(name, T):
    """T -> qd_cascade -> T recovers every element bit-for-bit.

    qd_cascade holds more significand than any registered type, so this is the
    universal preservation invariant: the cross-cast never corrupts a value the
    destination can already represent.
    """
    a = _arr(GRID, T)
    rt = a.astype(ud.qd_cascade).astype(T)
    assert a.tobytes() == rt.tobytes(), f"{name} lost bits through qd round-trip"


@pytest.mark.parametrize("name", sorted(WIDE))
def test_wide_types_need_the_compensated_path(name):
    """For a >53-bit value, the qd expansion round-trips exactly; double does not.

    Builds the value with the type's OWN arithmetic (a double *literal* could not
    even represent it), then shows the compensated intermediate is necessary.
    """
    T = ud.dtypes[name]
    one = _arr([1.0], T)
    wide = one + _arr([2.0**-54], T)  # in-type arithmetic -> below double's ULP@1
    assert float(wide.astype(np.float64)[0]) == 1.0  # double literally can't see it
    via_qd = wide.astype(ud.qd_cascade).astype(T)
    via_double = wide.astype(np.float64).astype(T)
    assert wide.tobytes() == via_qd.tobytes()  # compensated path: exact
    assert wide.tobytes() != via_double.tobytes()  # double path: lossy


@pytest.mark.parametrize("dst_name,dst", ALL)
@pytest.mark.parametrize("src_name,src", [(n, t) for n, t in ALL if n not in WIDE])
def test_narrow_source_matches_double_path(src_name, src, dst_name, dst):
    """A narrow source (<=53 bits) fits exactly in float64, so the cross-cast
    must agree bit-for-bit with the trusted src->float64->dst route."""
    a = _arr(GRID, src)
    direct = a.astype(dst)
    via_double = a.astype(np.float64).astype(dst)
    assert direct.tobytes() == via_double.tobytes()


@pytest.mark.parametrize("dst_name,dst", ALL)
@pytest.mark.parametrize("src_name,src", ALL)
def test_nan_propagates(src_name, src, dst_name, dst):
    """A NaN/NaR in the source lands as the destination's NaN/NaR."""
    a = _arr([0.5], src)
    a_nan = _arr([float("nan")], src)
    # only meaningful if the source can represent NaN (posit NaR, IEEE NaN, ...)
    if not np.isnan(a_nan.astype(np.float64)[0]):
        pytest.skip(f"{src_name} has no NaN representation")
    with np.errstate(invalid="ignore"):  # NaN cast legitimately trips this warning
        b = a_nan.astype(dst)
    # destination may or may not have NaN; if it does, it must be NaN
    if np.isnan(_arr([float("nan")], dst).astype(np.float64)[0]):
        assert np.isnan(b.astype(np.float64)[0])
    # a finite value next to it stays finite (no cross-contamination)
    assert np.isfinite(a.astype(dst).astype(np.float64)[0])


@pytest.mark.parametrize("dst_name,dst", ALL)
def test_cross_family_values_are_close(dst_name, dst):
    """Sanity on the actual numbers: casting posit16 values into any dtype lands
    within that dtype's coarse resolution of the original real values."""
    src_vals = _arr(GRID, ud.posit16)
    ref = src_vals.astype(np.float64)
    got = src_vals.astype(dst).astype(np.float64)
    # generous: destinations include 8-bit formats. This checks we transferred
    # the value, not a garbled reinterpretation.
    assert np.allclose(got, ref, atol=0.15), f"posit16 -> {dst_name}: {got} vs {ref}"
