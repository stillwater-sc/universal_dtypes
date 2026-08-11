"""takum<nbits,3> NumPy dtype family — tests (issue #63).

The LINEAR takum (Hunhold, 2024, arXiv:2404.18603), registered through the same
NEP-42 harness as the posit family. Universal's logarithmic variant is a separate
upstream type (``takum_log``), so the bare ``takum{nbits}`` names here stay the
linear encoding.

The property that earns takum its place next to posit is a dynamic range that is
essentially independent of width: extra bits buy precision rather than range,
where posit's range grows by tens of orders of magnitude per doubling. That
contrast is asserted directly below.
"""

import pickle
import warnings

import numpy as np
import pytest

import universal_dtypes as ud

TAKUMS = [
    ("takum8", ud.takum8, 1),
    ("takum16", ud.takum16, 2),
    ("takum32", ud.takum32, 4),
    ("takum64", ud.takum64, 8),
]

# Exactly representable in every width (powers of two and their simple sums).
EXACT = [1.0, 2.0, 0.5, -4.0, 0.25]


def maxpos(T):
    """Largest finite magnitude, without tripping the saturation report (#60)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.array([1e300], dtype=T)[0])


# ---- registration -----------------------------------------------------------


@pytest.mark.parametrize("name,scalar,itemsize", TAKUMS)
def test_dtype_resolves(name, scalar, itemsize):
    dt = np.dtype(scalar)
    assert dt.itemsize == itemsize
    assert np.dtype(name) == dt  # string-name resolution


def test_registry():
    assert set(ud.takum_dtypes) == {"takum8", "takum16", "takum32", "takum64"}
    assert set(ud.takum_dtypes) <= set(ud.dtypes)
    for name, scalar, _ in TAKUMS:
        assert ud.takum_dtypes[name] is scalar


# ---- the defining property: width-independent range -------------------------


def test_range_is_essentially_width_independent():
    """takum's headline property. From 16 bits up the maximum is flat (~5.8e76);
    the 8-bit config is close behind. Contrast test below."""
    mx = {n: maxpos(T) for n, T, _ in TAKUMS}
    assert mx["takum16"] == pytest.approx(mx["takum32"], rel=0.05)
    # 32 vs 64 agree to ~6 digits; they are not identical because maxpos depends
    # on the mantissa width, which does still grow.
    assert mx["takum32"] == pytest.approx(mx["takum64"], rel=1e-5)
    assert mx["takum8"] > 1e70  # even 1 byte reaches ~8.8e71
    for v in mx.values():
        assert v > 1e70


def test_range_contrasts_with_posit():
    """posit spends bits on range, takum on precision. At 8 bits takum's maximum
    is ~64 orders of magnitude larger than posit's; by 64 bits they converge."""
    assert maxpos(ud.takum8) / maxpos(ud.posit8) > 1e60
    assert maxpos(ud.takum16) / maxpos(ud.posit16) > 1e55
    # posit's range grows steeply with width; takum's does not.
    assert maxpos(ud.posit64) / maxpos(ud.posit8) > 1e60
    assert maxpos(ud.takum64) / maxpos(ud.takum8) < 1e10


def test_precision_improves_with_width():
    """The other half of the trade: fixed range, so wider means more accurate."""
    err = [abs(float(np.array([1 / 3], dtype=T)[0]) - 1 / 3) for _, T, _ in TAKUMS]
    assert err[0] > err[1] > err[2] > err[3]
    assert err[3] < 1e-15  # takum64 nails 1/3 to double precision or better


# ---- value semantics --------------------------------------------------------


@pytest.mark.parametrize("name,scalar,itemsize", TAKUMS)
def test_array_creation_and_roundtrip(name, scalar, itemsize):
    a = np.array(EXACT, dtype=scalar)
    assert a.dtype == np.dtype(scalar)
    np.testing.assert_array_equal(a.astype(np.float64), EXACT)


@pytest.mark.parametrize("name,scalar,itemsize", TAKUMS)
def test_exact_arithmetic(name, scalar, itemsize):
    a = np.array([1.0, 2.0, 0.5], dtype=scalar)
    b = np.array([2.0, 0.5, 0.5], dtype=scalar)
    np.testing.assert_array_equal((a + b).astype(np.float64), [3.0, 2.5, 1.0])
    np.testing.assert_array_equal((a * b).astype(np.float64), [2.0, 1.0, 0.25])
    np.testing.assert_array_equal((a - b).astype(np.float64), [-1.0, 1.5, 0.0])
    np.testing.assert_array_equal((a / b).astype(np.float64), [0.5, 4.0, 1.0])


@pytest.mark.parametrize("name,scalar,itemsize", TAKUMS)
def test_comparisons_and_sort(name, scalar, itemsize):
    a = np.array([2.0, 0.5, -4.0, 1.0], dtype=scalar)
    np.testing.assert_array_equal(np.sort(a).astype(np.float64), [-4.0, 0.5, 1.0, 2.0])
    np.testing.assert_array_equal(np.argsort(a), [2, 1, 3, 0])
    np.testing.assert_array_equal(a > np.array(1.0, dtype=scalar), [True, False, False, False])


@pytest.mark.parametrize("name,scalar,itemsize", TAKUMS)
def test_reductions(name, scalar, itemsize):
    a = np.array([1.0, 2.0, 0.5], dtype=scalar)
    assert float(np.sum(a)) == 3.5
    assert float(np.prod(a)) == 1.0
    assert float(np.min(a)) == 0.5
    assert float(np.max(a)) == 2.0


# ---- NaR: one exceptional value, no infinity --------------------------------


@pytest.mark.parametrize("name,scalar,itemsize", TAKUMS)
def test_nar_semantics(name, scalar, itemsize):
    """Like posit: a single NaR maps onto isnan, and there is no infinity, so any
    non-finite input converts to NaR."""
    nar = np.array([np.nan], dtype=scalar)
    assert np.isnan(nar)[0]
    assert not np.isinf(nar)[0]
    for non_finite in (np.inf, -np.inf):
        v = np.array([non_finite], dtype=scalar)
        assert np.isnan(v)[0], "non-finite float must become NaR"
        assert not np.isinf(v)[0], "takum has no infinity"
    assert not np.isfinite(nar)[0]
    # NaR propagates through arithmetic
    assert np.isnan(nar * np.array(2.0, dtype=scalar))[0]


@pytest.mark.parametrize("name,scalar,itemsize", TAKUMS)
def test_encoding_landmarks(name, scalar, itemsize):
    """Two's complement storage: zero is all-zero bits, NaR is the sign bit only
    (0x80.. in the most significant byte)."""
    zero = np.array([0.0], dtype=scalar).tobytes()
    assert zero == b"\x00" * itemsize
    nar = np.array([np.nan], dtype=scalar).tobytes()
    expected = b"\x00" * (itemsize - 1) + b"\x80"  # little-endian
    assert nar == expected


# ---- takum64 exceeds double, so it must not route through it ----------------


def test_takum64_orders_at_full_precision():
    """takum64's significand exceeds double's 53 bits. Two adjacent encodings
    collapse to the same float64, so comparison must use takum's own operators —
    a to_double shortcut would call them equal."""
    T = ud.takum64
    a = np.array([1.0], dtype=T)
    raw = int.from_bytes(a.tobytes(), "little")
    b = np.frombuffer((raw + 1).to_bytes(8, "little"), dtype=T)
    assert float(a[0]) == float(b[0]), "precondition: indistinguishable as float64"
    assert (a < b)[0], "comparison must see past double"
    assert not (a == b)[0]
    assert np.argsort(np.concatenate([b, a])).tolist() == [1, 0]


def test_takum64_cast_out_is_unsafe():
    """Because the out-cast to float64 loses low bits, it is graded unsafe — the
    same rule the cascades follow. The narrower configs stay safe."""
    assert not np.can_cast(np.dtype(ud.takum64), np.float64, "safe")
    for T in (ud.takum8, ud.takum16, ud.takum32):
        assert np.can_cast(np.dtype(T), np.float64, "safe")


def _one_ulp_above_one(T, itemsize):
    a = np.array([1.0], dtype=T)
    raw = int.from_bytes(a.tobytes(), "little")
    return np.frombuffer((raw + 1).to_bytes(itemsize, "little"), dtype=T)


def test_takum64_arithmetic_is_limited_to_double():
    """PINNED LIMITATION, not a preference.

    Universal implements takum's +, -, *, / by converting both operands to
    `double`, computing there, and converting back (takum_impl.hpp). For
    takum8/16/32 that is exact — their significands are well under double's 53
    bits, so it is a single correct rounding. For takum64, whose significand
    reaches ~59 bits, the operands are rounded to double *first*, so any detail
    below double precision is lost before the operation runs.

    The encoding, comparisons, sort and storage all still carry the full width
    (see test_takum64_orders_at_full_precision) — it is specifically arithmetic
    that is capped. posit64 has native arithmetic and does not share this, which
    is the contrast asserted below.
    """
    T = ud.takum64
    tricky = _one_ulp_above_one(T, 8)
    zero = np.array([0.0], dtype=T)
    assert tricky.tobytes() != np.array([1.0], dtype=T).tobytes()  # precondition
    assert (tricky + zero).tobytes() != tricky.tobytes(), "adding zero should be a no-op"
    assert (tricky * np.array([1.0], dtype=T)).tobytes() != tricky.tobytes()

    # posit64, by contrast, has native arithmetic at full width.
    P = ud.posit64
    p_tricky = _one_ulp_above_one(P, 8)
    assert (p_tricky + np.array([0.0], dtype=P)).tobytes() == p_tricky.tobytes()


def test_takum64_cross_cast_exact_within_double():
    """The cross-cast expansion builds its residual terms with the type's own
    arithmetic, so it inherits the cap above: takum64 round-trips exactly for
    anything double can hold, but not for detail below that.

    takum8/16/32 round-trip exactly for everything, since double covers them.
    """
    for _, T, size in TAKUMS:
        a = np.array([1.0, 0.5, -4.0, 1 / 3, 1e30], dtype=T)
        assert a.astype(ud.qd_cascade).astype(T).tobytes() == a.tobytes()
    # the sub-double bit is where takum64 cannot round-trip
    T = ud.takum64
    tricky = _one_ulp_above_one(T, 8)
    assert tricky.astype(ud.qd_cascade).astype(T).tobytes() != tricky.tobytes()


# ---- integration with the rest of the package -------------------------------


@pytest.mark.parametrize("name,scalar,itemsize", TAKUMS)
def test_cross_casts_and_builtin_casts(name, scalar, itemsize):
    a = np.array([1.0, 0.5], dtype=scalar)
    for target in (ud.posit32, ud.bfloat16, ud.dd_cascade, ud.fp16):
        assert a.astype(target).astype(np.float64).tolist() == [1.0, 0.5]
    assert a.astype(np.float32).tolist() == [1.0, 0.5]
    assert np.array([1.0, 0.5]).astype(scalar).astype(np.float64).tolist() == [1.0, 0.5]


@pytest.mark.parametrize("name,scalar,itemsize", TAKUMS)
def test_scalar_promotion_and_arange(name, scalar, itemsize):
    """The features added in #55 and #56 must cover takum too — they live in the
    shared harness, so this is a regression guard on that being true."""
    a = np.array([1.0, 2.0], dtype=scalar)
    assert (a * 2).dtype == np.dtype(scalar)
    np.testing.assert_array_equal((a * 2).astype(np.float64), [2.0, 4.0])
    np.testing.assert_array_equal(a > 1, [False, True])
    assert np.arange(4, dtype=scalar).astype(np.float64).tolist() == [0.0, 1.0, 2.0, 3.0]


@pytest.mark.parametrize("name,scalar,itemsize", TAKUMS)
def test_pickle_and_hash(name, scalar, itemsize):
    a = np.array(EXACT, dtype=scalar)
    assert pickle.loads(pickle.dumps(a)).tobytes() == a.tobytes()
    assert hash(scalar(1.0)) == hash(1.0)
