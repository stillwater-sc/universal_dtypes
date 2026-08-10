"""np.arange support — issue #56.

`np.arange(n, dtype=<universal dtype>)` raised "arange() not supported for inputs
with DType", while zeros/ones/full/empty all worked. NumPy builds an arithmetic
progression through the DType's `fill` ArrFuncs slot, which the harness did not
implement; these pin the slot and, more importantly, its value semantics.
"""

import numpy as np
import pytest

import universal_dtypes as ud

try:
    import ml_dtypes  # noqa: F401

    HAVE_ML = True
except ImportError:
    HAVE_ML = False

ALL = list(ud.dtypes.items())

# Steps that are exact in binary, so every format reproduces them faithfully.
EXACT_CASES = [(4,), (0, 1, 0.25), (0, 0.5, 0.125), (-0.5, 0.5, 0.25), (0, 3, 1)]


@pytest.mark.parametrize("name,T", ALL)
def test_arange_basic(name, T):
    a = np.arange(4, dtype=T)
    assert a.dtype == np.dtype(T)
    assert len(a) == 4


@pytest.mark.parametrize("name,T", ALL)
def test_arange_by_string_name(name, T):
    if name == "bfloat16" and HAVE_ML:
        pytest.skip("ml_dtypes also owns the 'bfloat16' name")
    assert np.arange(3, dtype=name).dtype == np.dtype(T)


@pytest.mark.parametrize("name,T", ALL)
@pytest.mark.parametrize("args", EXACT_CASES)
def test_arange_matches_float64_reference_for_exact_steps(name, T, args):
    """With a binary-exact step there is no rounding to disagree about, so arange
    must equal the float64 progression cast into the type."""
    got = np.arange(*args, dtype=T)
    ref = np.arange(*args, dtype=np.float64).astype(T)
    assert got.tobytes() == ref.tobytes()


@pytest.mark.parametrize("name,T", ALL)
def test_arange_uses_index_form_not_accumulation(name, T):
    """Each element is start + i*delta computed from the absolute index and
    rounded once — NOT v[i-1] + delta, which compounds error at every step.

    delta comes from the *rounded* first two elements, which is exactly what
    NumPy's own float fill does (see test_matches_numpy_float16_semantics)."""
    got = np.arange(0, 1, 0.1, dtype=T)
    if len(got) < 3:
        pytest.skip("format saturates too early for this to be meaningful")
    delta = float(np.array(0.1, dtype=T))
    index_form = np.array([0.0 + i * delta for i in range(len(got))], dtype=T)
    assert got.tobytes() == index_form.tobytes()

    # And show accumulation would genuinely have differed, so this test bites.
    acc, cur = [], np.array(0.0, dtype=T)
    step = np.array(delta, dtype=T)
    for _ in range(len(got)):
        acc.append(float(cur))
        cur = cur + step
    if not np.array_equal(acc, [float(x) for x in index_form]):
        assert got.tobytes() != np.array(acc, dtype=T).tobytes()


def test_matches_numpy_float16_semantics():
    """The reference for the rule: NumPy's own float16 derives delta from the
    rounded first two elements, so its arange also differs from
    arange(float64).astype(float16). Ours behaves identically."""
    ref = np.arange(0, 1, 0.1, dtype=np.float16)
    ref_delta = float(np.float16(0.1))
    ref_index = np.array([0.0 + i * ref_delta for i in range(len(ref))], dtype=np.float16)
    assert np.array_equal(ref, ref_index)
    assert not np.array_equal(ref, np.arange(0, 1, 0.1, dtype=np.float64).astype(np.float16))

    got = np.arange(0, 1, 0.1, dtype=ud.posit16)
    delta = float(np.array(0.1, dtype=ud.posit16))
    assert (
        got.tobytes()
        == np.array([0.0 + i * delta for i in range(len(got))], dtype=ud.posit16).tobytes()
    )


@pytest.mark.parametrize("name,T", ALL)
def test_arange_edge_lengths(name, T):
    """fill is only consulted for length >= 2; 0 and 1 must still work."""
    assert len(np.arange(0, dtype=T)) == 0
    assert np.arange(0, dtype=T).dtype == np.dtype(T)
    one = np.arange(1, dtype=T)
    assert len(one) == 1 and float(one[0]) == 0.0
    two = np.arange(2, dtype=T)
    assert len(two) == 2


@pytest.mark.parametrize("name,T", ALL)
def test_arange_negative_step(name, T):
    a = np.arange(0, -1, -0.25, dtype=T)
    ref = np.arange(0, -1, -0.25, dtype=np.float64).astype(T)
    assert a.tobytes() == ref.tobytes()
    assert float(a[0]) == 0.0
    assert float(a[1]) < 0.0


def test_arange_saturates_on_bounded_formats():
    """Pinned: a bounded format saturates rather than raising — arange(3) in q15
    (range ±1) is [0, maxpos, maxpos], by the type's own overflow rule."""
    for n in ("q7", "q15", "q31"):
        T = ud.dtypes[n]
        a = np.arange(3, dtype=T)
        maxpos = float(np.array([1e30], dtype=T)[0])
        assert [float(x) for x in a] == [0.0, maxpos, maxpos]


def test_arange_cascade_is_double_precision():
    """Documented limitation: the progression is computed in double, so the
    cascades get double-precision steps rather than their full significand.
    Exact-in-double values are still exact."""
    a = np.arange(0, 1, 0.25, dtype=ud.dd_cascade)
    assert [float(x) for x in a] == [0.0, 0.25, 0.5, 0.75]
