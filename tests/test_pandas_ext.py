"""pandas ExtensionDtype/ExtensionArray integration — issue #47.

Exercises the optional pandas layer (the ``[pandas]`` extra). Skipped entirely if
pandas isn't installed.
"""

import subprocess
import sys

import numpy as np
import pytest

import universal_dtypes as ud

# Several cases here intentionally use values the ±1 fractional formats cannot hold
# (e.g. the scalar 2), which now reports saturation — see #60 and
# test_saturation_warning.py. That signal is not what these tests are about.
pytestmark = pytest.mark.filterwarnings(
    "ignore:(value .* out of range|overflow encountered in cast)"
)

pd = pytest.importorskip("pandas")
import universal_dtypes.pandas_ext as upd  # noqa: E402

ALL = list(ud.dtypes.items())
HAS_NAN = [(n, T) for n, T in ALL if np.isnan(float(np.array([np.nan], dtype=T)[0]))]
# a representative spread for the heavier ops
REPR = [(n, ud.dtypes[n]) for n in ("posit16", "bfloat16", "dd_cascade", "fixpnt16", "q15")]


# ---- construction & round-trip ----------------------------------------------


@pytest.mark.parametrize("name,T", ALL)
def test_construct_by_string_name(name, T):
    a = pd.array([0.5, 0.25, -0.75], dtype=name)
    assert a.dtype.name == name
    assert type(a).__name__ == f"{upd._camel(name)}Array"
    assert list(a.astype(np.float64)) == list(
        np.array([0.5, 0.25, -0.75], dtype=T).astype(np.float64)
    )


@pytest.mark.parametrize("name,T", ALL)
def test_series_roundtrip(name, T):
    s = pd.Series([0.5, -0.25, 0.125], dtype=name)
    assert s.dtype.name == name
    assert (
        s.astype(np.float64).tolist()
        == np.array([0.5, -0.25, 0.125], dtype=T).astype(np.float64).tolist()
    )


@pytest.mark.parametrize("name,T", ALL)
def test_indexing_and_slicing(name, T):
    a = pd.array([0.5, 0.25, -0.5], dtype=name)
    assert float(a[0]) == float(T(0.5))
    sl = a[1:]
    assert type(sl) is type(a) and len(sl) == 2


# ---- NA handling ------------------------------------------------------------


@pytest.mark.parametrize("name,T", HAS_NAN)
def test_isna_true_for_nan(name, T):
    a = pd.array([1.0, float("nan"), 2.0], dtype=name)
    assert list(a.isna()) == [False, True, False]


@pytest.mark.parametrize("name,T", [(n, t) for n, t in ALL if (n, t) not in HAS_NAN])
def test_isna_all_false_without_nan(name, T):
    a = pd.array([1.0, 2.0, 3.0], dtype=name)
    assert not a.isna().any()


# ---- reductions -------------------------------------------------------------


@pytest.mark.parametrize("name,T", REPR)
def test_reductions(name, T):
    s = pd.Series([0.5, 0.25, 0.125], dtype=name)
    assert np.isclose(float(s.sum()), 0.875, atol=1e-2)
    assert float(s.max()) == float(T(0.5))
    assert float(s.min()) == float(T(0.125))


# ---- astype -----------------------------------------------------------------


def test_astype_cross_universal_dtype():
    a = pd.array([1.5, 2.25], dtype="posit16")
    b = a.astype("posit32")  # value-domain cross-cast (#39)
    assert b.dtype.name == "posit32"
    assert list(b.astype(np.float64)) == list(a.astype(np.float64))


def test_astype_to_and_from_numpy():
    a = pd.array([1.5, 2.5], dtype="posit16")
    f = a.astype(np.float64)
    assert isinstance(f, np.ndarray) and f.dtype == np.float64
    back = pd.array(f, dtype="posit16")
    assert list(back.astype(np.float64)) == [1.5, 2.5]


# ---- take / concat / copy ---------------------------------------------------


@pytest.mark.parametrize("name,T", REPR)
def test_take_with_fill(name, T):
    a = pd.array([0.5, 0.25, 0.75], dtype=name)
    t = a.take([2, 0, -1], allow_fill=True)
    vals = np.asarray(t).astype(np.float64)
    assert vals[0] == float(T(0.75)) and vals[1] == float(T(0.5))
    assert np.isnan(vals[2]) or vals[2] == float(T(np.nan))  # fill slot


@pytest.mark.parametrize("name,T", REPR)
def test_concat_and_copy(name, T):
    s1 = pd.Series([0.5, 0.25], dtype=name)
    s2 = pd.Series([-0.5], dtype=name)
    c = pd.concat([s1, s2], ignore_index=True)
    assert len(c) == 3 and c.dtype.name == name
    cp = s1.array.copy()
    cp[0] = T(0.125)
    assert float(s1.array[0]) == float(T(0.5))  # copy is independent


# ---- factorize / unique -----------------------------------------------------


def test_factorize_and_unique():
    s = pd.Series([0.5, 0.25, 0.5, 0.25, 0.5], dtype="posit16")
    codes, uniques = s.factorize()
    assert len(uniques) == 2
    assert list(codes) == [0, 1, 0, 1, 0]


# ---- dataframe & downstream re-export ---------------------------------------


def test_dataframe_column():
    df = pd.DataFrame({"x": pd.array([1.0, 2.0], dtype="posit16")})
    assert df["x"].dtype.name == "posit16"


def test_classes_exposed_for_reexport():
    # mtl5-python re-exports these by name
    assert hasattr(upd, "Posit16Dtype") and hasattr(upd, "Posit16Array")
    assert upd.dtypes["posit16"].name == "posit16"
    assert upd.arrays["posit16"] is upd.Posit16Array


# ---- the core stays pandas-free ---------------------------------------------


def test_core_import_does_not_import_pandas(tmp_path):
    # run from a neutral cwd so the *installed* package is imported, not the
    # source tree (which lacks the compiled _core in a non-editable install)
    r = subprocess.run(
        [sys.executable, "-c", "import sys, universal_dtypes; print('pandas' in sys.modules)"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"
