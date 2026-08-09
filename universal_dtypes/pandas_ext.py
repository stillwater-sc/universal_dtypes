"""Optional pandas integration for the universal dtypes (the ``[pandas]`` extra).

Import this module explicitly — the core package never imports pandas:

    import universal_dtypes.pandas_ext   # registers the pandas dtypes
    import pandas as pd, universal_dtypes as ud
    s = pd.array([1.5, 2.25, 3.0], dtype="posit16")

For every universal dtype it registers a pandas ``ExtensionDtype`` /
``ExtensionArray`` pair, thinly backed by the NumPy dtype (pure Python, **no**
MTL5). The classes are also exposed by CamelCase name, e.g. ``Posit16Dtype`` /
``Posit16Array``, so downstream packages (mtl5-python) can re-export them.
"""

from __future__ import annotations

import numpy as np

try:
    from pandas.api.extensions import (
        ExtensionArray,
        ExtensionDtype,
        register_extension_dtype,
        take,
    )
    from pandas.api.types import pandas_dtype
except ImportError as exc:  # pragma: no cover - exercised only without pandas
    raise ImportError(
        "universal_dtypes.pandas_ext requires pandas; install the extra:\n"
        "    pip install 'universal_dtypes[pandas]'"
    ) from exc

import universal_dtypes as _ud

__all__ = ["dtypes", "arrays"]


class _UniversalDtype(ExtensionDtype):
    """Base pandas dtype for a universal number type. Concrete per-type
    subclasses set ``name``, ``type`` (the scalar class) and ``_np_dtype``."""

    _np_dtype: np.dtype
    _array_cls: type
    _is_numeric = True
    _is_boolean = False

    @property
    def na_value(self):
        return np.nan

    @property
    def kind(self) -> str:
        # float-like: values carry a fractional part and NaN semantics
        return "f"

    @classmethod
    def construct_array_type(cls):
        return cls._array_cls


class _UniversalExtensionArray(ExtensionArray):
    """1-D pandas array backed by a NumPy array of the universal dtype. Concrete
    per-type subclasses set ``_scalar_type``, ``_np_dtype`` and ``_dtype``."""

    _scalar_type: type
    _np_dtype: np.dtype
    _dtype: _UniversalDtype

    # ---- construction -------------------------------------------------------
    def __init__(self, values, copy: bool = False):
        if isinstance(values, _UniversalExtensionArray):
            values = values._ndarray
        arr = np.asarray(values, dtype=self._np_dtype)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError("universal_dtypes pandas arrays are 1-dimensional")
        self._ndarray = arr.copy() if copy else arr

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        arr = np.asarray(list(scalars), dtype=cls._np_dtype)
        return cls(arr, copy=copy)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values)

    @classmethod
    def _concat_same_type(cls, to_concat):
        return cls(np.concatenate([x._ndarray for x in to_concat]))

    # ---- required interface -------------------------------------------------
    @property
    def dtype(self):
        return self._dtype

    def __len__(self):
        return len(self._ndarray)

    def __getitem__(self, item):
        result = self._ndarray[item]
        if np.ndim(result) == 0:  # scalar position -> a universal scalar
            return result
        return type(self)(result)

    def __setitem__(self, key, value):
        if isinstance(value, _UniversalExtensionArray):
            value = value._ndarray
        self._ndarray[key] = value

    @property
    def nbytes(self) -> int:
        return self._ndarray.nbytes

    def isna(self):
        # NaN/NaR maps onto NaN via the float64 view; types without a NaN never
        # report NA (like NumPy integers).
        with np.errstate(invalid="ignore"):
            return np.isnan(self._ndarray.astype(np.float64))

    def take(self, indices, *, allow_fill=False, fill_value=None):
        if allow_fill and fill_value is None:
            fill_value = np.nan
        result = take(self._ndarray, indices, allow_fill=allow_fill, fill_value=fill_value)
        return type(self)(np.asarray(result, dtype=self._np_dtype))

    def copy(self):
        return type(self)(self._ndarray.copy())

    # ---- interop / display --------------------------------------------------
    def __array__(self, dtype=None, copy=None):
        if dtype is None:
            return self._ndarray
        return self._ndarray.astype(dtype)

    def __eq__(self, other):
        if isinstance(other, _UniversalExtensionArray):
            other = other._ndarray
        return np.asarray(self._ndarray == other, dtype=bool)

    def __ne__(self, other):
        return ~(self == other)

    def astype(self, dtype, copy=True):
        dtype = pandas_dtype(dtype)
        if isinstance(dtype, _UniversalDtype):
            # value-domain cross-cast between universal dtypes (see issue #39)
            arr = self._ndarray.astype(dtype._np_dtype)
            return dtype.construct_array_type()(arr, copy=False)
        return self._ndarray.astype(dtype, copy=copy)

    def _values_for_factorize(self):
        # scalars are hashable (issue #42); factorize on the object view, NaN as
        # the NA sentinel.
        return self._ndarray.astype(object), np.nan

    def _reduce(self, name, *, skipna=True, keepdims=False, **kwargs):
        arr = self._ndarray
        if skipna:
            mask = self.isna()
            if mask.any():
                arr = arr[~mask]
        func = {"sum": np.sum, "prod": np.prod, "min": np.min, "max": np.max}.get(name)
        if func is None:
            raise TypeError(f"cannot perform '{name}' with dtype {self.dtype.name}")
        result = func(arr)
        return type(self)(np.asarray([result], dtype=self._np_dtype)) if keepdims else result

    def _formatter(self, boxed=False):
        def fmt(x):
            v = float(x)
            return "NaN" if np.isnan(v) else repr(v)

        return fmt


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


dtypes: dict[str, _UniversalDtype] = {}
arrays: dict[str, type] = {}


def _register(name: str, scalar_type: type) -> None:
    np_dtype = np.dtype(scalar_type)
    array_cls = type(
        f"{_camel(name)}Array",
        (_UniversalExtensionArray,),
        {"_scalar_type": scalar_type, "_np_dtype": np_dtype},
    )
    dtype_cls = register_extension_dtype(
        type(
            f"{_camel(name)}Dtype",
            (_UniversalDtype,),
            {
                "name": name,
                "type": scalar_type,
                "_np_dtype": np_dtype,
                "_array_cls": array_cls,
            },
        )
    )
    dtype_instance = dtype_cls()
    array_cls._dtype = dtype_instance
    dtypes[name] = dtype_instance
    arrays[name] = array_cls
    globals()[f"{_camel(name)}Dtype"] = dtype_cls
    globals()[f"{_camel(name)}Array"] = array_cls
    __all__.extend([f"{_camel(name)}Dtype", f"{_camel(name)}Array"])


for _name, _scalar in _ud.dtypes.items():
    _register(_name, _scalar)
