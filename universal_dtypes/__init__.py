"""NumPy dtypes for the Stillwater Universal number systems.

Early scaffold: the NumPy custom-dtype registration (posit / cfloat / fixpnt /
lns) is not implemented yet — see ``docs/design.md``. The compiled extension
currently exposes only a version, a build-info dict, and a proof-of-life helper
that exercises the Universal dependency.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _pkg_version

# Single source of truth: installed package metadata, falling back to the version
# compiled into the extension when running from a source tree without metadata.
try:
    __version__ = _pkg_version("universal_dtypes")
except _PackageNotFoundError:
    try:
        from universal_dtypes._core import __version__  # noqa: F811
    except ImportError:
        __version__ = "0.0.0-dev"

from universal_dtypes._core import (  # noqa: E402
    Bfloat16DType,
    bfloat16,
    build_info,
    posit16_roundtrip,
)

__all__ = ["__version__", "Bfloat16DType", "bfloat16", "build_info", "posit16_roundtrip"]
