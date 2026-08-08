"""Smoke tests for the universal_dtypes scaffold.

These validate that the extension builds, imports, and links Universal. The real
dtype tests arrive with the dtype implementation (see docs/design.md).
"""

import universal_dtypes


def test_version_is_semver():
    parts = universal_dtypes.__version__.split(".")
    assert len(parts) >= 3, f"expected semver, got {universal_dtypes.__version__!r}"
    assert all(p.isdigit() for p in parts[:3])


def test_build_info():
    info = universal_dtypes.build_info()
    assert info["universal"] is True
    assert info["dtypes"] is False  # not implemented yet


def test_posit16_roundtrip_exact_values():
    # Exactly representable in posit<16,2>: powers of two near 1.0.
    for x in (0.0, 0.5, 1.0, 2.0, -1.0):
        assert universal_dtypes.posit16_roundtrip(x) == x


def test_posit16_roundtrip_rounds():
    # A value not exactly representable comes back close but not identical.
    y = universal_dtypes.posit16_roundtrip(0.1)
    assert abs(y - 0.1) < 1e-3
