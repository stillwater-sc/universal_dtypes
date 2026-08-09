"""Run every example script as a subprocess and require a clean exit.

Each example under ``examples/`` asserts its own "universal wins" outcome, so
executing it here keeps the examples honest against API changes (anti-rot). Runs
on every CI platform via the normal pytest step.
"""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
# recursive: examples are organized into per-family subdirectories
EXAMPLE_SCRIPTS = sorted(EXAMPLES_DIR.rglob("*.py"))


@pytest.mark.skipif(not EXAMPLE_SCRIPTS, reason="examples/ not present (installed package)")
@pytest.mark.parametrize(
    "script", EXAMPLE_SCRIPTS, ids=[str(p.relative_to(EXAMPLES_DIR)) for p in EXAMPLE_SCRIPTS]
)
def test_example_runs(script):
    # Run from a neutral cwd so `import universal_dtypes` resolves to the installed
    # package, not the repo source tree.
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=str(EXAMPLES_DIR),
        timeout=120,
    )
    assert result.returncode == 0, (
        f"{script.name} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
