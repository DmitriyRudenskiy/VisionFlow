#!/usr/bin/env python3
"""Cross-platform test runner with import verification."""

import subprocess
import sys
from pathlib import Path


def run_pytest():
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def run_import_check():
    script = Path(__file__).parent / "verify_imports.py"
    result = subprocess.run([sys.executable, str(script)])
    return result.returncode


def main():
    exit_code = 0
    exit_code |= run_pytest()
    exit_code |= run_import_check()
    if exit_code == 0:
        print("\n✅ All checks passed.")
    else:
        print("\n❌ Some checks failed.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()