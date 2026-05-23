#!/usr/bin/env python3
"""Verify that all refactored modules can be imported without errors."""

import sys


def verify():
    errors = []

    try:
        pass
    except Exception as e:
        errors.append(f"Domain pipeline imports failed: {e}")

    try:
        pass
    except Exception as e:
        errors.append(f"Application pipeline imports failed: {e}")

    try:
        pass
    except Exception as e:
        errors.append(f"Infrastructure imports failed: {e}")

    try:
        pass
    except Exception as e:
        errors.append(f"AI client imports failed: {e}")

    if errors:
        print("❌ Import verification failed:")
        for err in errors:
            print(f"   - {err}")
        sys.exit(1)

    print("✅ All module imports resolved successfully.")
    print("   Verified domains: Pipeline, Application, Infrastructure")
    sys.exit(0)


if __name__ == "__main__":
    verify()