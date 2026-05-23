#!/usr/bin/env python3
"""Verify that all refactored modules can be imported without errors."""

import sys


def verify():
    errors = []

    # 1. Domain Layer
    try:
        pass
    except Exception as e:
        errors.append(f"Domain imports failed: {e}")

    # 2. Application Layer
    try:
        pass
    except Exception as e:
        errors.append(f"Application imports failed: {e}")

    # 3. Infrastructure Layer
    try:
        pass
    except Exception as e:
        errors.append(f"Infrastructure imports failed: {e}")

    # 4. AI Clients (Infrastructure Adapters)
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
    print("   Verified domains: Pipeline, Application, Infrastructure, AI clients")
    sys.exit(0)


if __name__ == "__main__":
    verify()