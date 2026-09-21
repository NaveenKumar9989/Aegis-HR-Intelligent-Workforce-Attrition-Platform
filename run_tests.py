"""
Standalone Master Test Runner.
Discovers and executes all unit and integration tests across the test suite.
Usage:
    py -3.14 run_tests.py
"""

import sys
import time
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    print("=" * 72)
    print("      AEGIS HR - AUTOMATED TEST SUITE (UNIT & INTEGRATION TESTS)       ")
    print("=" * 72)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py")

    total_tests = suite.countTestCases()
    print(f"[+] Discovered {total_tests} test cases across 'tests/' directory.")
    print("-" * 72)

    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    elapsed = time.time() - start_time

    print("\n" + "=" * 72)
    print(f"[*] Test Suite Execution Summary:")
    print(f"    - Tests Ran:    {result.testsRun}")
    print(f"    - Successes:    {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"    - Failures:     {len(result.failures)}")
    print(f"    - Errors:       {len(result.errors)}")
    print(f"    - Total Time:   {elapsed:.2f} seconds")
    print("=" * 72)

    if result.wasSuccessful():
        print("[OK] ALL TESTS PASSED SUCCESSFULLY! Codebase is ready for production.\n")
        sys.exit(0)
    else:
        print("[FAIL] Some tests failed. Please review error traces above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
