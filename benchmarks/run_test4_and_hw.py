"""Run Test 4 (Numerical Stability) and Hardware Analysis."""

import sys
from pathlib import Path

# Import test 4 from stress tests
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.kv_cache_stress import test_numerical_stability
from benchmarks.kv_cache_hw_analysis import main as hw_analysis_main

if __name__ == "__main__":
    print("Running Test 4: Numerical Stability")
    print("="*70)
    test_numerical_stability()

    print("\n\nRunning Hardware Analysis")
    print("="*70)
    hw_analysis_main()
