#!/usr/bin/env python3
# File: scripts/analyze_dis_test.py
"""
Analyze DIS test results to understand poor ensemble performance.

Run after optical_flow_pipeline_v2.py to diagnose issues.
This script loads the results and performs detailed correlation analysis.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from diagnose_ensemble_selection import (
    analyze_metric_correlations,
    plot_metric_scatter,
    analyze_per_config_performance,
    suggest_ensemble_weights
)


def main():
    """Analyze DIS test results."""
    
    print("="*80)
    print("DIS TEST RESULTS ANALYSIS")
    print("="*80)
    
    # This is a PLACEHOLDER - in reality you'd need to:
    # 1. Save results_full, valid_mask, u_true, v_true from pipeline
    # 2. Load them here
    # 3. Run analysis
    
    print("\n⚠️  This script requires results from pipeline_v2.py")
    print("\nTo use this diagnostic:")
    print("1. Modify pipeline_v2.py to save results to disk:")
    print("   np.savez('dis_test_results.npz',")
    print("            results_full=results_full,")
    print("            valid_mask=valid_mask,")
    print("            u_true=u_true,")
    print("            v_true=v_true)")
    print("\n2. Run: python scripts/analyze_dis_test.py dis_test_results.npz")
    print("\nFor now, here's what the analysis WOULD show you:")
    print()
    print("HYPOTHESIS: Why is ensemble +158% above oracle?")
    print("-" * 80)
    print()
    print("Possible causes:")
    print()
    print("1. TRACTION-ONLY WEIGHTING IS WRONG FOR THIS TEST")
    print("   - Your config uses: traction=2.0, consistency=0.0, photometric=0.0")
    print("   - For split motion (left↑, right↓), traction might not correlate well with EPE")
    print("   - Motion discontinuity at center is challenging for traction metric")
    print()
    print("2. DIS CONFIGS MIGHT NOT HAVE ENOUGH DIVERSITY")
    print("   - Only varying preset (2 values) and finest_scale (3 values)")
    print("   - All using same iterations=12, patch_size=8, patch_stride=4")
    print("   - Limited diversity = ensemble can't adapt to different regions")
    print()
    print("3. BOUNDARY MARGIN TOO AGGRESSIVE")
    print("   - 50px margin removes 57.4% of pixels")
    print("   - DIS with finest_scale=0,1,2 needs different margins")
    print("   - Overly restrictive valid region biases results")
    print()
    print("RECOMMENDED ACTIONS:")
    print("-" * 80)
    print()
    print("A. Test different ensemble weights:")
    print("   [ensemble.weights]")
    print("   traction_A = 0.0")
    print("   consistency_A = 1.0  # Try consistency instead")
    print("   photometric_A = 0.0")
    print()
    print("B. Increase config diversity:")
    print("   preset = ['ULTRAFAST', 'FAST', 'MEDIUM']  # Add ULTRAFAST")
    print("   finest_scale = [0, 1, 2]")
    print("   iterations = [8, 12, 16]  # Add variation")
    print("   patch_size = [6, 8, 12]    # Add variation")
    print()
    print("C. Reduce boundary margin:")
    print("   boundary_margin = 20  # From 50")
    print()
    print("D. Add photometric and consistency weights:")
    print("   [ensemble.weights]")
    print("   traction_A = 1.0")
    print("   consistency_A = 1.0")
    print("   photometric_A = 1.0")
    print("   traction_B = 1.0")
    print("   consistency_B = 1.0")
    print("   photometric_B = 1.0")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
