# File: scripts/analyze_by_region.py
"""
Analyze ensemble vs oracle performance by image region.

Tests hypothesis: ensemble fails at motion discontinuities but works in smooth regions.
"""

import numpy as np
import sys
from pathlib import Path


def analyze_regions(results_full, valid_mask, u_true, v_true, centerline_x=144):
    """
    Analyze performance separately for:
    1. Far left (smooth up motion)
    2. Centerline (motion discontinuity)
    3. Far right (smooth down motion)
    """
    H, W = u_true.shape
    
    # Define regions (avoid edges)
    margin = 30
    center_width = 20
    
    left_mask = valid_mask & (np.arange(W) < centerline_x - center_width)
    center_mask = valid_mask & (np.arange(W) >= centerline_x - center_width) & (np.arange(W) <= centerline_x + center_width)
    right_mask = valid_mask & (np.arange(W) > centerline_x + center_width)
    
    print("\n" + "="*80)
    print("REGIONAL ANALYSIS")
    print("="*80)
    print(f"Image width: {W}, centerline at x={centerline_x}")
    print(f"Regions:")
    print(f"  Left:   x < {centerline_x - center_width} ({left_mask.sum()} pixels)")
    print(f"  Center: {centerline_x - center_width} <= x <= {centerline_x + center_width} ({center_mask.sum()} pixels)")
    print(f"  Right:  x > {centerline_x + center_width} ({right_mask.sum()} pixels)")
    print()
    
    n_configs = len(results_full)
    
    # Compute EPE for all configs
    EPE_stack = []
    for result in results_full:
        u_AB = result['u_AB']
        v_AB = result['v_AB']
        epe = np.sqrt((u_AB - u_true)**2 + (v_AB - v_true)**2)
        EPE_stack.append(epe)
    EPE_stack = np.array(EPE_stack)
    
    # Oracle selection (per-pixel best)
    oracle_selection = np.argmin(EPE_stack, axis=0)
    
    # Compute ensemble cost (traction + consistency + photometric)
    ensemble_cost = np.zeros((n_configs, H, W))
    for i, result in enumerate(results_full):
        cost = (result['traction_A'] + result['traction_B'] +
                result['consistency_A'] + result['consistency_B'] +
                result['photometric_A'] + result['photometric_B'])
        ensemble_cost[i] = cost
    
    ensemble_selection = np.argmin(ensemble_cost, axis=0)
    
    # Analyze each region
    for region_name, region_mask in [('Left', left_mask), ('Center', center_mask), ('Right', right_mask)]:
        if not region_mask.any():
            print(f"{region_name}: No valid pixels")
            continue
            
        # Oracle EPE
        oracle_epe = np.nanmean(EPE_stack[oracle_selection, np.arange(H*W).reshape(H,W)][region_mask])
        
        # Ensemble EPE
        ensemble_epe = np.nanmean(EPE_stack[ensemble_selection, np.arange(H*W).reshape(H,W)][region_mask])
        
        # Best single config EPE
        best_idx = np.argmin([np.nanmean(EPE_stack[i][region_mask]) for i in range(n_configs)])
        best_epe = np.nanmean(EPE_stack[best_idx][region_mask])
        
        # Agreement between ensemble and oracle
        agreement = (ensemble_selection[region_mask] == oracle_selection[region_mask]).mean()
        
        print(f"\n{region_name} Region:")
        print(f"  Oracle EPE:      {oracle_epe:.4f} px")
        print(f"  Ensemble EPE:    {ensemble_epe:.4f} px ({100*(ensemble_epe/oracle_epe-1):+.1f}% above oracle)")
        print(f"  Best single:     {best_epe:.4f} px")
        print(f"  Ensemble-Oracle agreement: {100*agreement:.1f}%")
        
        if ensemble_epe / oracle_epe > 2.0:
            print(f"  ⚠️  Ensemble is >2x worse than oracle in this region!")


if __name__ == "__main__":
    print("This module requires results from pipeline.")
    print("Add to end of pipeline_v2.py main():")
    print()
    print("from analyze_by_region import analyze_regions")
    print("analyze_regions(results_full, valid_mask, u_true, v_true)")
