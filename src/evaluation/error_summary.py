#!/usr/bin/env python3
# File: src/evaluation/error_summary.py
"""
Error distribution summary for optical flow evaluation.

Prints breakdown of pixel errors across threshold bins.
"""

import numpy as np


def print_error_distribution(epe_best: np.ndarray,
                             epe_ensemble: np.ndarray,
                             epe_oracle: np.ndarray,
                             valid_mask: np.ndarray):
    """
    Print error distribution table comparing best single, ensemble, and oracle.
    
    Shows binned error counts. Omits final "EPE ≥ X" bin if no pixels exceed threshold.
    
    Args:
        epe_best: EPE map for best single config
        epe_ensemble: EPE map for ensemble
        epe_oracle: EPE map for oracle
        valid_mask: Boolean mask for valid pixels
    """
    # Apply valid mask
    epe_best = epe_best[valid_mask]
    epe_ensemble = epe_ensemble[valid_mask]
    epe_oracle = epe_oracle[valid_mask]
    n_valid = valid_mask.sum()
    
    # Define thresholds
    thresholds = [0.05, 0.1, 0.5, 1.0, 2.0]
    
    print("\nError Distribution (valid pixels):")
    print(f"{'Threshold':<20} {'Best Single':<15} {'Ensemble':<15} {'Oracle':<15}")
    print("-" * 65)
    
    # First bin: EPE < thresholds[0]
    thresh = thresholds[0]
    label = f"EPE < {thresh}px"
    count_best = np.sum(epe_best < thresh)
    count_ens = np.sum(epe_ensemble < thresh)
    count_ora = np.sum(epe_oracle < thresh)
    
    pct_best = 100 * count_best / n_valid
    pct_ens = 100 * count_ens / n_valid
    pct_ora = 100 * count_ora / n_valid
    
    print(f"{label:<20} {count_best:>6} ({pct_best:>5.1f}%)  "
          f"{count_ens:>6} ({pct_ens:>5.1f}%)  "
          f"{count_ora:>6} ({pct_ora:>5.1f}%)")
    
    # Middle bins: thresholds[i-1] ≤ EPE < thresholds[i]
    for i in range(1, len(thresholds)):
        prev = thresholds[i-1]
        curr = thresholds[i]
        label = f"{prev} ≤ EPE < {curr}"
        
        count_best = np.sum((epe_best >= prev) & (epe_best < curr))
        count_ens = np.sum((epe_ensemble >= prev) & (epe_ensemble < curr))
        count_ora = np.sum((epe_oracle >= prev) & (epe_oracle < curr))
        
        pct_best = 100 * count_best / n_valid
        pct_ens = 100 * count_ens / n_valid
        pct_ora = 100 * count_ora / n_valid
        
        print(f"{label:<20} {count_best:>6} ({pct_best:>5.1f}%)  "
              f"{count_ens:>6} ({pct_ens:>5.1f}%)  "
              f"{count_ora:>6} ({pct_ora:>5.1f}%)")
    
    # Final bin: EPE ≥ thresholds[-1] (only if there are such pixels)
    max_thresh = thresholds[-1]
    count_best = np.sum(epe_best >= max_thresh)
    count_ens = np.sum(epe_ensemble >= max_thresh)
    count_ora = np.sum(epe_oracle >= max_thresh)
    
    # Only print this row if at least one method has pixels in this range
    if count_best > 0 or count_ens > 0 or count_ora > 0:
        label = f"EPE ≥ {max_thresh}px"
        pct_best = 100 * count_best / n_valid
        pct_ens = 100 * count_ens / n_valid
        pct_ora = 100 * count_ora / n_valid
        
        print(f"{label:<20} {count_best:>6} ({pct_best:>5.1f}%)  "
              f"{count_ens:>6} ({pct_ens:>5.1f}%)  "
              f"{count_ora:>6} ({pct_ora:>5.1f}%)")


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    # Create test EPE maps (most pixels good, some bad)
    n_pixels = 70000
    valid_mask = np.ones(n_pixels, dtype=bool)
    
    # Best single: mostly < 0.05, some outliers
    epe_best = np.concatenate([
        np.random.uniform(0, 0.05, 66500),
        np.random.uniform(0.05, 0.1, 2500),
        np.random.uniform(0.1, 0.5, 800),
        np.random.uniform(0.5, 1.0, 150),
        np.random.uniform(1.0, 2.0, 50)
    ])
    
    # Ensemble: slightly better
    epe_ensemble = epe_best * 0.95
    
    # Oracle: much better
    epe_oracle = epe_best * 0.4
    
    print("Test error distribution:")
    print_error_distribution(epe_best, epe_ensemble, epe_oracle, valid_mask)
