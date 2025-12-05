# File: scripts/test_integrated_perturbation.py
"""
Test integrated perturbation metric for pixel-wise selection.

Proposed metric:
    score(x,y) = ∫∫ pert_raw(window around x,y) / winSize

This integrates over the algorithm's receptive field (winSize×winSize),
then divides by linear window size.

Compares pixel-wise correlation with EPE:
  - Current: pert_raw(x,y) × depth_scale  (single pixel, scaled)
  - Proposed: integrated(x,y) / winSize   (window-matched)

Usage:
    python scripts/test_integrated_perturbation.py data/.../results_full.pkl
"""

import numpy as np
import pickle
import sys
import re
import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt


def load_ground_truth(results_path: Path):
    """Load ground truth from frames directory."""
    pair_dir = results_path.parent
    sweep_dir = pair_dir.parent
    of_dir = sweep_dir.parent
    analysis_dir = of_dir.parent
    movie_dir = analysis_dir.parent
    frames_dir = movie_dir / 'frames'
    
    u_path = frames_dir / 'u_000.npz'
    v_path = frames_dir / 'v_000.npz'
    
    if not u_path.exists():
        print(f"❌ Ground truth not found at {frames_dir}")
        sys.exit(1)
    
    u_data = np.load(u_path)
    v_data = np.load(v_path)
    u_key = list(u_data.keys())[0]
    v_key = list(v_data.keys())[0]
    u_truth = u_data[u_key]
    v_truth = v_data[v_key]
    
    valid_mask = (
        ~np.isnan(u_truth) & ~np.isnan(v_truth) &
        (np.abs(u_truth) < 1e8) & (np.abs(v_truth) < 1e8)
    )
    
    return u_truth, v_truth, valid_mask


def compute_integrated_metric(pert_raw: np.ndarray, winsize: int) -> np.ndarray:
    """
    Compute integrated perturbation metric.
    
    score(x,y) = [∫∫ pert_raw over winSize×winSize window] / winSize
    
    Uses box filter (sum over window), then divides by linear winSize.
    """
    # Box filter with normalize=False gives sum over window
    integrated = cv2.boxFilter(
        pert_raw.astype(np.float32), 
        ddepth=-1, 
        ksize=(winsize, winsize), 
        normalize=False,
        borderType=cv2.BORDER_REFLECT
    )
    
    # Divide by linear window size (not area)
    return integrated / winsize


def compute_integrated_variance_metric(pert_raw: np.ndarray, winsize: int) -> np.ndarray:
    """
    Compute integrated variance (squared perturbation) metric.
    
    score(x,y) = ∫∫ pert_raw² over winSize×winSize window
    
    Total "instability energy" over receptive field.
    No division needed - naturally scales with area.
    """
    pert_squared = pert_raw.astype(np.float32) ** 2
    
    # Sum of squares over window
    integrated_var = cv2.boxFilter(
        pert_squared,
        ddepth=-1,
        ksize=(winsize, winsize),
        normalize=False,
        borderType=cv2.BORDER_REFLECT
    )
    
    return integrated_var


def compute_scaled_metric(pert_raw: np.ndarray, winsize: int, pert_dist: float = 1.0) -> np.ndarray:
    """
    Compute current depth-scaled metric.
    
    score(x,y) = pert_raw(x,y) × (winSize/2 + pert_dist)
    """
    depth_scale = winsize / 2 + pert_dist
    return pert_raw * depth_scale


def compute_pixelwise_correlation(metric_stack, epe_stack, valid_mask):
    """
    Compute Spearman correlation at each pixel (vectorized).
    """
    n_configs, H, W = metric_stack.shape
    corr_map = np.full((H, W), np.nan, dtype=np.float32)
    
    valid_ys, valid_xs = np.where(valid_mask)
    n_valid = len(valid_ys)
    
    # Extract values at valid pixels
    metric_vals = metric_stack[:, valid_ys, valid_xs]
    epe_vals = epe_stack[:, valid_ys, valid_xs]
    
    # Compute ranks for Spearman
    metric_ranks = np.argsort(np.argsort(metric_vals, axis=0), axis=0).astype(np.float32)
    epe_ranks = np.argsort(np.argsort(epe_vals, axis=0), axis=0).astype(np.float32)
    
    # Pearson on ranks = Spearman
    metric_centered = metric_ranks - metric_ranks.mean(axis=0, keepdims=True)
    epe_centered = epe_ranks - epe_ranks.mean(axis=0, keepdims=True)
    
    numerator = np.sum(metric_centered * epe_centered, axis=0)
    denom_metric = np.sqrt(np.sum(metric_centered**2, axis=0))
    denom_epe = np.sqrt(np.sum(epe_centered**2, axis=0))
    
    denom = denom_metric * denom_epe
    valid_denom = denom > 1e-10
    
    correlations = np.zeros(n_valid, dtype=np.float32)
    correlations[valid_denom] = numerator[valid_denom] / denom[valid_denom]
    correlations[~valid_denom] = np.nan
    
    corr_map[valid_ys, valid_xs] = correlations
    
    return corr_map


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_integrated_perturbation.py <results_full.pkl>")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    
    # Load results
    print(f"📂 Loading {results_path}")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    n_configs = len(results)
    print(f"   {n_configs} configurations")
    
    # Load frame constants
    fc_path = results_path.parent / 'frame_constants.json'
    pert_dist = 1.0
    if fc_path.exists():
        with open(fc_path) as f:
            fc = json.load(f)
        pert_dist = fc.get('perturbation_distance', 1.0)
    
    # Load ground truth
    u_truth, v_truth, valid_mask = load_ground_truth(results_path)
    H, W = u_truth.shape
    print(f"   Shape: {H}x{W}, valid: {valid_mask.sum()}")
    
    # Build EPE stack
    print("📊 Computing EPE stack...")
    epe_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    for i, r in enumerate(results):
        u = r['flows']['u_AB']
        v = r['flows']['v_AB']
        epe_stack[i] = np.sqrt((u - u_truth)**2 + (v - v_truth)**2)
    
    # Build metric stacks
    print("📊 Computing metric stacks...")
    
    # Current: single-pixel × depth_scale
    scaled_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    # Proposed: integrated / winSize
    integrated_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    # Also test: integrated / winSize² (normalize by area)
    integrated_area_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    # NEW: integrated variance (sum of squares, no division)
    variance_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    for i, r in enumerate(results):
        config_name = r['metadata'].get('config_name', '')
        match = re.search(r'win(\d+)', config_name)
        winsize = int(match.group(1)) if match else 15
        
        pert_raw = r['metrics']['perturbation_raw_A']
        
        # Current approach
        scaled_stack[i] = compute_scaled_metric(pert_raw, winsize, pert_dist)
        
        # Proposed: integrate / winSize
        integrated_stack[i] = compute_integrated_metric(pert_raw, winsize)
        
        # Alternative: integrate / winSize² (pure mean)
        integrated_area_stack[i] = cv2.boxFilter(
            pert_raw.astype(np.float32), -1, (winsize, winsize),
            normalize=True, borderType=cv2.BORDER_REFLECT
        ) * winsize  # mean × winSize to match scaling
        
        # NEW: integrate squared (variance) - no division needed
        variance_stack[i] = compute_integrated_variance_metric(pert_raw, winsize)
    
    # Compute correlations
    print("📊 Computing pixel-wise correlations...")
    
    print("   Scaled (current)...")
    corr_scaled = compute_pixelwise_correlation(scaled_stack, epe_stack, valid_mask)
    
    print("   Integrated (proposed)...")
    corr_integrated = compute_pixelwise_correlation(integrated_stack, epe_stack, valid_mask)
    
    print("   Integrated (area-normalized)...")
    corr_area = compute_pixelwise_correlation(integrated_area_stack, epe_stack, valid_mask)
    
    print("   Integrated variance (∑pert²)...")
    corr_variance = compute_pixelwise_correlation(variance_stack, epe_stack, valid_mask)
    
    # Also compute raw (no depth correction) for reference
    print("   Raw (no depth correction)...")
    raw_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    for i, r in enumerate(results):
        raw_stack[i] = r['metrics']['perturbation_raw_A']
    corr_raw = compute_pixelwise_correlation(raw_stack, epe_stack, valid_mask)
    
    # Summary statistics
    def get_stats(corr_map, valid_mask):
        valid_corrs = corr_map[valid_mask & ~np.isnan(corr_map)]
        return {
            'mean': np.mean(valid_corrs),
            'median': np.median(valid_corrs),
            'pos_frac': np.mean(valid_corrs > 0) * 100,
            'neg_frac': np.mean(valid_corrs < 0) * 100,
            'corrs': valid_corrs
        }
    
    stats_raw = get_stats(corr_raw, valid_mask)
    stats_scaled = get_stats(corr_scaled, valid_mask)
    stats_integrated = get_stats(corr_integrated, valid_mask)
    stats_area = get_stats(corr_area, valid_mask)
    stats_variance = get_stats(corr_variance, valid_mask)
    
    print("\n" + "="*80)
    print("PIXEL-WISE CORRELATION SUMMARY (metric vs EPE)")
    print("="*80)
    print(f"{'Method':<35} | {'mean ρ':>8} | {'median ρ':>8} | {'+ %':>6} | {'- %':>6}")
    print("-"*80)
    print(f"{'raw (no correction)':<35} | {stats_raw['mean']:>+8.3f} | {stats_raw['median']:>+8.3f} | {stats_raw['pos_frac']:>5.1f}% | {stats_raw['neg_frac']:>5.1f}%")
    print(f"{'scaled (current: ×depth)':<35} | {stats_scaled['mean']:>+8.3f} | {stats_scaled['median']:>+8.3f} | {stats_scaled['pos_frac']:>5.1f}% | {stats_scaled['neg_frac']:>5.1f}%")
    print(f"{'integrated (∑pert / winSize)':<35} | {stats_integrated['mean']:>+8.3f} | {stats_integrated['median']:>+8.3f} | {stats_integrated['pos_frac']:>5.1f}% | {stats_integrated['neg_frac']:>5.1f}%")
    print(f"{'integrated (mean × winSize)':<35} | {stats_area['mean']:>+8.3f} | {stats_area['median']:>+8.3f} | {stats_area['pos_frac']:>5.1f}% | {stats_area['neg_frac']:>5.1f}%")
    print(f"{'★ integrated variance (∑pert²)':<35} | {stats_variance['mean']:>+8.3f} | {stats_variance['median']:>+8.3f} | {stats_variance['pos_frac']:>5.1f}% | {stats_variance['neg_frac']:>5.1f}%")
    print("="*80)
    
    # Create figure: 2x3 grid (add variance)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    methods = [
        ('Raw (no correction)', corr_raw, stats_raw),
        ('Scaled (×depth)', corr_scaled, stats_scaled),
        ('Integrated (∑pert / winSize)', corr_integrated, stats_integrated),
        ('Integrated (mean × winSize)', corr_area, stats_area),
        ('★ Variance (∑pert²)', corr_variance, stats_variance),
    ]
    
    for idx, (name, corr_map, stats) in enumerate(methods):
        ax = axes.flat[idx]
        # Histogram
        ax.hist(stats['corrs'], bins=50, range=(-1, 1), 
                color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.axvline(stats['mean'], color='red', linestyle='-', linewidth=2)
        
        ax.set_xlabel('ρ(metric, EPE)')
        ax.set_ylabel('Pixel count')
        ax.set_title(f"{name}\nmean ρ={stats['mean']:.3f}, +:{stats['pos_frac']:.0f}% / -:{stats['neg_frac']:.0f}%")
        ax.set_xlim(-1, 1)
    
    # Hide empty subplot
    axes.flat[-1].axis('off')
    
    plt.suptitle('Pixel-wise Spearman correlation: perturbation metric vs EPE', fontsize=14)
    plt.tight_layout()
    
    output_path = results_path.parent / 'integrated_perturbation_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved histogram to {output_path}")
    
    # Create heatmap figure (2x3 grid)
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, (name, corr_map, stats) in enumerate(methods):
        ax = axes2.flat[idx]
        display_map = corr_map.copy()
        display_map[~valid_mask] = np.nan
        
        im = ax.imshow(display_map, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f"{name}\nmean ρ={stats['mean']:.3f}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide empty subplot
    axes2.flat[-1].axis('off')
    
    plt.suptitle('Pixel-wise correlation heatmaps', fontsize=14)
    plt.tight_layout()
    
    output_path2 = results_path.parent / 'integrated_perturbation_heatmaps.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved heatmaps to {output_path2}")


if __name__ == "__main__":
    main()
