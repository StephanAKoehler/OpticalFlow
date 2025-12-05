# File: scripts/pixel_correlation_heatmap.py
"""
Visualize spatial heatmaps of pixel-wise metric-EPE correlations.

For each pixel, computes Spearman correlation between metric values (across configs)
and EPE values (across configs). Shows where each metric is a good/bad predictor.

Usage:
    python scripts/pixel_correlation_heatmap.py data/.../results_full.pkl [--normalize]
"""

import numpy as np
import pickle
import sys
import re
import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_ground_truth(results_path: Path):
    """Load ground truth from frames directory."""
    # Navigate up to find frames
    # results_path: data/MOVIE/analysis/OF/sweep/pair_XXX/results_full.pkl
    # frames: data/MOVIE/frames
    pair_dir = results_path.parent      # pair_XXX
    sweep_dir = pair_dir.parent         # sweep
    of_dir = sweep_dir.parent           # OF hash
    analysis_dir = of_dir.parent        # analysis
    movie_dir = analysis_dir.parent     # MOVIE hash
    frames_dir = movie_dir / 'frames'
    
    u_path = frames_dir / 'u_000.npz'
    v_path = frames_dir / 'v_000.npz'
    
    if not u_path.exists():
        print(f"❌ Ground truth not found at {frames_dir}")
        sys.exit(1)
    
    # Load npz - handle different key names
    u_data = np.load(u_path)
    v_data = np.load(v_path)
    
    # Get the first (and likely only) array
    u_key = list(u_data.keys())[0]
    v_key = list(v_data.keys())[0]
    u_truth = u_data[u_key]
    v_truth = v_data[v_key]
    
    # Valid mask: not NaN and not exactly 1e9
    valid_mask = (
        ~np.isnan(u_truth) & ~np.isnan(v_truth) &
        (np.abs(u_truth) < 1e8) & (np.abs(v_truth) < 1e8)
    )
    
    return u_truth, v_truth, valid_mask, frames_dir


def compute_epe_stack(results, u_truth, v_truth):
    """Build EPE stack (n_configs, H, W)."""
    n_configs = len(results)
    H, W = u_truth.shape
    
    epe_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    for i, r in enumerate(results):
        u = r['flows']['u_AB']
        v = r['flows']['v_AB']
        epe = np.sqrt((u - u_truth)**2 + (v - v_truth)**2)
        epe_stack[i] = epe
    
    return epe_stack


def compute_metric_stack(results, metric_key):
    """Build metric stack (n_configs, H, W)."""
    n_configs = len(results)
    first = results[0]['metrics'][metric_key]
    H, W = first.shape
    
    stack = np.zeros((n_configs, H, W), dtype=np.float32)
    for i, r in enumerate(results):
        stack[i] = r['metrics'][metric_key]
    
    return stack


def compute_depth_scale_stack(results, pert_dist=1.0):
    """Get depth scale per config."""
    n_configs = len(results)
    depth_scales = np.zeros(n_configs, dtype=np.float32)
    
    for i, r in enumerate(results):
        config_name = r['metadata'].get('config_name', '')
        match = re.search(r'win(\d+)', config_name)
        winsize = int(match.group(1)) if match else 15
        depth_scales[i] = winsize / 2 + pert_dist
    
    return depth_scales


def compute_pixelwise_correlation(metric_stack, epe_stack, valid_mask):
    """
    Compute Spearman correlation at each pixel (vectorized).
    
    For each pixel (y, x), compute correlation between
    metric_stack[:, y, x] and epe_stack[:, y, x] across configs.
    
    Returns (H, W) correlation map.
    """
    n_configs, H, W = metric_stack.shape
    corr_map = np.full((H, W), np.nan, dtype=np.float32)
    
    # Get valid pixel indices
    valid_ys, valid_xs = np.where(valid_mask)
    n_valid = len(valid_ys)
    
    print(f"   Computing correlations for {n_valid} valid pixels (vectorized)...")
    
    # Extract values at valid pixels: (n_configs, n_valid)
    metric_vals = metric_stack[:, valid_ys, valid_xs]
    epe_vals = epe_stack[:, valid_ys, valid_xs]
    
    # Compute ranks along axis 0 (across configs) for Spearman
    # scipy.stats.rankdata can work on 2D but let's use argsort trick
    metric_ranks = np.argsort(np.argsort(metric_vals, axis=0), axis=0).astype(np.float32)
    epe_ranks = np.argsort(np.argsort(epe_vals, axis=0), axis=0).astype(np.float32)
    
    # Pearson correlation on ranks = Spearman correlation
    # Formula: r = (n*sum(xy) - sum(x)*sum(y)) / sqrt((n*sum(x²)-(sum(x))²) * (n*sum(y²)-(sum(y))²))
    
    n = n_configs
    
    # Center the ranks
    metric_centered = metric_ranks - metric_ranks.mean(axis=0, keepdims=True)
    epe_centered = epe_ranks - epe_ranks.mean(axis=0, keepdims=True)
    
    # Compute correlation
    numerator = np.sum(metric_centered * epe_centered, axis=0)
    denom_metric = np.sqrt(np.sum(metric_centered**2, axis=0))
    denom_epe = np.sqrt(np.sum(epe_centered**2, axis=0))
    
    # Avoid division by zero
    denom = denom_metric * denom_epe
    valid_denom = denom > 1e-10
    
    correlations = np.zeros(n_valid, dtype=np.float32)
    correlations[valid_denom] = numerator[valid_denom] / denom[valid_denom]
    correlations[~valid_denom] = np.nan
    
    # Place back into map
    corr_map[valid_ys, valid_xs] = correlations
    
    return corr_map


def main():
    if len(sys.argv) < 2:
        print("Usage: python pixel_correlation_heatmap.py <results_full.pkl> [--normalize]")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    normalize = '--normalize' in sys.argv
    
    # Load data
    print(f"📂 Loading {results_path}")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    print(f"   {len(results)} configurations")
    
    # Load ground truth
    u_truth, v_truth, valid_mask, frames_dir = load_ground_truth(results_path)
    H, W = u_truth.shape
    print(f"   Shape: {H}x{W}, valid pixels: {valid_mask.sum()}/{H*W}")
    
    # Load frame_constants if normalizing
    pert_dist = 1.0
    if normalize:
        fc_path = results_path.parent / 'frame_constants.json'
        if fc_path.exists():
            with open(fc_path) as f:
                fc = json.load(f)
            pert_dist = fc.get('perturbation_distance', 1.0)
    
    # Compute EPE stack
    print("📊 Computing EPE stack...")
    epe_stack = compute_epe_stack(results, u_truth, v_truth)
    
    # Define metrics to analyze
    metrics = {
        'photometric': 'photometric_A',
        'traction': 'traction_raw_A' if normalize else 'traction_A',
        'consistency': 'consistency_raw_A' if normalize else 'consistency_A',
        'perturbation': 'perturbation_raw_A' if normalize else 'displacements_sensitivity_A2B',
    }
    
    # Compute correlation maps
    corr_maps = {}
    for name, key in metrics.items():
        print(f"📊 Processing {name}...")
        
        if key not in results[0]['metrics']:
            print(f"   ⚠️  {key} not found, skipping")
            continue
        
        metric_stack = compute_metric_stack(results, key)
        
        # Apply depth normalization if requested
        if normalize and name in ['traction', 'consistency', 'perturbation']:
            depth_scales = compute_depth_scale_stack(results, pert_dist)
            # Scale each config's metric by its depth_scale
            for i in range(len(results)):
                metric_stack[i] *= depth_scales[i]
            print(f"   Applied depth normalization (scales: {depth_scales.min():.1f}-{depth_scales.max():.1f})")
        
        corr_map = compute_pixelwise_correlation(metric_stack, epe_stack, valid_mask)
        corr_maps[name] = corr_map
    
    # Load first frame for overlay
    frame_path = frames_dir / 'image_000.png'
    if frame_path.exists():
        frame = plt.imread(frame_path)
        if frame.ndim == 3:
            frame_gray = np.mean(frame, axis=2)
        else:
            frame_gray = frame
    else:
        frame_gray = None
    
    # Create figure
    n_metrics = len(corr_maps)
    fig, axes = plt.subplots(2, n_metrics, figsize=(4*n_metrics, 8))
    
    if n_metrics == 1:
        axes = axes.reshape(2, 1)
    
    norm_str = " (depth-normalized)" if normalize else ""
    fig.suptitle(f'Pixel-wise Spearman ρ(metric, EPE){norm_str}', fontsize=14)
    
    for col, (name, corr_map) in enumerate(corr_maps.items()):
        # Top row: correlation heatmap
        ax = axes[0, col]
        
        # Mask invalid pixels
        display_map = corr_map.copy()
        display_map[~valid_mask] = np.nan
        
        im = ax.imshow(display_map, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f'{name}\nmean ρ={np.nanmean(corr_map):.3f}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Bottom row: histogram of correlations
        ax2 = axes[1, col]
        valid_corrs = corr_map[valid_mask & ~np.isnan(corr_map)]
        
        ax2.hist(valid_corrs, bins=50, range=(-1, 1), color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='black', linestyle='--', linewidth=1)
        ax2.axvline(np.nanmean(valid_corrs), color='red', linestyle='-', linewidth=2, label=f'mean={np.nanmean(valid_corrs):.3f}')
        ax2.set_xlabel('ρ(metric, EPE)')
        ax2.set_ylabel('Pixel count')
        ax2.legend(loc='upper right')
        
        # Stats
        pos_frac = np.mean(valid_corrs > 0) * 100
        neg_frac = np.mean(valid_corrs < 0) * 100
        ax2.set_title(f'+: {pos_frac:.0f}%, -: {neg_frac:.0f}%')
    
    plt.tight_layout()
    
    # Save
    suffix = '_normalized' if normalize else ''
    output_path = results_path.parent / f'pixel_correlation_heatmap{suffix}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, corr_map in corr_maps.items():
        valid_corrs = corr_map[valid_mask & ~np.isnan(corr_map)]
        pos_frac = np.mean(valid_corrs > 0) * 100
        print(f"  {name:15s}  mean ρ={np.nanmean(valid_corrs):+.3f}  (+:{pos_frac:.0f}%)")


if __name__ == "__main__":
    main()
