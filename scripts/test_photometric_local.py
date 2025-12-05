# File: scripts/test_photometric_local.py
"""
Test photometric mean squared error averaged over each config's window.

photo_ms_local(x,y) = mean(photo_log_raw²) over winsize×winsize

Parallels the perturbation approach where per-config mean(MS) works well.

Usage:
    python scripts/test_photometric_local.py data/.../results_full.pkl
"""

import numpy as np
import pickle
import sys
import re
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats


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
    u_truth = u_data[list(u_data.keys())[0]]
    v_truth = v_data[list(v_data.keys())[0]]
    
    valid_mask = (
        ~np.isnan(u_truth) & ~np.isnan(v_truth) &
        (np.abs(u_truth) < 1e8) & (np.abs(v_truth) < 1e8)
    )
    
    return u_truth, v_truth, valid_mask


def compute_pixelwise_correlation(metric_stack, epe_stack, valid_mask):
    """Compute Spearman correlation at each pixel, return array of correlations."""
    n_configs, H, W = metric_stack.shape
    
    valid_ys, valid_xs = np.where(valid_mask)
    
    metric_vals = metric_stack[:, valid_ys, valid_xs]
    epe_vals = epe_stack[:, valid_ys, valid_xs]
    
    # Spearman via ranks
    metric_ranks = np.argsort(np.argsort(metric_vals, axis=0), axis=0).astype(np.float32)
    epe_ranks = np.argsort(np.argsort(epe_vals, axis=0), axis=0).astype(np.float32)
    
    metric_centered = metric_ranks - metric_ranks.mean(axis=0, keepdims=True)
    epe_centered = epe_ranks - epe_ranks.mean(axis=0, keepdims=True)
    
    numerator = np.sum(metric_centered * epe_centered, axis=0)
    denom = np.sqrt(np.sum(metric_centered**2, axis=0)) * np.sqrt(np.sum(epe_centered**2, axis=0))
    
    valid_denom = denom > 1e-10
    correlations = np.full(len(valid_ys), np.nan, dtype=np.float32)
    correlations[valid_denom] = numerator[valid_denom] / denom[valid_denom]
    
    # Also return full map for visualization
    corr_map = np.full((H, W), np.nan, dtype=np.float32)
    corr_map[valid_ys, valid_xs] = correlations
    
    return correlations, corr_map


def compute_config_correlation(metric_stack, epe_stack, valid_mask):
    """Compute Spearman correlation across configs using image means."""
    n_configs = metric_stack.shape[0]
    
    mean_metrics = np.zeros(n_configs)
    mean_epes = np.zeros(n_configs)
    
    for i in range(n_configs):
        mean_metrics[i] = np.nanmean(metric_stack[i][valid_mask])
        mean_epes[i] = np.nanmean(epe_stack[i][valid_mask])
    
    rho, _ = scipy_stats.spearmanr(mean_metrics, mean_epes)
    return rho


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_photometric_local.py <results_full.pkl>")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    
    print(f"📂 Loading {results_path}")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    n_configs = len(results)
    print(f"   {n_configs} configurations")
    
    # Load ground truth
    u_truth, v_truth, valid_mask = load_ground_truth(results_path)
    H, W = u_truth.shape
    print(f"   Shape: {H}×{W}, valid: {valid_mask.sum()}")
    
    # Build EPE stack
    print("\n📊 Computing EPE stack...")
    epe_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    for i, r in enumerate(results):
        u = r['flows']['u_AB']
        v = r['flows']['v_AB']
        epe_stack[i] = np.sqrt((u - u_truth)**2 + (v - v_truth)**2)
    
    # Build metric stacks
    print("\n📊 Computing metric stacks...")
    
    # 1. Raw photo_log (baseline)
    photo_raw_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    # 2. Mean squared over winsize (proposed)
    photo_ms_local_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    # 3. Mean (not squared) over winsize 
    photo_mean_local_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    # 4. Sum squared over winsize (energy, not mean)
    photo_energy_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    for i, r in enumerate(results):
        config_name = r['metadata'].get('config_name', '')
        match = re.search(r'win(\d+)', config_name)
        winsize = int(match.group(1)) if match else 15
        
        photo_raw = r['metrics']['photo_log_raw_A']
        photo_squared = photo_raw ** 2
        
        # 1. Raw
        photo_raw_stack[i] = photo_raw
        
        # 2. Mean squared over winsize
        photo_ms_local_stack[i] = cv2.boxFilter(
            photo_squared.astype(np.float32), -1, (winsize, winsize),
            normalize=True, borderType=cv2.BORDER_REFLECT
        )
        
        # 3. Mean (not squared) over winsize
        photo_mean_local_stack[i] = cv2.boxFilter(
            photo_raw.astype(np.float32), -1, (winsize, winsize),
            normalize=True, borderType=cv2.BORDER_REFLECT
        )
        
        # 4. Sum squared over winsize (energy)
        photo_energy_stack[i] = cv2.boxFilter(
            photo_squared.astype(np.float32), -1, (winsize, winsize),
            normalize=False, borderType=cv2.BORDER_REFLECT
        )
    
    # Compute correlations
    print("\n📊 Computing correlations...")
    
    methods = {
        'photo_log raw (baseline)': photo_raw_stack,
        'mean(photo²) over winsize': photo_ms_local_stack,
        'mean(photo) over winsize': photo_mean_local_stack,
        'sum(photo²) over winsize': photo_energy_stack,
    }
    
    results_table = {}
    
    for name, stack in methods.items():
        print(f"   {name}...")
        pixel_corrs, corr_map = compute_pixelwise_correlation(stack, epe_stack, valid_mask)
        config_rho = compute_config_correlation(stack, epe_stack, valid_mask)
        
        results_table[name] = {
            'pixel_corrs': pixel_corrs,
            'corr_map': corr_map,
            'pixel_mean': np.nanmean(pixel_corrs),
            'pixel_median': np.nanmedian(pixel_corrs),
            'pixel_pos_frac': np.nanmean(pixel_corrs > 0) * 100,
            'config_rho': config_rho,
        }
    
    # Print summary
    print("\n" + "=" * 90)
    print("PHOTOMETRIC LOCAL AVERAGING: CORRELATION SUMMARY")
    print("=" * 90)
    print(f"{'Method':<30} | {'Per-pixel ρ':>12} | {'median':>8} | {'+ %':>6} | {'Per-config ρ':>13}")
    print("-" * 90)
    for name, stats in results_table.items():
        print(f"{name:<30} | {stats['pixel_mean']:>+12.3f} | {stats['pixel_median']:>+8.3f} | "
              f"{stats['pixel_pos_frac']:>5.1f}% | {stats['config_rho']:>+13.3f}")
    print("=" * 90)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (name, stats) in enumerate(results_table.items()):
        ax = axes.flat[idx]
        ax.hist(stats['pixel_corrs'], bins=50, range=(-1, 1),
                color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.axvline(stats['pixel_mean'], color='red', linestyle='-', linewidth=2)
        ax.set_xlabel('ρ(metric, EPE)')
        ax.set_ylabel('Pixel count')
        ax.set_title(f"{name}\nmean ρ={stats['pixel_mean']:.3f}, config ρ={stats['config_rho']:.3f}")
        ax.set_xlim(-1, 1)
    
    plt.suptitle('Photometric Local Averaging: Pixel-wise Correlation Distributions', fontsize=14)
    plt.tight_layout()
    
    output_path = results_path.parent / 'photometric_local_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved histogram to {output_path}")
    
    # Heatmaps
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (name, stats) in enumerate(results_table.items()):
        ax = axes2.flat[idx]
        display_map = stats['corr_map'].copy()
        display_map[~valid_mask] = np.nan
        
        im = ax.imshow(display_map, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f"{name}\nmean ρ={stats['pixel_mean']:.3f}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Photometric Local Averaging: Pixel-wise Correlation Heatmaps', fontsize=14)
    plt.tight_layout()
    
    output_path2 = results_path.parent / 'photometric_local_heatmaps.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved heatmaps to {output_path2}")


if __name__ == "__main__":
    main()
