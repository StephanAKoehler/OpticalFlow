# File: scripts/test_energy_window_scaling.py
"""
Test energy integration at multiple window scales.

For each k = 1, 2, 4, 8, ...:
    energy(x,y) = ∫∫ var over (k × winsize)²

Two correlations computed:
1. Per-pixel: at each pixel, rank configs by energy vs EPE → mean ρ across pixels
2. Per-config: mean(energy) over image vs mean(EPE) → ρ across configs

Usage:
    python scripts/test_energy_window_scaling.py data/.../results_full.pkl
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


def compute_energy(var_map: np.ndarray, winsize: int, H: int, W: int) -> np.ndarray:
    """
    Compute ∫∫ var over winsize×winsize window.
    Clamps winsize to image dimensions.
    """
    # Clamp to image size (must be odd for boxFilter)
    effective_win = min(winsize, H, W)
    if effective_win % 2 == 0:
        effective_win -= 1
    effective_win = max(effective_win, 1)
    
    energy = cv2.boxFilter(
        var_map.astype(np.float32),
        ddepth=-1,
        ksize=(effective_win, effective_win),
        normalize=False,
        borderType=cv2.BORDER_REFLECT
    )
    
    return energy


def compute_pixelwise_correlation(metric_stack, epe_stack, valid_mask):
    """Compute Spearman correlation at each pixel, return mean."""
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
    
    return correlations


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
        print("Usage: python test_energy_window_scaling.py <results_full.pkl>")
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
    
    # Extract config data
    print("\n📊 Extracting config data...")
    config_data = []
    for i, r in enumerate(results):
        config_name = r['metadata'].get('config_name', '')
        match = re.search(r'win(\d+)', config_name)
        winsize = int(match.group(1)) if match else 15
        pert_raw = r['metrics']['perturbation_raw_A']
        var_map = pert_raw ** 2  # variance
        
        config_data.append({
            'idx': i,
            'name': config_name,
            'winsize': winsize,
            'var_map': var_map,
        })
    
    # Window multipliers to test
    k_values = [1, 2, 4, 8, 16, 32]
    
    # Store results
    results_table = []
    
    print("\n📊 Testing window scales...")
    print("-" * 70)
    
    for k in k_values:
        print(f"\n   k = {k}×")
        
        # Build energy stack for this k
        energy_stack = np.zeros((n_configs, H, W), dtype=np.float32)
        
        effective_winsizes = []
        for c in config_data:
            i = c['idx']
            winsize = c['winsize']
            var_map = c['var_map']
            
            scaled_win = k * winsize
            effective_winsizes.append(min(scaled_win, H, W))
            
            energy_stack[i] = compute_energy(var_map, scaled_win, H, W)
        
        # Compute per-pixel correlation
        pixel_corrs = compute_pixelwise_correlation(energy_stack, epe_stack, valid_mask)
        mean_pixel_rho = np.nanmean(pixel_corrs)
        median_pixel_rho = np.nanmedian(pixel_corrs)
        pos_frac = np.nanmean(pixel_corrs > 0) * 100
        
        # Compute per-config correlation
        config_rho = compute_config_correlation(energy_stack, epe_stack, valid_mask)
        
        # Effective window range
        min_eff = min(effective_winsizes)
        max_eff = max(effective_winsizes)
        
        results_table.append({
            'k': k,
            'min_win': min_eff,
            'max_win': max_eff,
            'pixel_rho_mean': mean_pixel_rho,
            'pixel_rho_median': median_pixel_rho,
            'pixel_pos_frac': pos_frac,
            'config_rho': config_rho,
            'pixel_corrs': pixel_corrs,
        })
        
        print(f"      Window range: {min_eff}-{max_eff} px")
        print(f"      Per-pixel:  mean ρ = {mean_pixel_rho:+.3f}, median ρ = {median_pixel_rho:+.3f}, +:{pos_frac:.1f}%")
        print(f"      Per-config: ρ = {config_rho:+.3f}")
    
    # Summary table
    print("\n" + "=" * 85)
    print("ENERGY INTEGRATION: WINDOW SCALE ANALYSIS")
    print("=" * 85)
    print(f"{'k':<5} | {'Window (px)':<12} | {'Per-pixel ρ':>12} | {'median':>8} | {'+ %':>6} | {'Per-config ρ':>13}")
    print("-" * 85)
    for r in results_table:
        win_str = f"{r['min_win']}-{r['max_win']}"
        print(f"{r['k']:<5} | {win_str:<12} | {r['pixel_rho_mean']:>+12.3f} | {r['pixel_rho_median']:>+8.3f} | {r['pixel_pos_frac']:>5.1f}% | {r['config_rho']:>+13.3f}")
    print("=" * 85)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Transition curves
    ax = axes[0]
    ks = [r['k'] for r in results_table]
    pixel_rhos = [r['pixel_rho_mean'] for r in results_table]
    pixel_medians = [r['pixel_rho_median'] for r in results_table]
    config_rhos = [r['config_rho'] for r in results_table]
    
    ax.plot(ks, pixel_rhos, 'o-', color='blue', linewidth=2, markersize=8, label='Per-pixel (mean)')
    ax.plot(ks, pixel_medians, 's--', color='cyan', linewidth=2, markersize=6, label='Per-pixel (median)')
    ax.plot(ks, config_rhos, 'D-', color='red', linewidth=2, markersize=8, label='Per-config')
    
    ax.axhline(0, color='black', linestyle=':', alpha=0.5)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Window multiplier k', fontsize=12)
    ax.set_ylabel('Spearman ρ with EPE', fontsize=12)
    ax.set_title('Correlation vs Integration Window Scale', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.1)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    
    # Right: Histogram at selected k values
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_table)))
    
    for i, r in enumerate(results_table):
        ax.hist(r['pixel_corrs'], bins=50, range=(-1, 1), 
                alpha=0.4, color=colors[i], label=f"k={r['k']}")
    
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('ρ(energy, EPE) per pixel', fontsize=12)
    ax.set_ylabel('Pixel count', fontsize=12)
    ax.set_title('Pixel-wise Correlation Distributions', fontsize=14)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(-1, 1)
    
    plt.tight_layout()
    
    output_path = results_path.parent / 'energy_window_scaling.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved figure to {output_path}")


if __name__ == "__main__":
    main()
