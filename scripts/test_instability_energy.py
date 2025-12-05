# File: scripts/test_instability_energy.py
"""
Test instability energy approach for pixel-wise config selection.

Key idea:
    energy(x,y) = ∫∫ (pert_raw)² over winSize×winSize
                = total variance (instability) the algorithm "sees"

Each config integrates over ITS OWN window size.
This should naturally balance small-window (high density, small area) 
vs large-window (low density, large area) configs.

Usage:
    python scripts/test_instability_energy.py data/.../results_full.pkl
"""

import numpy as np
import pickle
import sys
import re
import json
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


def compute_instability_energy(pert_raw: np.ndarray, winsize: int) -> np.ndarray:
    """
    Compute total instability energy over config's receptive field.
    
    energy(x,y) = ∫∫ pert_raw² over winSize×winSize
    
    No normalization - total energy naturally scales with window area.
    """
    variance = pert_raw.astype(np.float32) ** 2
    
    # Sum over window (not mean!)
    energy = cv2.boxFilter(
        variance,
        ddepth=-1,
        ksize=(winsize, winsize),
        normalize=False,
        borderType=cv2.BORDER_REFLECT
    )
    
    return energy


def compute_pixelwise_correlation(metric_stack, epe_stack, valid_mask):
    """Compute Spearman correlation at each pixel."""
    n_configs, H, W = metric_stack.shape
    corr_map = np.full((H, W), np.nan, dtype=np.float32)
    
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
    
    corr_map[valid_ys, valid_xs] = correlations
    return corr_map


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_instability_energy.py <results_full.pkl>")
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
    
    # Extract window sizes and pert_raw for each config
    print("\n📊 Extracting perturbation data per config...")
    config_data = []
    for i, r in enumerate(results):
        config_name = r['metadata'].get('config_name', '')
        match = re.search(r'win(\d+)', config_name)
        winsize = int(match.group(1)) if match else 15
        pert_raw = r['metrics']['perturbation_raw_A']
        
        config_data.append({
            'idx': i,
            'name': config_name,
            'winsize': winsize,
            'pert_raw': pert_raw,
            'mean_epe': np.nanmean(epe_stack[i][valid_mask])
        })
        
    # Show config summary
    print("\n   Config summary (first 5):")
    for c in config_data[:5]:
        pr = c['pert_raw']
        print(f"      {c['name'][:30]}: win={c['winsize']}, "
              f"pert_raw: mean={np.nanmean(pr):.4f}, var={np.nanvar(pr):.6f}")
    
    # ========================================================================
    # Build metric stacks for different approaches
    # ========================================================================
    print("\n📊 Computing metric stacks...")
    
    # 1. Raw pert (baseline)
    raw_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    # 2. Scaled by depth (current approach)
    scaled_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    # 3. Energy: ∫∫ pert² over config's own window
    energy_own_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    # 4. Energy: ∫∫ pert² over 2× config's window  
    energy_2x_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    # 5. Mean variance over config's window (density, not total)
    var_density_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    # 6. Mean variance × depth² (to match per-config formula)
    var_scaled_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    for c in config_data:
        i = c['idx']
        winsize = c['winsize']
        pert_raw = c['pert_raw']
        depth = winsize / 2 + 1.0  # assuming pert_dist=1.0
        
        variance = pert_raw ** 2
        
        # 1. Raw
        raw_stack[i] = pert_raw
        
        # 2. Scaled
        scaled_stack[i] = pert_raw * depth
        
        # 3. Energy over own window (total variance)
        energy_own_stack[i] = compute_instability_energy(pert_raw, winsize)
        
        # 4. Energy over 2× window
        energy_2x_stack[i] = compute_instability_energy(pert_raw, winsize * 2)
        
        # 5. Variance density (mean over window)
        var_density_stack[i] = cv2.boxFilter(
            variance.astype(np.float32), -1, (winsize, winsize),
            normalize=True, borderType=cv2.BORDER_REFLECT
        )
        
        # 6. Variance density × depth²
        var_scaled_stack[i] = var_density_stack[i] * (depth ** 2)
    
    # ========================================================================
    # Compute correlations
    # ========================================================================
    print("\n📊 Computing pixel-wise correlations...")
    
    methods = {
        'raw (no correction)': raw_stack,
        'scaled (×depth)': scaled_stack,
        'energy (∫pert² own win)': energy_own_stack,
        'energy (∫pert² 2×win)': energy_2x_stack,
        'var density (mean pert²)': var_density_stack,
        'var×depth² (density scaled)': var_scaled_stack,
    }
    
    results_table = {}
    for name, stack in methods.items():
        print(f"   {name}...")
        corr_map = compute_pixelwise_correlation(stack, epe_stack, valid_mask)
        valid_corrs = corr_map[valid_mask & ~np.isnan(corr_map)]
        results_table[name] = {
            'corr_map': corr_map,
            'mean': np.mean(valid_corrs),
            'median': np.median(valid_corrs),
            'pos_frac': np.mean(valid_corrs > 0) * 100,
            'neg_frac': np.mean(valid_corrs < 0) * 100,
            'corrs': valid_corrs
        }
    
    # ========================================================================
    # Check if energy changes config rankings
    # ========================================================================
    print("\n📊 DEBUG: Do different metrics change config rankings?")
    
    # Sample pixels
    np.random.seed(42)
    valid_idx = np.where(valid_mask.ravel())[0]
    sample_idx = np.random.choice(valid_idx, min(5000, len(valid_idx)), replace=False)
    sample_ys = sample_idx // W
    sample_xs = sample_idx % W
    
    # Compare rankings between raw and energy
    ranking_changes = 0
    total_compared = 0
    
    for sy, sx in zip(sample_ys[:1000], sample_xs[:1000]):
        raw_vals = raw_stack[:, sy, sx]
        energy_vals = energy_own_stack[:, sy, sx]
        
        raw_order = np.argsort(raw_vals)
        energy_order = np.argsort(energy_vals)
        
        if not np.array_equal(raw_order, energy_order):
            ranking_changes += 1
        total_compared += 1
    
    print(f"   Rankings changed at {ranking_changes}/{total_compared} pixels ({100*ranking_changes/total_compared:.1f}%)")
    
    # Correlation between raw and energy rankings
    raw_sample = raw_stack[:, sample_ys, sample_xs]
    energy_sample = energy_own_stack[:, sample_ys, sample_xs]
    
    rank_corrs = []
    for i in range(len(sample_ys)):
        rho, _ = scipy_stats.spearmanr(raw_sample[:, i], energy_sample[:, i])
        rank_corrs.append(rho)
    
    print(f"   Rank correlation (raw vs energy): mean ρ = {np.nanmean(rank_corrs):.4f}")
    
    # ========================================================================
    # Print summary
    # ========================================================================
    print("\n" + "="*85)
    print("PIXEL-WISE CORRELATION SUMMARY (metric vs EPE)")
    print("="*85)
    print(f"{'Method':<30} | {'mean ρ':>8} | {'median ρ':>9} | {'+ %':>6} | {'- %':>6}")
    print("-"*85)
    for name, stats in results_table.items():
        print(f"{name:<30} | {stats['mean']:>+8.3f} | {stats['median']:>+9.3f} | "
              f"{stats['pos_frac']:>5.1f}% | {stats['neg_frac']:>5.1f}%")
    print("="*85)
    
    # ========================================================================
    # Create visualization
    # ========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, (name, stats) in enumerate(results_table.items()):
        ax = axes.flat[idx]
        ax.hist(stats['corrs'], bins=50, range=(-1, 1), 
                color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.axvline(stats['mean'], color='red', linestyle='-', linewidth=2)
        ax.set_xlabel('ρ(metric, EPE)')
        ax.set_ylabel('Pixel count')
        ax.set_title(f"{name}\nmean ρ={stats['mean']:.3f}, +:{stats['pos_frac']:.0f}% / -:{stats['neg_frac']:.0f}%")
        ax.set_xlim(-1, 1)
    
    plt.suptitle('Instability Energy: Pixel-wise Spearman correlation vs EPE', fontsize=14)
    plt.tight_layout()
    
    output_path = results_path.parent / 'instability_energy_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved histogram to {output_path}")
    
    # Heatmaps
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, (name, stats) in enumerate(results_table.items()):
        ax = axes2.flat[idx]
        display_map = stats['corr_map'].copy()
        display_map[~valid_mask] = np.nan
        
        im = ax.imshow(display_map, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f"{name}\nmean ρ={stats['mean']:.3f}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Instability Energy: Pixel-wise correlation heatmaps', fontsize=14)
    plt.tight_layout()
    
    output_path2 = results_path.parent / 'instability_energy_heatmaps.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved heatmaps to {output_path2}")


if __name__ == "__main__":
    main()
