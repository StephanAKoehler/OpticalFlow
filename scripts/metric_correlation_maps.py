# File: scripts/metric_correlation_maps.py
"""
Per-pixel correlation maps between self-supervised metrics and EPE.

For each pixel, compute Spearman correlation across configs.
Shows WHERE each metric is predictive vs anti-predictive.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def load_results(pkl_path: Path) -> list[dict]:
    """Load results from pickle file."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_ground_truth(frames_dir: Path, pair_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Load ground truth flow for a specific pair."""
    u_file = frames_dir / f"u_{pair_idx:03d}.npz"
    v_file = frames_dir / f"v_{pair_idx:03d}.npz"
    
    if not u_file.exists() or not v_file.exists():
        print(f"❌ Ground truth not found: {u_file}")
        sys.exit(1)
    
    u_data = np.load(u_file)
    v_data = np.load(v_file)
    
    u_truth = u_data[list(u_data.keys())[0]]
    v_truth = v_data[list(v_data.keys())[0]]
    
    return u_truth, v_truth


def build_epe_stack(results: list[dict], u_truth: np.ndarray, v_truth: np.ndarray) -> np.ndarray:
    """
    Build EPE stack from flow results and ground truth.
    
    Returns:
        epe_stack: (n_configs, H, W)
    """
    n_configs = len(results)
    H, W = u_truth.shape
    epe_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    for i, cfg in enumerate(results):
        u = cfg["flows"]["u_AB"]
        v = cfg["flows"]["v_AB"]
        epe_stack[i] = np.sqrt((u - u_truth)**2 + (v - v_truth)**2)
    
    return epe_stack


def build_metric_stack(results: list[dict], metric_key_a: str, metric_key_b: str) -> np.ndarray:
    """
    Build metric stack, averaging A and B directions.
    
    Returns:
        metric_stack: (n_configs, H, W)
    """
    n_configs = len(results)
    first_metric = results[0]["metrics"].get(metric_key_a)
    if first_metric is None:
        print(f"❌ Metric {metric_key_a} not found")
        sys.exit(1)
    
    H, W = first_metric.shape
    metric_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    for i, cfg in enumerate(results):
        val_a = cfg["metrics"].get(metric_key_a)
        val_b = cfg["metrics"].get(metric_key_b)
        
        if val_a is not None and val_b is not None:
            metric_stack[i] = (val_a + val_b) / 2
        elif val_a is not None:
            metric_stack[i] = val_a
        elif val_b is not None:
            metric_stack[i] = val_b
    
    return metric_stack


def compute_correlation_map(metric_stack: np.ndarray, epe_stack: np.ndarray, 
                            valid_mask: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel Spearman correlation between metric and EPE across configs.
    
    Vectorized implementation using rank-based correlation.
    
    Args:
        metric_stack: (n_configs, H, W)
        epe_stack: (n_configs, H, W)
        valid_mask: (H, W) boolean
        
    Returns:
        corr_map: (H, W) correlation values, NaN for invalid pixels
    """
    from scipy.stats import rankdata
    
    n_configs, H, W = metric_stack.shape
    
    if n_configs < 3:
        print("❌ Need at least 3 configs for correlation")
        sys.exit(1)
    
    print(f"   Computing correlations (vectorized) for {valid_mask.sum()} valid pixels...")
    
    # Reshape to (n_configs, n_pixels) for vectorized ranking
    metric_flat = metric_stack.reshape(n_configs, -1)
    epe_flat = epe_stack.reshape(n_configs, -1)
    
    # Rank along config axis (axis=0)
    metric_ranks = rankdata(metric_flat, axis=0)
    epe_ranks = rankdata(epe_flat, axis=0)
    
    # Compute Pearson correlation on ranks = Spearman correlation
    # ρ = Σ[(x - x̄)(y - ȳ)] / √[Σ(x - x̄)² Σ(y - ȳ)²]
    
    metric_centered = metric_ranks - metric_ranks.mean(axis=0, keepdims=True)
    epe_centered = epe_ranks - epe_ranks.mean(axis=0, keepdims=True)
    
    numerator = (metric_centered * epe_centered).sum(axis=0)
    denom_metric = np.sqrt((metric_centered ** 2).sum(axis=0))
    denom_epe = np.sqrt((epe_centered ** 2).sum(axis=0))
    
    # Avoid division by zero
    denom = denom_metric * denom_epe
    denom[denom < 1e-10] = np.nan
    
    corr_flat = numerator / denom
    
    # Reshape back to (H, W)
    corr_map = corr_flat.reshape(H, W)
    
    # Mask invalid pixels
    corr_map[~valid_mask] = np.nan
    
    return corr_map


def plot_correlation_maps(corr_maps: dict[str, np.ndarray], 
                          valid_mask: np.ndarray,
                          output_path: Path,
                          filter_label: str = "all configs"):
    """
    Plot 2x2 grid of correlation maps with shared colorbar.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    metrics = list(corr_maps.keys())
    
    # Shared colormap settings
    vmin, vmax = -1, 1
    cmap = "RdBu_r"
    
    for ax, metric_name in zip(axes, metrics):
        corr_map = corr_maps[metric_name]
        
        # Mask invalid pixels
        display_map = np.where(valid_mask, corr_map, np.nan)
        
        im = ax.imshow(display_map, cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Stats
        valid_corrs = corr_map[valid_mask & ~np.isnan(corr_map)]
        mean_corr = np.nanmean(valid_corrs)
        pct_positive = 100 * np.sum(valid_corrs > 0) / len(valid_corrs)
        
        ax.set_title(f"{metric_name}\nmean ρ = {mean_corr:.2f}, {pct_positive:.0f}% positive")
        ax.axis("off")
    
    # Shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Spearman ρ (metric vs EPE)", fontsize=12)
    
    fig.suptitle(f"Per-Pixel Correlation: Metric vs EPE\n({filter_label})", fontsize=14, fontweight="bold")
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved correlation maps to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Per-pixel correlation maps between self-supervised metrics and EPE"
    )
    parser.add_argument("pkl_path", type=Path, help="Path to results_full.pkl")
    parser.add_argument(
        "--epe-percentile", type=float, default=100.0,
        help="Keep only configs in lowest X%% by mean EPE (default: 100 = all configs)"
    )
    args = parser.parse_args()
    
    pkl_path = args.pkl_path
    epe_percentile = args.epe_percentile
    
    if not pkl_path.exists():
        print(f"❌ File not found: {pkl_path}")
        sys.exit(1)
    
    if not 0 < epe_percentile <= 100:
        print(f"❌ epe-percentile must be in (0, 100], got {epe_percentile}")
        sys.exit(1)
    
    # Derive paths
    pair_dir = pkl_path.parent
    pair_name = pair_dir.name  # e.g., "pair_000"
    pair_idx = int(pair_name.split("_")[1])
    
    sweep_dir = pair_dir.parent
    of_hash_dir = sweep_dir.parent
    analysis_dir = of_hash_dir.parent
    data_hash_dir = analysis_dir.parent
    frames_dir = data_hash_dir / "frames"
    
    # Load data
    print(f"📂 Loading {pkl_path}")
    results = load_results(pkl_path)
    n_total = len(results)
    print(f"   {n_total} configurations")
    
    print(f"📂 Loading ground truth from {frames_dir}")
    u_truth, v_truth = load_ground_truth(frames_dir, pair_idx)
    print(f"   Shape: {u_truth.shape}")
    
    valid_mask = ~np.isnan(u_truth)
    print(f"   Valid pixels: {valid_mask.sum()}/{valid_mask.size}")
    
    # Build EPE stack (all configs first)
    print("📊 Building EPE stack...")
    epe_stack = build_epe_stack(results, u_truth, v_truth)
    
    # Filter by EPE percentile
    if epe_percentile < 100:
        # Compute mean EPE per config (over valid pixels only)
        mean_epe_per_config = np.array([
            np.nanmean(epe_stack[i][valid_mask]) for i in range(n_total)
        ])
        
        # Find threshold
        threshold = np.percentile(mean_epe_per_config, epe_percentile)
        keep_mask = mean_epe_per_config <= threshold
        keep_indices = np.where(keep_mask)[0]
        
        n_keep = len(keep_indices)
        print(f"🔍 EPE filtering: keeping {n_keep}/{n_total} configs (lowest {epe_percentile}%)")
        print(f"   EPE threshold: {threshold:.4f}")
        print(f"   EPE range kept: [{mean_epe_per_config[keep_mask].min():.4f}, {mean_epe_per_config[keep_mask].max():.4f}]")
        
        if n_keep < 3:
            print(f"❌ Need at least 3 configs, got {n_keep}")
            sys.exit(1)
        
        # Filter
        results = [results[i] for i in keep_indices]
        epe_stack = epe_stack[keep_indices]
        
        filter_label = f"top {epe_percentile:.0f}% ({n_keep} configs)"
    else:
        filter_label = f"all {n_total} configs"
    
    # Metrics to analyze
    metrics_to_plot = {
        "photometric": ("photometric_A", "photometric_B"),
        "traction": ("traction_A", "traction_B"),
        "consistency": ("consistency_A", "consistency_B"),
        "perturbation": ("displacements_sensitivity_A2B", "displacements_sensitivity_B2A"),
    }
    
    # Compute correlation maps
    corr_maps = {}
    for metric_name, (key_a, key_b) in metrics_to_plot.items():
        print(f"📊 Processing {metric_name}...")
        metric_stack = build_metric_stack(results, key_a, key_b)
        corr_map = compute_correlation_map(metric_stack, epe_stack, valid_mask)
        corr_maps[metric_name] = corr_map
    
    # Output filename includes percentile
    if epe_percentile < 100:
        output_path = pkl_path.parent / f"metric_correlation_maps_top{epe_percentile:.0f}pct.png"
    else:
        output_path = pkl_path.parent / "metric_correlation_maps.png"
    
    plot_correlation_maps(corr_maps, valid_mask, output_path, filter_label)
    
    # Print summary
    print("\n" + "="*50)
    print(f"SUMMARY ({filter_label})")
    print("="*50)
    for metric_name, corr_map in corr_maps.items():
        valid_corrs = corr_map[valid_mask & ~np.isnan(corr_map)]
        mean_corr = np.nanmean(valid_corrs)
        pct_pos = 100 * np.sum(valid_corrs > 0) / len(valid_corrs)
        pct_neg = 100 * np.sum(valid_corrs < 0) / len(valid_corrs)
        print(f"  {metric_name:15s}  mean ρ={mean_corr:+.3f}  (+:{pct_pos:.0f}% / -:{pct_neg:.0f}%)")


if __name__ == "__main__":
    main()
