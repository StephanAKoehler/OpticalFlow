# File: scripts/normalization_comparison.py
"""
Compare three normalization approaches for perturbation-based metrics.

1. bounded: raw / hypot(raw, scale)  [current, dimensionless]
2. bounded * depth  [what worked, pixels]
3. raw * (depth + pert_dist)  [proposed, pixels²]
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


def load_results(pkl_path: Path) -> list[dict]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_ground_truth(frames_dir: Path, pair_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
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


def invert_bounded(bounded: float, scale: float) -> float:
    """
    Recover raw from bounded normalization.
    
    bounded = raw / hypot(raw, scale)
    => raw = bounded * scale / sqrt(1 - bounded²)
    """
    bounded = np.clip(bounded, 0, 0.9999)  # Avoid division by zero
    return bounded * scale / np.sqrt(1 - bounded**2)


def compute_normalizations(results: list[dict], u_truth: np.ndarray, v_truth: np.ndarray,
                           valid_mask: np.ndarray, pert_scale: float = 2.5) -> pd.DataFrame:
    """
    Compute all three normalizations for each metric and config.
    """
    rows = []
    
    metrics_config = {
        "traction": ("traction_A", "traction_B"),
        "consistency": ("consistency_A", "consistency_B"),
        "perturbation": ("displacements_sensitivity_A2B", "displacements_sensitivity_B2A"),
    }
    
    for cfg in results:
        # EPE
        u = cfg["flows"]["u_AB"]
        v = cfg["flows"]["v_AB"]
        epe = np.sqrt((u - u_truth)**2 + (v - v_truth)**2)
        mean_epe = np.nanmean(epe[valid_mask])
        
        # Config parameters
        params = cfg.get("params", cfg.get("config", {}))
        winsize = params.get("winsize", params.get("winSize", 15))
        
        # Pollution depth (theoretical for now)
        depth = winsize / 2
        
        for metric_name, (key_a, key_b) in metrics_config.items():
            val_a = cfg["metrics"].get(key_a)
            val_b = cfg["metrics"].get(key_b)
            
            if val_a is not None and val_b is not None:
                bounded_map = (val_a + val_b) / 2
            elif val_a is not None:
                bounded_map = val_a
            else:
                continue
            
            bounded = np.nanmean(bounded_map[valid_mask])
            
            # Three normalizations
            # 1. Current (bounded, dimensionless)
            norm1 = bounded
            
            # 2. bounded * depth (pixels)
            norm2 = bounded * depth
            
            # 3. raw * (depth + pert_dist) (pixels²)
            raw = invert_bounded(bounded, pert_scale)
            norm3 = raw * (depth + pert_scale)
            
            rows.append({
                "metric": metric_name,
                "epe": mean_epe,
                "winSize": winsize,
                "bounded": norm1,
                "bounded×depth": norm2,
                "raw×(depth+pert)": norm3,
            })
    
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, output_path: Path, pert_scale: float):
    """
    Plot all three normalizations on same axes, one subplot per metric.
    """
    metrics = ["traction", "consistency", "perturbation"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    norm_cols = ["bounded", "bounded×depth", "raw×(depth+pert)"]
    markers = ["o", "s", "D"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for ax, metric_name in zip(axes, metrics):
        metric_df = df[df["metric"] == metric_name]
        
        for norm_col, marker, color in zip(norm_cols, markers, colors):
            rho, _ = spearmanr(metric_df[norm_col], metric_df["epe"])
            
            ax.scatter(
                metric_df[norm_col], 
                metric_df["epe"],
                c=metric_df["winSize"],
                cmap="viridis",
                marker=marker,
                s=80,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
                label=f"{norm_col} (ρ={rho:.2f})"
            )
        
        ax.set_xlabel(f"Normalized {metric_name}", fontsize=11)
        ax.set_ylabel("Mean EPE", fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Normalization Comparison (pert_scale={pert_scale:.1f}px)\nColor = winSize", 
                 fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare normalization approaches")
    parser.add_argument("pkl_path", type=Path, help="Path to results_full.pkl")
    parser.add_argument("--pert-scale", type=float, default=2.5,
                        help="Perturbation scale used in bounded normalization (default: 2.5)")
    args = parser.parse_args()
    
    pkl_path = args.pkl_path
    if not pkl_path.exists():
        print(f"❌ File not found: {pkl_path}")
        sys.exit(1)
    
    # Derive paths
    pair_dir = pkl_path.parent
    pair_name = pair_dir.name
    pair_idx = int(pair_name.split("_")[1])
    
    sweep_dir = pair_dir.parent
    of_hash_dir = sweep_dir.parent
    analysis_dir = of_hash_dir.parent
    data_hash_dir = analysis_dir.parent
    frames_dir = data_hash_dir / "frames"
    
    # Load
    print(f"📂 Loading {pkl_path}")
    results = load_results(pkl_path)
    print(f"   {len(results)} configurations")
    
    print(f"📂 Loading ground truth from {frames_dir}")
    u_truth, v_truth = load_ground_truth(frames_dir, pair_idx)
    
    valid_mask = ~np.isnan(u_truth)
    
    # Compute
    print(f"📊 Computing normalizations (pert_scale={args.pert_scale})...")
    df = compute_normalizations(results, u_truth, v_truth, valid_mask, args.pert_scale)
    
    # Plot
    output_path = pkl_path.parent / "normalization_comparison.png"
    plot_comparison(df, output_path, args.pert_scale)
    
    # Summary
    print("\n" + "="*60)
    print("CORRELATION SUMMARY")
    print("="*60)
    
    for metric_name in ["traction", "consistency", "perturbation"]:
        metric_df = df[df["metric"] == metric_name]
        print(f"\n{metric_name}:")
        for norm_col in ["bounded", "bounded×depth", "raw×(depth+pert)"]:
            rho, _ = spearmanr(metric_df[norm_col], metric_df["epe"])
            print(f"  {norm_col:20s}  ρ={rho:+.3f}")


if __name__ == "__main__":
    main()
