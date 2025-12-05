# File: scripts/metric_vs_epe_scatter.py
"""
Scatter plot of mean EPE vs mean metric error for each config.

Shows whether self-supervised metrics predict actual accuracy.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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


def compute_config_means(results: list[dict], u_truth: np.ndarray, v_truth: np.ndarray,
                         valid_mask: np.ndarray, use_measured_margin: bool = True) -> pd.DataFrame:
    """
    Compute mean EPE and mean metrics for each config.
    
    Returns DataFrame with EPE, metrics, config parameters, and normalized metrics.
    """
    # Try to import contamination module for measured margins
    margin_lookup = None
    if use_measured_margin:
        try:
            from src_contamination import get_margin
            margin_lookup = get_margin
            print("   Using measured pollution depths from contamination cache")
        except ImportError:
            print("   ⚠️  src_contamination not found, using theoretical winSize/2")
    
    rows = []
    
    # Metrics (average A and B directions)
    metrics_config = {
        "photometric": ("photometric_A", "photometric_B"),
        "traction": ("traction_A", "traction_B"),
        "consistency": ("consistency_A", "consistency_B"),
        "perturbation": ("displacements_sensitivity_A2B", "displacements_sensitivity_B2A"),
    }
    
    for cfg in results:
        row = {}
        
        # EPE
        u = cfg["flows"]["u_AB"]
        v = cfg["flows"]["v_AB"]
        epe = np.sqrt((u - u_truth)**2 + (v - v_truth)**2)
        row["epe"] = np.nanmean(epe[valid_mask])
        
        # Metrics
        for metric_name, (key_a, key_b) in metrics_config.items():
            val_a = cfg["metrics"].get(key_a)
            val_b = cfg["metrics"].get(key_b)
            
            if val_a is not None and val_b is not None:
                combined = (val_a + val_b) / 2
            elif val_a is not None:
                combined = val_a
            elif val_b is not None:
                combined = val_b
            else:
                combined = np.full_like(u_truth, np.nan)
            
            row[metric_name] = np.nanmean(combined[valid_mask])
        
        # Config parameters
        params = cfg.get("params", cfg.get("config", {}))
        winsize = params.get("winsize", params.get("winSize", 15))
        poly_n = params.get("poly_n", params.get("polyN", 5))
        poly_sigma = params.get("poly_sigma", params.get("polySigma", 1.1))
        
        row["winSize"] = winsize
        row["polyN"] = poly_n
        row["polySigma"] = poly_sigma
        
        # Get pollution depth (measured or theoretical)
        if margin_lookup is not None:
            try:
                contam_config = {
                    "algorithm": "farneback",
                    "winsize": winsize,
                    "poly_n": poly_n,
                    "poly_sigma": poly_sigma,
                    "pyr_scale": params.get("pyr_scale", 0.5),
                    "levels": params.get("levels", 3),
                    "iterations": params.get("iterations", 3),
                    "flags": params.get("flags", 0),
                }
                margin = margin_lookup(contam_config, magnitude=1.0)
                if margin is None:
                    margin = winsize / 2  # Fallback
            except Exception:
                margin = winsize / 2
        else:
            margin = winsize / 2  # Theoretical fallback
        
        row["pollution_depth"] = margin
        
        # Normalized metrics (multiply by pollution depth)
        row["traction_norm"] = row["traction"] * margin
        row["consistency_norm"] = row["consistency"] * margin
        row["perturbation_norm"] = row["perturbation"] * margin
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_scatter(df: pd.DataFrame, output_path: Path, normalize: bool = False):
    """
    Create 2x2 scatter plot of EPE vs each metric.
    
    Visual encodings:
    - Color: winSize
    - Marker: polyN
    - Size: polySigma
    """
    import seaborn as sns
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()
    
    n_configs = len(df)
    
    if normalize:
        metrics = ["photometric", "traction_norm", "consistency_norm", "perturbation_norm"]
        metric_labels = ["photometric", "traction × depth", "consistency × depth", "perturbation × depth"]
    else:
        metrics = ["photometric", "traction", "consistency", "perturbation"]
        metric_labels = metrics
    
    # Marker mapping for polyN
    polyn_values = sorted(df["polyN"].unique())
    markers = ["o", "s", "D", "^", "v", "p", "h"][:len(polyn_values)]
    marker_map = dict(zip(polyn_values, markers))
    
    for ax, metric_name, label in zip(axes, metrics, metric_labels):
        # Spearman correlation
        rho_all, _ = spearmanr(df[metric_name], df["epe"])
        
        sns.scatterplot(
            data=df,
            x=metric_name,
            y="epe",
            hue="winSize",
            style="polyN",
            size="polySigma",
            sizes=(50, 200),
            markers=marker_map,
            palette="viridis",
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            ax=ax,
            legend=(metric_name == metrics[0])  # Only show legend on first plot
        )
        
        # Annotation box
        textstr = f"ρ = {rho_all:.2f}"
        props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray")
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment="top", bbox=props)
        
        ax.set_xlabel(f"Mean {label}", fontsize=11)
        ax.set_ylabel("Mean EPE", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
    
    # Move legend outside
    axes[0].legend(
        bbox_to_anchor=(1.02, 1), 
        loc="upper left", 
        fontsize=9,
        title_fontsize=10
    )
    
    norm_label = "normalized by pollution depth" if normalize else "raw metrics"
    fig.suptitle(f"Config Selection: Mean EPE vs Mean Metric\n({n_configs} configs, {norm_label})", 
                 fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved scatter plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Scatter plot of mean EPE vs mean metric error per config"
    )
    parser.add_argument("pkl_path", type=Path, help="Path to results_full.pkl")
    parser.add_argument(
        "--normalize", action="store_true",
        help="Normalize traction/consistency/perturbation by pollution depth"
    )
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
    
    # Load data
    print(f"📂 Loading {pkl_path}")
    results = load_results(pkl_path)
    print(f"   {len(results)} configurations")
    
    print(f"📂 Loading ground truth from {frames_dir}")
    u_truth, v_truth = load_ground_truth(frames_dir, pair_idx)
    print(f"   Shape: {u_truth.shape}")
    
    valid_mask = ~np.isnan(u_truth)
    print(f"   Valid pixels: {valid_mask.sum()}/{valid_mask.size}")
    
    # Compute means -> DataFrame
    print("📊 Computing config means...")
    df = compute_config_means(results, u_truth, v_truth, valid_mask, 
                              use_measured_margin=args.normalize)
    
    # Print param ranges
    print(f"\n📋 Config parameters:")
    print(f"   winSize: {sorted(df['winSize'].unique())}")
    print(f"   polyN: {sorted(df['polyN'].unique())}")
    print(f"   polySigma: {sorted(df['polySigma'].unique())}")
    
    if args.normalize:
        print(f"   pollution_depth: {sorted(df['pollution_depth'].unique())}")
    
    # Plot
    suffix = "_normalized" if args.normalize else ""
    output_path = pkl_path.parent / f"metric_vs_epe_scatter{suffix}.png"
    plot_scatter(df, output_path, normalize=args.normalize)
    
    # Print summary
    print("\n" + "="*50)
    print("CORRELATION SUMMARY")
    print("="*50)
    
    if args.normalize:
        metrics = [("photometric", "photometric"), 
                   ("traction_norm", "traction×depth"),
                   ("consistency_norm", "consistency×depth"),
                   ("perturbation_norm", "perturbation×depth")]
    else:
        metrics = [(m, m) for m in ["photometric", "traction", "consistency", "perturbation"]]
    
    for col, label in metrics:
        rho, _ = spearmanr(df[col], df["epe"])
        print(f"  {label:20s}  ρ={rho:+.3f}")


if __name__ == "__main__":
    main()
