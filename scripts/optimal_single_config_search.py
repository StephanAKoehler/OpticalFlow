# File: scripts/optimal_single_config_search.py
"""
Find optimal single config using self-supervised metrics.

Methods:
  1. pert×depth: mean(perturbation_raw_A) × (winSize/2 + pert_dist)
  2. photo_log: mean(photo_log_raw_A)
  3. two-stage: top-K by pert×depth, then best by photo_log

Usage:
    python scripts/optimal_single_config_search.py data/.../results_full.pkl [--k 5]
"""

import numpy as np
import pickle
import sys
import re
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats


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


def extract_config_metrics(results, u_truth, v_truth, valid_mask, pert_dist=1.0):
    """Extract all metrics for each config."""
    configs = []
    
    for i, r in enumerate(results):
        config_name = r['metadata'].get('config_name', f'config_{i}')
        
        # Parse winSize from config name
        match = re.search(r'win(\d+)', config_name)
        winsize = int(match.group(1)) if match else 15
        depth_scale = winsize / 2 + pert_dist
        
        # Perturbation score (depth-scaled)
        pert_raw = np.nanmean(r['metrics']['perturbation_raw_A'])
        pert_score = depth_scale * pert_raw
        
        # Photometric log
        photo_log = np.nanmean(r['metrics']['photo_log_raw_A'])
        
        # EPE (ground truth)
        u_est = r['flows']['u_AB']
        v_est = r['flows']['v_AB']
        epe = np.sqrt((u_est - u_truth)**2 + (v_est - v_truth)**2)
        mean_epe = np.nanmean(epe[valid_mask])
        
        configs.append({
            'idx': i,
            'name': config_name,
            'winsize': winsize,
            'depth_scale': depth_scale,
            'pert_raw': pert_raw,
            'pert_score': pert_score,
            'photo_log': photo_log,
            'epe': mean_epe,
        })
    
    return configs


def select_configs(configs, K=5):
    """Select configs using different methods."""
    # Oracle (best EPE)
    oracle = min(configs, key=lambda x: x['epe'])
    
    # By pert×depth
    by_pert = min(configs, key=lambda x: x['pert_score'])
    
    # By photo_log
    by_photo = min(configs, key=lambda x: x['photo_log'])
    
    # Two-stage: top-K by pert×depth, then best by photo_log
    top_k_by_pert = sorted(configs, key=lambda x: x['pert_score'])[:K]
    two_stage = min(top_k_by_pert, key=lambda x: x['photo_log'])
    
    return {
        'oracle': oracle,
        'pert': by_pert,
        'photo': by_photo,
        'two_stage': two_stage,
        'top_k_by_pert': top_k_by_pert,
    }


def create_scatter_plots(configs, selections, K, output_path):
    """Create scatter plots of metrics vs EPE."""
    # Extract arrays
    pert_scores = np.array([c['pert_score'] for c in configs])
    photo_logs = np.array([c['photo_log'] for c in configs])
    epes = np.array([c['epe'] for c in configs])
    winsizes = np.array([c['winsize'] for c in configs])
    
    # Unique winsizes for coloring
    unique_wins = sorted(set(winsizes))
    cmap = plt.cm.viridis
    colors = {w: cmap(i / (len(unique_wins) - 1)) if len(unique_wins) > 1 else cmap(0.5) 
              for i, w in enumerate(unique_wins)}
    point_colors = [colors[w] for w in winsizes]
    
    # Compute correlations
    rho_pert, _ = stats.spearmanr(pert_scores, epes)
    rho_photo, _ = stats.spearmanr(photo_logs, epes)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top-K threshold for shading
    top_k_indices = set(c['idx'] for c in selections['top_k_by_pert'])
    pert_threshold = max(c['pert_score'] for c in selections['top_k_by_pert'])
    
    # === Left panel: pert×depth vs EPE ===
    ax = axes[0]
    
    # Shaded region for top-K
    ax.axvspan(0, pert_threshold, alpha=0.15, color='green', label=f'Top-{K} region')
    
    # Scatter points
    for i, c in enumerate(configs):
        ax.scatter(c['pert_score'], c['epe'], c=colors[c['winsize']], 
                   s=100, edgecolors='black', linewidths=0.5, zorder=2)
    
    # Mark selected configs
    oracle = selections['oracle']
    by_pert = selections['pert']
    ax.scatter(oracle['pert_score'], oracle['epe'], marker='D', s=250, 
               facecolors='none', edgecolors='red', linewidths=3, zorder=4, label='Oracle')
    ax.scatter(by_pert['pert_score'], by_pert['epe'], marker='*', s=400, 
               facecolors='gold', edgecolors='black', linewidths=1, zorder=5, label='Selected')
    
    # Regression line
    z = np.polyfit(pert_scores, epes, 1)
    p = np.poly1d(z)
    x_line = np.linspace(pert_scores.min(), pert_scores.max(), 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('pert×depth (lower = better)', fontsize=12)
    ax.set_ylabel('EPE (lower = better)', fontsize=12)
    ax.set_title(f'Perturbation×Depth vs EPE\nρ = {rho_pert:.3f}', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # === Right panel: photo_log vs EPE ===
    ax = axes[1]
    
    # Highlight top-K configs
    for i, c in enumerate(configs):
        alpha = 1.0 if c['idx'] in top_k_indices else 0.3
        ax.scatter(c['photo_log'], c['epe'], c=colors[c['winsize']], 
                   s=100, edgecolors='black', linewidths=0.5, zorder=2, alpha=alpha)
    
    # Mark selected configs
    by_photo = selections['photo']
    two_stage = selections['two_stage']
    ax.scatter(oracle['photo_log'], oracle['epe'], marker='D', s=250, 
               facecolors='none', edgecolors='red', linewidths=3, zorder=4, label='Oracle')
    ax.scatter(by_photo['photo_log'], by_photo['epe'], marker='*', s=400, 
               facecolors='gold', edgecolors='black', linewidths=1, zorder=5, label='By photo_log')
    ax.scatter(two_stage['photo_log'], two_stage['epe'], marker='P', s=300, 
               facecolors='lime', edgecolors='black', linewidths=1, zorder=5, label=f'Two-stage (K={K})')
    
    # Regression line
    z = np.polyfit(photo_logs, epes, 1)
    p = np.poly1d(z)
    x_line = np.linspace(photo_logs.min(), photo_logs.max(), 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('photo_log (lower = better)', fontsize=12)
    ax.set_ylabel('EPE (lower = better)', fontsize=12)
    ax.set_title(f'Photo Log vs EPE\nρ = {rho_photo:.3f} (faded = outside top-{K})', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Colorbar for winSize
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(unique_wins), vmax=max(unique_wins)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('winSize', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved figure to {output_path}")


def print_summary(configs, selections, K):
    """Print summary table."""
    n_configs = len(configs)
    oracle_epe = selections['oracle']['epe']
    
    print()
    print("=" * 80)
    print(f"SINGLE CONFIG SELECTION ({n_configs} configs, K={K})")
    print("=" * 80)
    print(f"{'Method':<20} | {'EPE':>8} | {'Δ oracle':>10} | Config")
    print("-" * 80)
    
    methods = [
        ('pert×depth', selections['pert']),
        ('photo_log', selections['photo']),
        (f'two-stage (K={K})', selections['two_stage']),
        ('oracle', selections['oracle']),
    ]
    
    for name, sel in methods:
        delta = sel['epe'] - oracle_epe
        delta_str = f"{delta:+.4f}" if name != 'oracle' else '-'
        print(f"{name:<20} | {sel['epe']:>8.4f} | {delta_str:>10} | {sel['name'][:35]}")
    
    print("=" * 80)
    
    # Show top-K
    print(f"\n📋 Top-{K} by pert×depth:")
    for i, c in enumerate(selections['top_k_by_pert']):
        marker = "→" if c['idx'] == selections['two_stage']['idx'] else " "
        print(f"  {marker} {i+1}. EPE={c['epe']:.4f}  pert={c['pert_score']:.4f}  photo={c['photo_log']:.4f}  {c['name'][:30]}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python optimal_single_config_search.py <results_full.pkl> [--k 5]")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    
    # Parse K
    K = 5
    if '--k' in sys.argv:
        k_idx = sys.argv.index('--k')
        K = int(sys.argv[k_idx + 1])
    
    # Load results
    print(f"📂 Loading {results_path}")
    if not results_path.exists():
        print(f"❌ File not found: {results_path}")
        sys.exit(1)
    
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
        print(f"   perturbation_distance = {pert_dist}")
    
    # Load ground truth
    print("📂 Loading ground truth...")
    u_truth, v_truth, valid_mask = load_ground_truth(results_path)
    print(f"   Shape: {u_truth.shape}, valid: {valid_mask.sum()}")
    
    # Extract metrics
    print("📊 Computing metrics...")
    configs = extract_config_metrics(results, u_truth, v_truth, valid_mask, pert_dist)
    
    # Select configs
    selections = select_configs(configs, K)
    
    # Print summary
    print_summary(configs, selections, K)
    
    # Create plots
    output_path = results_path.parent / 'optimal_single_config.png'
    create_scatter_plots(configs, selections, K, output_path)


if __name__ == "__main__":
    main()
