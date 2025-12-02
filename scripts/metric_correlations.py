#!/usr/bin/env python3
# File: scripts/metric_correlations.py
"""
Analyze correlations between uncertainty metrics and EPE.

Produces 4x5 correlation matrix:
- Diagonal (col 0-3): metric → EPE correlation (grayscale)
- Upper triangle (col 0-3): metric → metric correlation (blue-red)
- Column 4: aggregated method → EPE correlation (grayscale)

Usage:
    python scripts/metric_correlations.py config.toml
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tomli

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.data_loader import load_movie_sequence
from src.evaluation.ground_truth import compute_epe


# =============================================================================
# Auto-detection (same as evaluate_weights.py)
# =============================================================================

def auto_detect_experiment(data_dir: Path) -> tuple[str, str]:
    """Auto-detect movie_hash and of_hash from data directory."""
    if not data_dir.exists():
        print(f"❌ ERROR: Data directory not found: {data_dir}")
        sys.exit(1)
    
    movie_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if len(movie_dirs) == 0:
        print(f"❌ ERROR: No experiments found in {data_dir}")
        sys.exit(1)
    elif len(movie_dirs) > 1:
        print(f"❌ ERROR: Multiple movie hashes found in {data_dir}:")
        for d in sorted(movie_dirs):
            print(f"   - {d.name}")
        sys.exit(1)
    
    movie_hash = movie_dirs[0].name
    analysis_dir = movie_dirs[0] / 'analysis'
    
    if not analysis_dir.exists():
        print(f"❌ ERROR: No analysis directory in {movie_dirs[0]}")
        sys.exit(1)
    
    of_dirs = [d for d in analysis_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if len(of_dirs) == 0:
        print(f"❌ ERROR: No OF analysis found in {analysis_dir}")
        sys.exit(1)
    elif len(of_dirs) > 1:
        print(f"❌ ERROR: Multiple OF hashes found in {analysis_dir}:")
        for d in sorted(of_dirs):
            print(f"   - {d.name}")
        sys.exit(1)
    
    return movie_hash, of_dirs[0].name


# =============================================================================
# Correlation Computation
# =============================================================================

def compute_pixel_correlations(results_full: list,
                                u_truth: np.ndarray,
                                v_truth: np.ndarray,
                                valid_mask: np.ndarray,
                                epe_power: float) -> dict:
    """
    Compute Spearman correlations at each pixel across configs.
    
    Returns dict with correlation arrays for each pair.
    """
    n_configs = len(results_full)
    H, W = valid_mask.shape
    
    # Metric keys
    metric_keys = {
        'pert': 'displacements_sensitivity_A2B',
        'cons': 'consistency_A',
        'phot': 'photometric_A',
        'trac': 'traction_A'
    }
    metric_names = ['pert', 'cons', 'phot', 'trac']
    
    # Stack all metrics and compute EPE per config
    metric_stacks = {}
    for name, key in metric_keys.items():
        metric_stacks[name] = np.stack([r['metrics'][key] for r in results_full], axis=0)
    
    # Compute EPE for each config
    epe_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    for i, r in enumerate(results_full):
        u_est = r['flows']['u_AB']
        v_est = r['flows']['v_AB']
        epe_stack[i] = compute_epe(u_est, v_est, u_truth, v_truth, valid_mask, power=epe_power)
    
    # Collect all valid pixel data
    # Shape: (n_valid_pixels, n_configs) for each metric
    valid_indices = np.where(valid_mask)
    n_valid = len(valid_indices[0])
    
    pixel_data = {'epe': epe_stack[:, valid_indices[0], valid_indices[1]].T}  # (n_pixels, n_configs)
    for name in metric_names:
        pixel_data[name] = metric_stacks[name][:, valid_indices[0], valid_indices[1]].T
    
    return pixel_data, n_valid


def vectorized_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute mean Spearman correlation across pixels (rows).
    
    Args:
        x: (n_pixels, n_configs) array
        y: (n_pixels, n_configs) array
    
    Returns:
        Mean Spearman rho across all pixels
    """
    # Rank along config axis (axis=1)
    x_ranks = np.argsort(np.argsort(x, axis=1), axis=1).astype(np.float64)
    y_ranks = np.argsort(np.argsort(y, axis=1), axis=1).astype(np.float64)
    
    # Center ranks
    x_c = x_ranks - x_ranks.mean(axis=1, keepdims=True)
    y_c = y_ranks - y_ranks.mean(axis=1, keepdims=True)
    
    # Pearson on ranks = Spearman
    numer = (x_c * y_c).sum(axis=1)
    denom = np.sqrt((x_c**2).sum(axis=1) * (y_c**2).sum(axis=1))
    
    # Avoid division by zero (constant rows)
    valid = denom > 1e-10
    rho = np.zeros(x.shape[0])
    rho[valid] = numer[valid] / denom[valid]
    rho[~valid] = np.nan
    
    return float(np.nanmean(rho))


def vectorized_spearman_topk(x: np.ndarray, y: np.ndarray, epe: np.ndarray, k: int) -> float:
    """
    Compute mean Spearman correlation across pixels, restricted to top-k configs by oracle EPE.
    
    Args:
        x: (n_pixels, n_configs) array - metric values
        y: (n_pixels, n_configs) array - values to correlate with (usually EPE)
        epe: (n_pixels, n_configs) array - oracle EPE for filtering
        k: number of top configs to keep per pixel
    
    Returns:
        Mean Spearman rho across all pixels (on top-k only)
    """
    n_pixels, n_configs = x.shape
    
    # Get indices of top-k configs per pixel (lowest EPE)
    # argsort gives indices sorted by ascending EPE, take first k
    topk_indices = np.argsort(epe, axis=1)[:, :k]  # (n_pixels, k)
    
    # Gather top-k values for x and y
    rows = np.arange(n_pixels)[:, None]  # (n_pixels, 1)
    x_topk = x[rows, topk_indices]  # (n_pixels, k)
    y_topk = y[rows, topk_indices]  # (n_pixels, k)
    
    # Compute Spearman on filtered data
    return vectorized_spearman(x_topk, y_topk)


def compute_correlation_matrix(pixel_data: dict, 
                                opt_weights: dict = None,
                                max_pixels: int = 10000) -> dict:
    """
    Compute correlation matrices from pooled pixel data.
    
    Args:
        pixel_data: Dict with 'epe', 'pert', 'cons', 'phot', 'trac' arrays
        opt_weights: Dict of optimized weights per method, or None for equal weights only
        max_pixels: Maximum pixels to sample for speed
    
    Returns:
        Dict with 'all' and 'topk' keys, each containing:
            diag_corrs: (4,) array of metric→EPE correlations
            upper_corrs: (4,4) array with upper triangle metric→metric correlations
            agg_eq_corrs: (4,) array of equal-weight aggregated method→EPE correlations
            agg_opt_corrs: (4,) array of optimized-weight aggregated method→EPE correlations
    """
    metric_names = ['pert', 'cons', 'phot', 'trac']
    n_metrics = len(metric_names)
    
    epe = pixel_data['epe']
    n_pixels, n_configs = epe.shape
    
    # Determine top-k: 3 if n_configs <= 30, else 10%
    if n_configs <= 30:
        k = 3
    else:
        k = max(3, n_configs // 10)
    
    # Sample pixels if too many
    if n_pixels > max_pixels:
        rng = np.random.default_rng(42)
        indices = rng.choice(n_pixels, max_pixels, replace=False)
        epe = epe[indices]
        sampled_data = {name: pixel_data[name][indices] for name in metric_names}
    else:
        sampled_data = {name: pixel_data[name] for name in metric_names}
    
    # Get raw metrics
    pert = sampled_data['pert']
    cons = sampled_data['cons']
    phot = sampled_data['phot']
    
    # Compute MAD-normalized versions per pixel
    def mad_normalize(arr):
        """MAD normalize along config axis (axis=1)."""
        median = np.median(arr, axis=1, keepdims=True)
        mad = np.median(np.abs(arr - median), axis=1, keepdims=True)
        mad = np.maximum(mad, 1e-10)
        return (arr - median) / mad
    
    pert_mad = mad_normalize(pert)
    cons_mad = mad_normalize(cons)
    phot_mad = mad_normalize(phot)
    
    # Method names
    method_names = ['raw_sum', 'raw_max', 'mad_sum', 'mad_max']
    
    # Helper to compute penalty with weights
    def compute_penalty(w_pert, w_cons, w_phot, use_mad, use_max):
        if use_mad:
            p, c, ph = pert_mad, cons_mad, phot_mad
        else:
            p, c, ph = pert, cons, phot
        
        if use_max:
            return np.maximum(np.maximum(w_pert * p**2, w_cons * c**2), w_phot * ph**2)
        else:
            return w_pert * p**2 + w_cons * c**2 + w_phot * ph**2
    
    # Equal weights
    eq_weights = {'perturbation_rms': 1.0, 'consistency': 1.0, 'photometric': 1.0}
    
    results = {}
    
    for mode, use_topk in [('all', False), ('topk', True)]:
        # Choose correlation function
        if use_topk:
            def corr_fn(x, y):
                return vectorized_spearman_topk(x, y, epe, k)
        else:
            def corr_fn(x, y):
                return vectorized_spearman(x, y)
        
        # Diagonal: metric → EPE correlations
        diag_corrs = np.zeros(n_metrics)
        for i, name in enumerate(metric_names):
            diag_corrs[i] = corr_fn(sampled_data[name], epe)
        
        # Upper triangle: metric → metric correlations
        upper_corrs = np.full((n_metrics, n_metrics), np.nan)
        for i in range(n_metrics):
            for j in range(i + 1, n_metrics):
                if use_topk:
                    upper_corrs[i, j] = vectorized_spearman_topk(
                        sampled_data[metric_names[i]], 
                        sampled_data[metric_names[j]],
                        epe, k
                    )
                else:
                    upper_corrs[i, j] = vectorized_spearman(
                        sampled_data[metric_names[i]], 
                        sampled_data[metric_names[j]]
                    )
        
        # Equal weights aggregation
        agg_eq_corrs = np.zeros(4)
        eq_penalties = [
            compute_penalty(1, 1, 1, use_mad=False, use_max=False),  # raw_sum
            compute_penalty(1, 1, 1, use_mad=False, use_max=True),   # raw_max
            compute_penalty(1, 1, 1, use_mad=True, use_max=False),   # mad_sum
            compute_penalty(1, 1, 1, use_mad=True, use_max=True),    # mad_max
        ]
        for m, penalty in enumerate(eq_penalties):
            agg_eq_corrs[m] = corr_fn(penalty, epe)
        
        # Optimized weights aggregation
        agg_opt_corrs = np.zeros(4)
        if opt_weights is not None:
            for m, method in enumerate(method_names):
                if method in opt_weights:
                    w = opt_weights[method]
                    w_pert = w.get('perturbation_rms', 1.0)
                    w_cons = w.get('consistency', 1.0)
                    w_phot = w.get('photometric', 1.0)
                    use_mad = 'mad' in method
                    use_max = 'max' in method
                    penalty = compute_penalty(w_pert, w_cons, w_phot, use_mad, use_max)
                    agg_opt_corrs[m] = corr_fn(penalty, epe)
                else:
                    agg_opt_corrs[m] = agg_eq_corrs[m]
        else:
            agg_opt_corrs = agg_eq_corrs.copy()
        
        results[mode] = {
            'diag': diag_corrs,
            'upper': upper_corrs,
            'agg_eq': agg_eq_corrs,
            'agg_opt': agg_opt_corrs,
        }
    
    results['k'] = k
    results['n_configs'] = n_configs
    
    return results


# =============================================================================
# Figure Generation
# =============================================================================

def generate_correlation_figure(results: dict,
                                 output_path: Path):
    """
    Generate 2-row stacked correlation matrix visualization.
    
    Top row: all configs
    Bottom row: top-k configs
    
    Each row has:
    - Columns 0-3: Individual metrics (pert, cons, phot, trac)
    - Columns 4-5: Aggregated methods (eq, opt)
    
    Diagonal (col 0-3): metric→EPE (grayscale)
    Upper triangle (col 0-3): metric→metric (blue-white-red)
    Lower triangle: white (N/A)
    Columns 4-5: agg→EPE (diverging blue-white-red)
    """
    metric_names = ['pert', 'cons', 'phot', 'trac']
    agg_names = ['raw_sum', 'raw_max', 'mad_sum', 'mad_max']
    n = len(metric_names)
    k = results['k']
    n_configs = results['n_configs']
    
    fig, axes = plt.subplots(2, 1, figsize=(11, 10), height_ratios=[1, 1])
    
    # Colormaps
    cmap_epe = plt.cm.Greys
    cmap_metric = plt.cm.RdBu_r
    cmap_agg = plt.cm.RdBu_r  # Diverging for agg columns
    
    subplot_titles = ['all configs', f'top {k} OF configs per pixel']
    modes = ['all', 'topk']
    
    for row_idx, (ax, mode, subplot_title) in enumerate(zip(axes, modes, subplot_titles)):
        diag_corrs = results[mode]['diag']
        upper_corrs = results[mode]['upper']
        agg_eq_corrs = results[mode]['agg_eq']
        agg_opt_corrs = results[mode]['agg_opt']
        
        # Draw main 4x4 grid
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal - metric→EPE correlation (grayscale)
                    val = diag_corrs[i]
                    color = cmap_epe(val) if not np.isnan(val) else 'white'
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                         facecolor=color, edgecolor='black', linewidth=2)
                    ax.add_patch(rect)
                    if not np.isnan(val):
                        text_color = 'white' if val > 0.5 else 'black'
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                               fontsize=14, fontweight='bold', color=text_color)
                elif j > i:
                    # Upper triangle - metric→metric correlation (blue-white-red)
                    val = upper_corrs[i, j]
                    norm_val = (val + 1) / 2 if not np.isnan(val) else 0.5
                    color = cmap_metric(norm_val) if not np.isnan(val) else 'white'
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         facecolor=color, edgecolor='black', linewidth=1)
                    ax.add_patch(rect)
                    if not np.isnan(val):
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                               fontsize=12)
                else:
                    # Lower triangle - white (N/A)
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         facecolor='white', edgecolor='black', linewidth=1)
                    ax.add_patch(rect)
        
        # Offset for agg columns (gap serves as separator)
        agg_offset = 0.5
        
        # Draw equal weights column (column 4)
        eq_col = n + agg_offset
        for i in range(n):
            val = agg_eq_corrs[i]
            # Diverging colormap for agg columns
            norm_val = (val + 1) / 2 if not np.isnan(val) else 0.5
            color = cmap_agg(norm_val) if not np.isnan(val) else 'white'
            rect = plt.Rectangle((eq_col - 0.5, i - 0.5), 1, 1,
                                 facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            if not np.isnan(val):
                # White text on strong colors (far from 0)
                text_color = 'white' if abs(val) > 0.4 else 'black'
                ax.text(eq_col, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=14, fontweight='bold', color=text_color)
        
        # Draw optimized weights column (column 5)
        opt_col = n + 1 + agg_offset
        for i in range(n):
            val = agg_opt_corrs[i]
            # Diverging colormap for agg columns
            norm_val = (val + 1) / 2 if not np.isnan(val) else 0.5
            color = cmap_agg(norm_val) if not np.isnan(val) else 'white'
            rect = plt.Rectangle((opt_col - 0.5, i - 0.5), 1, 1,
                                 facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            if not np.isnan(val):
                text_color = 'white' if abs(val) > 0.4 else 'black'
                ax.text(opt_col, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=14, fontweight='bold', color=text_color)
        
        # Add aggregation method labels on right side
        for i, agg_name in enumerate(agg_names):
            ax.text(opt_col + 0.6, i, agg_name, ha='left', va='center', fontsize=10)
        
        ax.set_xlim(-0.5, opt_col + 1.5)
        ax.set_ylim(n - 0.5, -0.5)
        
        # Subplot title
        ax.set_title(subplot_title, fontsize=12, fontweight='bold', pad=8)
        
        # X-axis labels only on bottom row
        ax.set_xticks(list(range(n)) + [eq_col, opt_col])
        if row_idx == 1:
            ax.set_xticklabels(metric_names + ['eq', 'opt'], fontsize=12)
        else:
            ax.set_xticklabels([])
        
        # Y-axis labels (metric names only, no row label)
        ax.set_yticks(range(n))
        ax.set_yticklabels(metric_names, fontsize=12)
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Suptitle
    fig.suptitle('Spearman pixel-wise metric correlations', fontsize=14, fontweight='bold', y=0.98)
    
    # Add legend/annotation on top row (lower left)
    legend_text = "Diagonal + agg: metric → EPE\nUpper triangle: metric → metric"
    axes[0].text(0.02, 0.02, legend_text, transform=axes[0].transAxes,
                verticalalignment='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # Colorbars (on the right side of the figure)
    fig.subplots_adjust(right=0.82)
    
    # EPE colorbar (grayscale, 0 to 1)
    cbar_ax1 = fig.add_axes([0.88, 0.55, 0.02, 0.35])
    sm_epe = plt.cm.ScalarMappable(cmap=cmap_epe, norm=plt.Normalize(0, 1))
    cbar_epe = fig.colorbar(sm_epe, cax=cbar_ax1)
    cbar_epe.set_label('ρ(*, EPE)', fontsize=10)
    
    # Metric-metric colorbar (diverging, -1 to 1)
    cbar_ax2 = fig.add_axes([0.88, 0.1, 0.02, 0.35])
    sm_metric = plt.cm.ScalarMappable(cmap=cmap_metric, norm=plt.Normalize(-1, 1))
    cbar_metric = fig.colorbar(sm_metric, cax=cbar_ax2)
    cbar_metric.set_label('ρ(metric, metric)', fontsize=10)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================

def run_correlation_analysis(config: dict, data_dir: Path = Path('data')):
    """Run correlation analysis on detected experiment."""
    print("=" * 60)
    print("📊 METRIC CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Auto-detect experiment
    movie_hash, of_hash = auto_detect_experiment(data_dir)
    
    # Paths
    movie_dir = data_dir / movie_hash
    analysis_dir = movie_dir / 'analysis' / of_hash
    sweep_dir = analysis_dir / 'sweep'
    figures_dir = analysis_dir / 'figures'
    optimization_dir = analysis_dir / 'optimization'
    figures_dir.mkdir(exist_ok=True)
    
    print(f"Experiment: {analysis_dir}")
    
    # Load optimized weights if available
    opt_weights = {}
    method_names = ['raw_sum', 'raw_max', 'mad_sum', 'mad_max']
    for method in method_names:
        weights_path = optimization_dir / method / 'best_weights.json'
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                data = json.load(f)
            opt_weights[method] = data.get('best_selection_config', data.get('best_weights', {}))
    
    if opt_weights:
        print(f"   (loaded optimized weights for {len(opt_weights)} methods)")
    else:
        print("   (no optimized weights found, using equal for both columns)")
    
    # Get EPE power from config
    eval_config = config.get('evaluation', {})
    epe_power = eval_config.get('epe_power', 2.0)
    
    # Load movie sequence (silently)
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        movie = load_movie_sequence(movie_dir)
    n_pairs = len(movie.pairs)
    
    print(f"Pooling {n_pairs} pairs...")
    
    # Pool pixel data from all pairs
    all_pixel_data = {'epe': [], 'pert': [], 'cons': [], 'phot': [], 'trac': []}
    total_pixels = 0
    
    for pair_idx in range(n_pairs):
        pair_dir = sweep_dir / f'pair_{pair_idx:03d}'
        
        results_path = pair_dir / 'results_full.pkl'
        if not results_path.exists():
            print(f"❌ ERROR: Missing {results_path}")
            sys.exit(1)
        
        with open(results_path, 'rb') as f:
            results_full = pickle.load(f)
        
        pair = movie.pairs[pair_idx]
        pixel_data, n_valid = compute_pixel_correlations(
            results_full,
            pair.u_truth, pair.v_truth, pair.valid_mask,
            epe_power
        )
        
        for key in all_pixel_data:
            all_pixel_data[key].append(pixel_data[key])
        total_pixels += n_valid
    
    # Concatenate all pixels
    for key in all_pixel_data:
        all_pixel_data[key] = np.concatenate(all_pixel_data[key], axis=0)
    
    print(f"Total pixels: {total_pixels:,}")
    
    # Compute correlations
    max_sample = 10000
    n_sample = min(total_pixels, max_sample)
    print(f"Computing correlations (sampling {n_sample:,} pixels)...")
    results = compute_correlation_matrix(
        all_pixel_data, opt_weights=opt_weights or None, max_pixels=max_sample
    )
    
    k = results['k']
    n_configs = results['n_configs']
    
    # Print results
    metric_names = ['pert', 'cons', 'phot', 'trac']
    agg_names = ['raw_sum', 'raw_max', 'mad_sum', 'mad_max']
    
    print(f"\n--- All {n_configs} configs ---")
    print(f"Metric → EPE correlations:")
    for i, name in enumerate(metric_names):
        print(f"   {name}: ρ = {results['all']['diag'][i]:.3f}")
    
    print(f"\nAggregated → EPE correlations:")
    print(f"   {'Method':<10} {'eq':<8} {'opt':<8}")
    print(f"   {'-'*26}")
    for i, name in enumerate(agg_names):
        print(f"   {name:<10} {results['all']['agg_eq'][i]:<8.3f} {results['all']['agg_opt'][i]:<8.3f}")
    
    print(f"\n--- Top {k} configs ---")
    print(f"Metric → EPE correlations:")
    for i, name in enumerate(metric_names):
        print(f"   {name}: ρ = {results['topk']['diag'][i]:.3f}")
    
    print(f"\nAggregated → EPE correlations:")
    print(f"   {'Method':<10} {'eq':<8} {'opt':<8}")
    print(f"   {'-'*26}")
    for i, name in enumerate(agg_names):
        print(f"   {name:<10} {results['topk']['agg_eq'][i]:<8.3f} {results['topk']['agg_opt'][i]:<8.3f}")
    
    # Generate figure
    output_path = figures_dir / 'metric_correlations.png'
    generate_correlation_figure(results, output_path)
    
    print(f"\n📊 {output_path}")
    print("=" * 60)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze metric correlations with EPE'
    )
    parser.add_argument('config', type=Path, help='TOML config file')
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                       help='Base data directory (default: data/)')
    
    args = parser.parse_args()
    
    if not args.config.exists():
        print(f"❌ ERROR: Config file not found: {args.config}")
        sys.exit(1)
    
    with open(args.config, 'rb') as f:
        config = tomli.load(f)
    
    run_correlation_analysis(config, args.data_dir)


if __name__ == "__main__":
    main()
