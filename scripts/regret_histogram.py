#!/usr/bin/env python3
# File: scripts/regret_histogram.py
"""
Analyze selection regret: how much worse is metric-based selection vs oracle?

Produces heatmap:
- Rows: 8 methods (4 aggregations × 2 weight schemes)
- Columns: log2-binned absolute regret (agg_EPE - oracle_EPE)
- Color: log10(pixel count)

Usage:
    python scripts/regret_histogram.py config.toml
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tomli

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.data_loader import load_movie_sequence
from src.evaluation.ground_truth import compute_epe


# =============================================================================
# Auto-detection
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
# Selection and Regret Computation
# =============================================================================

def compute_selection_epe(results_full: list,
                          u_truth: np.ndarray,
                          v_truth: np.ndarray, 
                          valid_mask: np.ndarray,
                          weights: dict,
                          use_mad: bool,
                          use_max: bool,
                          epe_power: float) -> np.ndarray:
    """
    Compute EPE for metric-based config selection.
    
    Returns EPE map for selected configs.
    """
    n_configs = len(results_full)
    H, W = valid_mask.shape
    
    # Get metrics
    pert = np.stack([r['metrics']['displacements_sensitivity_A2B'] for r in results_full], axis=0)
    cons = np.stack([r['metrics']['consistency_A'] for r in results_full], axis=0)
    phot = np.stack([r['metrics']['photometric_A'] for r in results_full], axis=0)
    
    # MAD normalize if needed
    if use_mad:
        def mad_normalize(arr):
            median = np.median(arr, axis=0, keepdims=True)
            mad = np.median(np.abs(arr - median), axis=0, keepdims=True)
            mad = np.maximum(mad, 1e-10)
            return (arr - median) / mad
        pert = mad_normalize(pert)
        cons = mad_normalize(cons)
        phot = mad_normalize(phot)
    
    # Get weights
    w_pert = weights.get('perturbation_rms', 1.0)
    w_cons = weights.get('consistency', 1.0)
    w_phot = weights.get('photometric', 1.0)
    
    # Compute penalty
    if use_max:
        penalty = np.maximum(
            np.maximum(w_pert * pert**2, w_cons * cons**2),
            w_phot * phot**2
        )
    else:
        penalty = w_pert * pert**2 + w_cons * cons**2 + w_phot * phot**2
    
    # Select best config per pixel (argmin penalty)
    selection = np.argmin(penalty, axis=0)
    
    # Compute EPE for selected configs
    u_stack = np.stack([r['flows']['u_AB'] for r in results_full], axis=0)
    v_stack = np.stack([r['flows']['v_AB'] for r in results_full], axis=0)
    
    # Gather selected flow
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    u_selected = u_stack[selection, ii, jj]
    v_selected = v_stack[selection, ii, jj]
    
    # Compute EPE
    epe_map = compute_epe(u_selected, v_selected, u_truth, v_truth, valid_mask, power=epe_power)
    
    return epe_map


def compute_oracle_epe(results_full: list,
                       u_truth: np.ndarray,
                       v_truth: np.ndarray,
                       valid_mask: np.ndarray,
                       epe_power: float) -> np.ndarray:
    """
    Compute oracle EPE (best possible per pixel).
    
    Returns EPE map for oracle selection.
    """
    n_configs = len(results_full)
    H, W = valid_mask.shape
    
    # Compute EPE for all configs
    epe_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    for i, r in enumerate(results_full):
        u_est = r['flows']['u_AB']
        v_est = r['flows']['v_AB']
        epe_stack[i] = compute_epe(u_est, v_est, u_truth, v_truth, valid_mask, power=epe_power)
    
    # Oracle = min EPE per pixel
    oracle_epe = np.nanmin(epe_stack, axis=0)
    
    return oracle_epe


# =============================================================================
# Binning
# =============================================================================

def create_log2_bins(max_regret: float, min_power: int = -4) -> tuple[np.ndarray, list]:
    """
    Create log2 bins from 0 to max_regret.
    
    Returns:
        bin_edges: array for np.digitize
        bin_labels: list of strings for display (LaTeX format)
    """
    # Find max power needed
    if max_regret <= 0:
        max_power = 0
    else:
        max_power = int(np.ceil(np.log2(max_regret)))
    
    # Build edges: 0, 2^min_power, 2^(min_power+1), ..., 2^max_power
    powers = list(range(min_power, max_power + 1))
    edges = [0.0] + [2.0**p for p in powers]
    
    # Labels with exponent range notation
    labels = ['=0']
    for i, p in enumerate(powers):
        if i == 0:
            # First bin: (0, 2^min_power] -> 2^(−∞, min_power]
            labels.append(f'$2^{{(-\\infty,{p}]}}$')
        else:
            # Other bins: (2^prev, 2^p] -> 2^(prev, p]
            labels.append(f'$2^{{({powers[i-1]},{p}]}}$')
    
    return np.array(edges), labels


def bin_regret(regret: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Bin regret values.
    
    Returns array of counts per bin.
    """
    # Handle exact zeros separately
    exact_zero = np.sum(regret == 0)
    
    # Bin non-zeros
    nonzero_regret = regret[regret > 0]
    if len(nonzero_regret) > 0:
        # digitize returns 1-indexed bins, edges[0]=0 so bin 1 is (0, edges[1]]
        bin_indices = np.digitize(nonzero_regret, bin_edges[1:])  # Skip 0 edge
        counts = np.bincount(bin_indices, minlength=len(bin_edges))
    else:
        counts = np.zeros(len(bin_edges), dtype=int)
    
    # Insert exact zero count at position 0
    counts[0] = exact_zero
    
    return counts[:len(bin_edges)]


# =============================================================================
# Figure Generation
# =============================================================================

def generate_regret_figure(all_counts: dict,
                           bin_labels: list,
                           output_path: Path):
    """
    Generate regret histogram heatmap.
    
    Args:
        all_counts: {method_name: counts_array}
        bin_labels: list of bin label strings
        output_path: where to save
    """
    # Row order (grouped by method)
    row_order = [
        'raw_sum (eq)', 'raw_sum (opt)',
        'raw_max (eq)', 'raw_max (opt)',
        'mad_sum (eq)', 'mad_sum (opt)',
        'mad_max (eq)', 'mad_max (opt)',
    ]
    
    # Filter to rows that exist
    row_order = [r for r in row_order if r in all_counts]
    n_rows = len(row_order)
    n_cols = len(bin_labels)
    
    # Build count matrix
    count_matrix = np.zeros((n_rows, n_cols), dtype=float)
    for i, row_name in enumerate(row_order):
        counts = all_counts[row_name]
        count_matrix[i, :len(counts)] = counts[:n_cols]
    
    # Find right truncation (last column with any counts)
    col_has_counts = np.any(count_matrix > 0, axis=0)
    last_col = np.max(np.where(col_has_counts)[0]) + 1 if np.any(col_has_counts) else n_cols
    
    # Truncate
    count_matrix = count_matrix[:, :last_col]
    bin_labels = bin_labels[:last_col]
    n_cols = last_col
    
    # Compute column-wise ranks (1=most, 8=least)
    # Higher count = rank 1 (most pixels in that bin)
    rank_matrix = np.zeros_like(count_matrix)
    for j in range(n_cols):
        col = count_matrix[:, j]
        # Only rank non-zero entries
        nonzero_mask = col > 0
        if np.any(nonzero_mask):
            # Rank by count descending (most = rank 1)
            # argsort of -col gives indices sorted by descending count
            order = np.argsort(-col)
            for rank, idx in enumerate(order):
                if col[idx] > 0:
                    rank_matrix[idx, j] = rank + 1
                else:
                    rank_matrix[idx, j] = np.nan  # No rank for zeros
        else:
            rank_matrix[:, j] = np.nan
    
    # Create figure
    fig_width = max(10, n_cols * 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    # Colormap: 1 (most) = dark, 8 (least) = light
    # Use reversed colormap so low rank = dark
    cmap = plt.cm.YlOrRd_r
    
    # Normalize ranks to [0, 1] for colormap (1->0, 8->1)
    norm = mcolors.Normalize(vmin=1, vmax=n_rows)
    
    # Draw tiles manually
    for i in range(n_rows):
        for j in range(n_cols):
            count = int(count_matrix[i, j])
            rank = rank_matrix[i, j]
            
            if count == 0:
                # White for zero counts
                color = 'white'
            else:
                color = cmap(norm(rank))
            
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
            # Add count text
            if count > 0:
                # Determine text color based on rank
                text_color = 'white' if rank <= 3 else 'black'
                ax.text(j, i, f'{count:,}', ha='center', va='center',
                       fontsize=8, color=text_color)
    
    # Set axis limits
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    
    # Labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(bin_labels, fontsize=9, rotation=45, ha='right')
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_order, fontsize=10)
    
    # Add horizontal lines between method groups
    for i in [2, 4, 6]:
        if i < n_rows:
            ax.axhline(y=i - 0.5, color='black', linewidth=2)
    
    ax.set_xlabel('Regret (selected EPE − oracle EPE)', fontsize=12)
    ax.set_title('Selection Regret Distribution', fontsize=14, fontweight='bold')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Rank (1=most, 8=least)', fontsize=10)
    cbar.set_ticks([1, 2, 3, 4, 5, 6, 7, 8])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================

def run_regret_analysis(config: dict, data_dir: Path = Path('data')):
    """Run regret analysis on detected experiment."""
    print("=" * 60)
    print("📊 SELECTION REGRET ANALYSIS")
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
    
    # Get EPE power from config
    eval_config = config.get('evaluation', {})
    epe_power = eval_config.get('epe_power', 2.0)
    
    # Load movie sequence (silently)
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        movie = load_movie_sequence(movie_dir)
    n_pairs = len(movie.pairs)
    
    print(f"Processing {n_pairs} pairs...")
    
    # Equal weights
    eq_weights = {'perturbation_rms': 1.0, 'consistency': 1.0, 'photometric': 1.0}
    
    # Collect all regret values per method
    all_regrets = {f'{m} (eq)': [] for m in method_names}
    all_regrets.update({f'{m} (opt)': [] for m in method_names})
    
    # Process each pair
    for pair_idx in range(n_pairs):
        pair_dir = sweep_dir / f'pair_{pair_idx:03d}'
        
        results_path = pair_dir / 'results_full.pkl'
        if not results_path.exists():
            print(f"❌ ERROR: Missing {results_path}")
            sys.exit(1)
        
        with open(results_path, 'rb') as f:
            results_full = pickle.load(f)
        
        pair = movie.pairs[pair_idx]
        u_truth = pair.u_truth
        v_truth = pair.v_truth
        valid_mask = pair.valid_mask
        
        # Compute oracle EPE
        oracle_epe = compute_oracle_epe(results_full, u_truth, v_truth, valid_mask, epe_power)
        
        # Compute selection EPE for each method
        for method in method_names:
            use_mad = 'mad' in method
            use_max = 'max' in method
            
            # Equal weights
            sel_epe_eq = compute_selection_epe(
                results_full, u_truth, v_truth, valid_mask,
                eq_weights, use_mad, use_max, epe_power
            )
            regret_eq = sel_epe_eq - oracle_epe
            all_regrets[f'{method} (eq)'].append(regret_eq[valid_mask].flatten())
            
            # Optimized weights
            weights = opt_weights.get(method, eq_weights)
            sel_epe_opt = compute_selection_epe(
                results_full, u_truth, v_truth, valid_mask,
                weights, use_mad, use_max, epe_power
            )
            regret_opt = sel_epe_opt - oracle_epe
            all_regrets[f'{method} (opt)'].append(regret_opt[valid_mask].flatten())
    
    # Concatenate all pairs
    for key in all_regrets:
        all_regrets[key] = np.concatenate(all_regrets[key])
    
    # Find max regret across all methods
    max_regret = max(np.max(r) for r in all_regrets.values() if len(r) > 0)
    print(f"Max regret: {max_regret:.4f}")
    
    # Create bins
    bin_edges, bin_labels = create_log2_bins(max_regret)
    print(f"Bins: {len(bin_labels)}")
    
    # Bin each method
    all_counts = {}
    for key, regret in all_regrets.items():
        counts = bin_regret(regret, bin_edges)
        all_counts[key] = counts
    
    # Print summary
    print(f"\nExact matches (regret = 0):")
    total_pixels = len(all_regrets['raw_sum (eq)'])
    for key in all_counts:
        exact = all_counts[key][0]
        pct = 100 * exact / total_pixels
        print(f"   {key:<18}: {exact:>8,} ({pct:>5.1f}%)")
    
    # Generate figure
    output_path = figures_dir / 'regret_histogram.png'
    generate_regret_figure(all_counts, bin_labels, output_path)
    
    print(f"\n📊 {output_path}")
    print("=" * 60)
    
    return all_counts


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze selection regret distribution'
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
    
    run_regret_analysis(config, args.data_dir)


if __name__ == "__main__":
    main()
