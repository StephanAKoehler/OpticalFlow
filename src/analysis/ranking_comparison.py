# File: src/analysis/ranking_comparison.py
"""
Config ranking comparison analysis.

Computes per-config metrics and rankings, saves to CSV for later analysis.
Provides print function for summary table.

Uses unified selection module for consistent computation with optimizer.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Import from unified selection module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ensemble.selection import (
    ALL_METRICS,
    GAIN_METRICS,
    METRIC_KEY_MAP,
    build_metric_stacks,
    normalize_stacks,
    compute_config_means,
    compute_penalty,
    compute_config_penalty,
    select_ensemble,
    gather_flow,
    compute_epe_stats,
    get_enabled_metrics,
)


# =============================================================================
# Constants
# =============================================================================

# Default weights (photometric only)
DEFAULT_WEIGHTS = {
    'traction': 0.0,
    'perturbation_rms': 0.0,
    'consistency': 0.0,
    'photometric': 1.0,
    'photometric_rgb': 0.0,
    'photometric_rgb_log': 0.0,
    'speed_sym': 0.0,
}


# =============================================================================
# Weight Handling
# =============================================================================

def get_effective_weights(
    optimization_summary: Optional[dict],
    method: str,
    config_weights: Optional[dict],
    default_weights: Optional[dict] = None
) -> dict:
    """
    Get effective weights for a method, checking sources in priority order.
    
    Priority: optimization results > config weights > default weights
    
    Args:
        optimization_summary: Dict with optimization results (may have per-method weights)
        method: Method name (e.g., 'raw_sum', 'mad_sum')
        config_weights: Weights from config file
        default_weights: Fallback default weights
        
    Returns:
        Weights dict with all metric names
    """
    # Priority 1: Optimized weights for this method
    if optimization_summary and 'results' in optimization_summary:
        if method in optimization_summary['results']:
            opt_weights = optimization_summary['results'][method].get('best_weights')
            if opt_weights:
                return opt_weights.copy()
    
    # Priority 2: Config weights (combine weights + fixed_weights)
    if config_weights:
        weights = config_weights.get('weights', {}).copy()
        fixed = config_weights.get('fixed_weights', {})
        weights.update(fixed)
        if weights:
            return weights
    
    # Priority 3: Provided defaults
    if default_weights:
        return default_weights.copy()
    
    # Fallback: default weights
    return DEFAULT_WEIGHTS.copy()


# =============================================================================
# Flow Stack Building
# =============================================================================

def build_flow_stacks(results_full: list) -> tuple[np.ndarray, np.ndarray]:
    """Build u and v flow stacks from results."""
    n_configs = len(results_full)
    first_flow = results_full[0]['flows']['u_AB']
    H, W = first_flow.shape
    
    u_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    v_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    for i, r in enumerate(results_full):
        u_stack[i] = r['flows']['u_AB']
        v_stack[i] = r['flows']['v_AB']
    
    return u_stack, v_stack


# =============================================================================
# Config Rankings (for SINGLE CONFIG EPE table)
# =============================================================================

def compute_config_rankings(
    pair_dir: Path,
    optimization_summary: Optional[dict] = None,
    config_weights: Optional[dict] = None,
    default_weights: Optional[dict] = None
) -> pd.DataFrame:
    """
    Compute per-config metrics and rankings.
    
    Args:
        pair_dir: Directory containing pair results (results_full.pkl, sweep_results.csv)
        optimization_summary: Optional dict with optimized weights
        config_weights: Optional dict with weights from config file
        default_weights: Default weights if no other source available
        
    Returns:
        DataFrame with config metrics and ranks
    """
    # Load data
    results_path = pair_dir / 'results_full.pkl'
    csv_path = pair_dir / 'sweep_results.csv'
    
    if not results_path.exists() or not csv_path.exists():
        return None
    
    with open(results_path, 'rb') as f:
        results_full = pickle.load(f)
    
    sweep_df = pd.read_csv(csv_path)
    
    if 'mean_epe_powered' not in sweep_df.columns:
        return None
    
    # Start building output dataframe
    df = pd.DataFrame()
    df['config_name'] = sweep_df['config_name']
    df['EPE'] = sweep_df['mean_epe_powered']
    
    # Find available metrics
    first_metrics = results_full[0]['metrics']
    available_metrics = []
    for metric_name, metric_key in METRIC_KEY_MAP.items():
        if metric_key in first_metrics:
            available_metrics.append(metric_name)
    
    # Build metric stacks and compute config means
    stacks = build_metric_stacks(results_full, available_metrics)
    config_means = compute_config_means(stacks)
    
    # Add config means to dataframe
    for metric_name in available_metrics:
        df[metric_name] = config_means[metric_name]
    
    # Compute penalty for each method
    methods = ['raw_sum', 'mad_sum', 'raw_max', 'mad_max']
    
    for method in methods:
        weights = get_effective_weights(optimization_summary, method, config_weights, default_weights)
        
        # Determine enabled metrics
        enabled = get_enabled_metrics(weights)
        if not enabled:
            continue
        
        # Build stacks for enabled metrics only
        enabled_stacks = {k: v for k, v in stacks.items() if k in enabled}
        
        # Normalize if needed
        normalize_method = 'mad' if method.startswith('mad_') else 'raw'
        norm_stacks = normalize_stacks(enabled_stacks, normalize_method, 
                                       {k: v for k, v in config_means.items() if k in enabled})
        
        # Compute config-level penalty
        aggregation = 'sum' if method.endswith('_sum') else 'max'
        penalty = compute_config_penalty(norm_stacks, weights, aggregation)
        
        df[f'{method}_penalty'] = penalty
    
    # Compute ranks
    df['EPE_rank'] = df['EPE'].rank(method='dense').astype(int)
    
    for metric in available_metrics:
        df[f'{metric}_rank'] = df[metric].rank(method='dense').astype(int)
    
    for method in methods:
        col = f'{method}_penalty'
        if col in df.columns:
            df[f'{method}_rank'] = df[col].rank(method='dense').astype(int)
    
    return df


# =============================================================================
# Ensemble EPE (for EPE SUMMARY table)
# =============================================================================

def compute_ensemble_epe(
    pair_dir: Path,
    optimization_summary: Optional[dict] = None,
    config_weights: Optional[dict] = None,
    default_weights: Optional[dict] = None
) -> pd.DataFrame:
    """
    Compute EPE (mean and std) for oracle, best_single, and ensemble methods.
    
    Args:
        pair_dir: Directory containing oracle.npz, results_full.pkl
        optimization_summary: Dict with optimization results (has optimized weights)
        config_weights: Weights from config file
        default_weights: Default weights
        
    Returns:
        DataFrame with columns: method, mean_epe, std_epe, median_epe, w_*
    """
    rows = []
    
    # Load oracle data
    oracle_path = pair_dir / 'oracle.npz'
    if not oracle_path.exists():
        return pd.DataFrame(rows)
    
    oracle_data = np.load(oracle_path)
    epe_power = float(oracle_data.get('epe_power', 2.0))
    
    # Oracle EPE
    oracle_epe = float(oracle_data.get('oracle_epe_forward_powered', 
                                       oracle_data.get('oracle_epe_forward', np.nan)))
    oracle_std = float(oracle_data.get('oracle_epe_forward_powered_std',
                                       oracle_data.get('oracle_epe_forward_std', 0.0)))
    oracle_median = float(oracle_data.get('oracle_epe_forward_powered_median', np.nan))
    rows.append({'method': 'oracle', 'mean_epe': oracle_epe, 'std_epe': oracle_std, 'median_epe': oracle_median})
    
    # Load results
    results_path = pair_dir / 'results_full.pkl'
    if not results_path.exists():
        return pd.DataFrame(rows)
    
    with open(results_path, 'rb') as f:
        results_full = pickle.load(f)
    
    # Load ground truth
    movie_dir = pair_dir.parent.parent.parent.parent
    u_truth_path = movie_dir / 'frames' / 'u_000.npz'
    v_truth_path = movie_dir / 'frames' / 'v_000.npz'
    
    if not u_truth_path.exists() or not v_truth_path.exists():
        return pd.DataFrame(rows)
    
    u_truth = np.load(u_truth_path)['u']
    v_truth = np.load(v_truth_path)['v']
    H, W = u_truth.shape
    
    # Create valid mask
    valid_mask = ~(np.isnan(u_truth) | np.isnan(v_truth))
    
    # Apply boundary margin
    config_path = pair_dir.parent.parent / 'optical_flow.toml'
    if config_path.exists():
        import tomli
        with open(config_path, 'rb') as f:
            of_config = tomli.load(f)
        winsize = of_config.get('parameter_sweep', {}).get('winsize', [15])
        if isinstance(winsize, list):
            winsize = max(winsize)
        margin = winsize // 2 + 1
        valid_mask[:margin, :] = False
        valid_mask[-margin:, :] = False
        valid_mask[:, :margin] = False
        valid_mask[:, -margin:] = False
    
    # Build flow stacks
    u_stack, v_stack = build_flow_stacks(results_full)
    n_configs = len(results_full)
    
    # Compute EPE for each config
    epe_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    for i in range(n_configs):
        diff = np.sqrt((u_stack[i] - u_truth)**2 + (v_stack[i] - v_truth)**2)
        epe_stack[i] = diff ** epe_power
    
    # Best single config
    mean_epes = np.array([np.nanmean(epe_stack[i][valid_mask]) for i in range(n_configs)])
    best_idx = np.argmin(mean_epes)
    rows.append({
        'method': 'best_single',
        'mean_epe': mean_epes[best_idx],
        'std_epe': np.nanstd(epe_stack[best_idx][valid_mask]),
        'median_epe': np.nanmedian(epe_stack[best_idx][valid_mask])
    })
    
    # Find available metrics
    first_metrics = results_full[0]['metrics']
    available_metrics = []
    for metric_name, metric_key in METRIC_KEY_MAP.items():
        if metric_key in first_metrics:
            available_metrics.append(metric_name)
    
    # Compute ensemble EPE for each method
    methods = ['raw_sum', 'mad_sum', 'raw_max', 'mad_max']
    
    for method in methods:
        weights = get_effective_weights(optimization_summary, method, config_weights, default_weights)
        
        # Determine enabled metrics
        enabled = get_enabled_metrics(weights)
        if not enabled:
            # No metrics enabled - skip
            continue
        
        # Filter to available metrics
        enabled = [m for m in enabled if m in available_metrics]
        if not enabled:
            continue
        
        # Build metric stacks
        stacks = build_metric_stacks(results_full, enabled)
        
        # Normalize
        normalize_method = 'mad' if method.startswith('mad_') else 'raw'
        stacks = normalize_stacks(stacks, normalize_method)
        
        # Compute penalty
        aggregation = 'sum' if method.endswith('_sum') else 'max'
        penalty = compute_penalty(stacks, weights, aggregation)
        
        # Select and gather flow
        selection = select_ensemble(penalty)
        u_ens, v_ens = gather_flow(u_stack, v_stack, selection)
        
        # Compute EPE stats
        stats = compute_epe_stats(u_ens, v_ens, u_truth, v_truth, valid_mask, epe_power)
        
        # Build row with weights
        row = {
            'method': method,
            'mean_epe': stats['mean'],
            'std_epe': stats['std'],
            'median_epe': stats['median'],
        }
        for metric_name in ALL_METRICS:
            row[f'w_{metric_name}'] = weights.get(metric_name, 0.0)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


# =============================================================================
# Analysis and Display
# =============================================================================

def run_ranking_analysis(
    pair_dir: Path,
    optimization_summary: Optional[dict] = None,
    config_weights: Optional[dict] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full ranking analysis for a pair directory.
    
    Returns:
        (config_rankings_df, ensemble_epe_df)
    """
    config_df = compute_config_rankings(pair_dir, optimization_summary, config_weights)
    ensemble_df = compute_ensemble_epe(pair_dir, optimization_summary, config_weights)
    
    return config_df, ensemble_df


def save_ranking_analysis(
    pair_dir: Path,
    config_df: pd.DataFrame,
    ensemble_df: pd.DataFrame
) -> None:
    """Save ranking analysis to CSV files."""
    if config_df is not None and len(config_df) > 0:
        config_df.to_csv(pair_dir / 'config_rankings.csv', index=False)
    
    if ensemble_df is not None and len(ensemble_df) > 0:
        ensemble_df.to_csv(pair_dir / 'ensemble_epe.csv', index=False)


def print_ranking_summary(
    pair_dir: Path,
    optimization_summary: Optional[dict] = None,
    config_weights: Optional[dict] = None
) -> None:
    """
    Print ranking summary tables.
    """
    # Load or compute rankings
    config_csv = pair_dir / 'config_rankings.csv'
    ensemble_csv = pair_dir / 'ensemble_epe.csv'
    
    if config_csv.exists():
        config_df = pd.read_csv(config_csv)
    else:
        config_df = compute_config_rankings(pair_dir, optimization_summary, config_weights)
        if config_df is not None:
            config_df.to_csv(config_csv, index=False)
    
    if ensemble_csv.exists():
        ensemble_df = pd.read_csv(ensemble_csv)
    else:
        ensemble_df = compute_ensemble_epe(pair_dir, optimization_summary, config_weights)
        if ensemble_df is not None:
            ensemble_df.to_csv(ensemble_csv, index=False)
    
    # Print SINGLE CONFIG EPE table
    if config_df is not None and len(config_df) > 0:
        _print_config_rankings(config_df)
    
    # Print EPE SUMMARY table
    if ensemble_df is not None and len(ensemble_df) > 0:
        _print_epe_summary(ensemble_df)


def _print_config_rankings(config_df: pd.DataFrame, n_show: int = 5) -> None:
    """Print SINGLE CONFIG EPE ranking table."""
    n_configs = len(config_df)
    n_unique = config_df['EPE'].nunique()
    
    # Build EPE rank mapping
    sorted_epes = sorted(config_df['EPE'].unique())
    epe_to_rank = {epe: i+1 for i, epe in enumerate(sorted_epes)}
    
    print("=" * 78)
    print(f"SINGLE CONFIG EPE (ranked by each criterion; {n_configs} configs with {n_unique} unique EPE values)")
    print("=" * 78)
    
    # Header
    header = f"{'Ranking':<16} |"
    for i in range(1, n_show + 1):
        header += f" {'Rank ' + str(i):^14} |"
    print(header)
    print("-" * 78)
    
    # Determine columns to show
    ranking_cols = [('EPE', 'EPE')]
    ranking_cols.append((None, '---'))  # Separator
    
    metric_labels = {
        'traction': 'Traction',
        'perturbation': 'Perturbation',
        'consistency': 'Consistency',
        'photometric': 'Photometric',
        'photometric_rgb': 'Photo_RGB',
        'photometric_rgb_log': 'Photo_Log',
        'speed_sym': 'Speed_Sym',
    }
    
    for metric, label in metric_labels.items():
        if metric in config_df.columns:
            ranking_cols.append((metric, label))
    
    # Add penalty columns
    for method in ['raw_sum', 'mad_sum', 'raw_max', 'mad_max']:
        col = f'{method}_penalty'
        if col in config_df.columns:
            ranking_cols.append((col, method))
    
    # Print each row
    for col, label in ranking_cols:
        if col is None:
            print("-" * 78)
            continue
        
        if col not in config_df.columns:
            continue
        
        sorted_df = config_df.sort_values(col)
        
        seen_epes = set()
        top_epes = []
        for _, row in sorted_df.iterrows():
            epe = row['EPE']
            if epe not in seen_epes:
                seen_epes.add(epe)
                epe_rank = epe_to_rank[epe]
                top_epes.append((epe_rank, epe))
                if len(top_epes) >= n_show:
                    break
        
        row_str = f"{label:<16} |"
        for epe_rank, epe in top_epes:
            row_str += f" ({epe_rank:>2}) {epe:>8.4f}  |"
        print(row_str)
    
    print("=" * 78)


def _print_epe_summary(ensemble_df: pd.DataFrame) -> None:
    """Print EPE SUMMARY table."""
    # Weight columns
    weight_cols = [
        ('w_traction', 'trac'),
        ('w_perturbation_rms', 'pert'),
        ('w_consistency', 'cons'),
        ('w_photometric', 'phot'),
        ('w_photometric_rgb', 'pRGB'),
        ('w_photometric_rgb_log', 'pLog'),
        ('w_speed_sym', 'sSpd'),
    ]
    
    available_weights = [(col, abbr) for col, abbr in weight_cols if col in ensemble_df.columns]
    
    print("=" * 130)
    print("EPE SUMMARY (over valid pixels)")
    print("=" * 130)
    
    header = f"{'Method':<12} | {'Mean':>8} | {'Median':>8} | {'Std':>8} | {'vs Best':>8} |"
    for col, abbr in available_weights:
        header += f" {abbr:>5} |"
    print(header)
    print("-" * 130)
    
    # Get best_single as baseline
    best_row = ensemble_df[ensemble_df['method'] == 'best_single']
    best_epe = best_row['mean_epe'].values[0] if len(best_row) > 0 else None
    
    # Order
    method_order = ['best_single', 'oracle', 'raw_sum', 'mad_sum', 'raw_max', 'mad_max']
    
    for method in method_order:
        row = ensemble_df[ensemble_df['method'] == method]
        if len(row) == 0:
            continue
        
        mean_epe = row['mean_epe'].values[0]
        std_epe = row['std_epe'].values[0] if 'std_epe' in row.columns else 0.0
        median_epe = row['median_epe'].values[0] if 'median_epe' in row.columns else np.nan
        
        # vs best
        if best_epe and best_epe > 0:
            if method == 'best_single':
                vs_best = "baseline"
            else:
                pct = 100 * (mean_epe - best_epe) / best_epe
                vs_best = f"{pct:+.1f}%"
        else:
            vs_best = "-"
        
        median_str = f"{median_epe:>8.4f}" if not np.isnan(median_epe) else f"{'-':>8}"
        
        row_str = f"{method:<12} | {mean_epe:>8.4f} | {median_str} | {std_epe:>8.4f} | {vs_best:>8} |"
        
        if available_weights:
            if method in ['best_single', 'oracle']:
                for _ in available_weights:
                    row_str += f" {'-':>5} |"
            else:
                for col, abbr in available_weights:
                    w_val = row[col].values[0] if col in row.columns else 0.0
                    row_str += f" {w_val:>5.2f} |"
        
        print(row_str)
    
    print("=" * 130)


# =============================================================================
# Backward Compatibility Functions
# =============================================================================

def save_ranking_comparison(
    analysis_dir: Path,
    sweep_dir: Path,
    optimization_dir: Optional[Path] = None,
    config: Optional[dict] = None
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Compute and save ranking comparison CSVs.
    
    Args:
        analysis_dir: Base analysis directory
        sweep_dir: Directory containing sweep results
        optimization_dir: Directory containing optimization results.
            Can be either:
            - Full path to opt_hash subdir: .../optimization/{opt_hash}/
            - Parent optimization dir: .../optimization/ (will find latest)
        config: Optional config dict with [optimization] section
        
    Returns:
        Tuple of (config_rankings_path, ensemble_epe_path) or (None, None) if no GT
    """
    # Find first pair directory
    pair_dirs = sorted(sweep_dir.glob('pair_*'))
    if not pair_dirs:
        return None, None
    
    pair_dir = pair_dirs[0]
    
    # Load optimization summary if available
    optimization_summary = None
    if optimization_dir:
        # Check if this is the parent dir or a specific opt_hash dir
        summary_path = optimization_dir / 'optimization_summary.json'
        if not summary_path.exists():
            # Try legacy location
            summary_path = optimization_dir / 'summary.json'
        if not summary_path.exists():
            # Maybe this is the parent dir - find most recent opt_hash subdir
            opt_subdirs = sorted(optimization_dir.glob('*/optimization_summary.json'))
            if opt_subdirs:
                # Use the most recently modified one
                summary_path = max(opt_subdirs, key=lambda p: p.stat().st_mtime)
        
        if summary_path.exists():
            with open(summary_path) as f:
                optimization_summary = json.load(f)
    
    # Get config weights
    config_weights = None
    if config:
        config_weights = config.get('optimization', {})
    
    # Compute rankings
    config_df = compute_config_rankings(pair_dir, optimization_summary, config_weights)
    if config_df is None:
        return None, None
    
    # Compute ensemble EPE
    ensemble_df = compute_ensemble_epe(pair_dir, optimization_summary, config_weights)
    
    # Save CSVs
    config_path = analysis_dir / 'config_rankings.csv'
    ensemble_path = analysis_dir / 'ensemble_epe.csv'
    
    config_df.to_csv(config_path, index=False)
    if ensemble_df is not None:
        ensemble_df.to_csv(ensemble_path, index=False)
    
    return config_path, ensemble_path


def print_ranking_comparison(
    analysis_dir: Path,
    n_top: int = 5
) -> None:
    """
    Print ranking comparison table from saved CSVs.
    
    Args:
        analysis_dir: Directory containing config_rankings.csv and ensemble_epe.csv
        n_top: Number of top ranks to display
    """
    config_path = analysis_dir / 'config_rankings.csv'
    ensemble_path = analysis_dir / 'ensemble_epe.csv'
    
    if not config_path.exists():
        print("   (No ranking comparison available - requires ground truth)")
        return
    
    config_df = pd.read_csv(config_path)
    ensemble_df = pd.read_csv(ensemble_path) if ensemble_path.exists() else None
    
    # Print tables
    _print_config_rankings(config_df, n_top)
    if ensemble_df is not None and len(ensemble_df) > 0:
        _print_epe_summary(ensemble_df)


# =============================================================================
# Main Entry Point
# =============================================================================

def analyze_sweep_rankings(
    analysis_dir: Path,
    optimization_summary: Optional[dict] = None,
    config_weights: Optional[dict] = None
) -> None:
    """
    Analyze rankings for all pairs in a sweep.
    
    Args:
        analysis_dir: Directory containing sweep results (with pair_* subdirs)
        optimization_summary: Optional optimization results
        config_weights: Optional config weights
    """
    pair_dirs = sorted(analysis_dir.glob('pair_*'))
    
    if not pair_dirs:
        print("No pair directories found")
        return
    
    # For now, just use first pair
    pair_dir = pair_dirs[0]
    
    print_ranking_summary(pair_dir, optimization_summary, config_weights)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ranking_comparison.py <analysis_dir>")
        sys.exit(1)
    
    analysis_dir = Path(sys.argv[1])
    analyze_sweep_rankings(analysis_dir)
