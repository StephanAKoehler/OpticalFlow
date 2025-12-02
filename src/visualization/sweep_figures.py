# File: src/visualization/sweep_figures.py
"""
Generate standard figures for parameter sweep results.

Creates:
- EPE distribution as dual heatmap (parameter ranks + error bins)
- Oracle comparison (separate figures for forward and symmetric flows)
- Metric correlations with EPE
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

from src.evaluation.flow_deformation import compute_flow_deformation, plot_flow_decomposition


# ============================================================================
# Data Structure Helpers
# ============================================================================

def _ensure_flattened(results: List[Dict]) -> List[Dict]:
    """
    Ensure results are in flattened format for visualization.
    
    Handles both old flat format and new structured format.
    """
    if len(results) == 0:
        return results
    
    # Check if structured format (has 'metadata' key)
    if 'metadata' in results[0]:
        from src.core.data_structures import flatten_for_visualization
        return [flatten_for_visualization(r) for r in results]
    else:
        return results


def flatten_param(key: str, value) -> Dict[str, Any]:
    """
    Flatten a parameter into one or more rankable values.
    
    Returns dict of {param_name: scalar_value}
    """
    if isinstance(value, dict):
        # Flatten nested dicts: {'refine': {'iter': 5}} → 'refine.iter': 5
        result = {}
        for k, v in value.items():
            nested = flatten_param(f"{key}.{k}", v)
            result.update(nested)
        return result
    elif isinstance(value, (list, np.ndarray)):
        # Convert arrays to scalar summary
        if len(value) == 0:
            return {key: 0}
        # Use length as the rankable value
        return {f"{key}_len": len(value)}
    elif isinstance(value, bool):
        # Boolean to 0/1
        return {key: int(value)}
    elif isinstance(value, (int, float)):
        # Already scalar
        return {key: value}
    elif isinstance(value, str):
        # Keep string as-is for alphabetical ranking
        return {key: value}
    else:
        # Unknown type - convert to string
        return {key: str(value)}


def make_hashable(val):
    """Convert any value to a hashable type for set operations."""
    if isinstance(val, np.ndarray):
        return tuple(val.flat)
    elif isinstance(val, dict):
        return tuple(sorted(val.items()))
    elif isinstance(val, list):
        return tuple(val)
    else:
        return val


def flatten_all_params(results_full: List[Dict], 
                       exclude_keys: List[str]) -> Tuple[List[Dict], List[str]]:
    """
    Flatten all parameters from configs, excluding specified keys.
    
    Returns:
        (flattened_params, flat_param_names)
    """
    # First pass: flatten all configs and collect all possible param names
    all_flat_params = []
    all_param_names = set()
    
    for res in results_full:
        flat_params = {}
        for key, val in res.items():
            if key not in exclude_keys:
                flat_params.update(flatten_param(key, val))
        all_flat_params.append(flat_params)
        all_param_names.update(flat_params.keys())
    
    # Sort param names for consistent ordering
    sorted_param_names = sorted(list(all_param_names))
    
    return all_flat_params, sorted_param_names


# ============================================================================
# EPE Distribution Helpers
# ============================================================================

def _extract_algorithm(results_flat: List[Dict]) -> str:
    """Extract algorithm name from first config."""
    if 'algorithm' in results_flat[0]:
        algorithm = results_flat[0]['algorithm']
    elif 'config' in results_flat[0] and isinstance(results_flat[0]['config'], dict):
        algorithm = results_flat[0]['config'].get('algorithm', 'UNKNOWN')
    else:
        algorithm = 'UNKNOWN'
    return algorithm.upper()


def _prepare_params(results_flat: List[Dict], 
                    sweep_params: Optional[List[str]]) -> Tuple[List[Dict], List[str]]:
    """
    Prepare parameter data for visualization.
    
    Returns:
        (flattened_params, flat_param_names)
    """
    # Define keys to exclude from parameter extraction
    exclude_keys = ['algorithm', 'u_AB', 'v_AB', 'u_sym_A', 'v_sym_A', 
                   'u_BA', 'v_BA', 'u_sym_B', 'v_sym_B', 'config_name',
                   # Exclude metric outputs
                   'traction_A', 'traction_B', 
                   'consistency_A', 'consistency_B',
                   'photometric_A', 'photometric_B',
                   'displacements_sensitivity_A2B', 'displacements_sensitivity_B2A',
                   'consistency_variance_A', 'consistency_variance_B',
                   'consistency_2_A', 'consistency_2_B',
                   'consistency_variance_2_A', 'consistency_variance_2_B',
                   'displacements_sensitivity_A2A', 'displacements_sensitivity_B2B']
    
    # If sweep_params specified, use only those; otherwise flatten all non-excluded params
    if sweep_params is not None:
        # First flatten everything, then filter to keep only swept params
        all_flattened_params, all_flat_param_names = flatten_all_params(results_flat, exclude_keys)
        
        # Filter to keep only parameters that match swept param names
        kept_param_names = []
        for flat_name in all_flat_param_names:
            # Check if this flattened name corresponds to a swept param
            for swept_p in sweep_params:
                # Match multiple patterns including config. prefix
                if (flat_name == swept_p or 
                    flat_name.startswith(f"{swept_p}.") or
                    flat_name == f"{swept_p}_len" or
                    flat_name == f"config.{swept_p}" or
                    flat_name.startswith(f"config.{swept_p}.")):
                    kept_param_names.append(flat_name)
                    break
        
        # Filter flattened params to only include kept names
        flattened_params = []
        for flat_params in all_flattened_params:
            filtered = {k: v for k, v in flat_params.items() if k in kept_param_names}
            flattened_params.append(filtered)
        
        flat_param_names = kept_param_names
    else:
        # Flatten all parameters (old behavior with expanded exclusion list)
        flattened_params, flat_param_names = flatten_all_params(results_flat, exclude_keys)
    
    return flattened_params, flat_param_names


def _prepare_config_epe_data(results_flat: List[Dict],
                             u_truth: np.ndarray,
                             v_truth: np.ndarray,
                             valid_mask: np.ndarray,
                             flattened_params: List[Dict]) -> List[Dict]:
    """
    Compute EPE for each config.
    
    Returns list of dicts with keys: name, params, epe
    """
    config_data = []
    
    for i, res in enumerate(results_flat):
        u_sym = res['u_sym_A']
        v_sym = res['v_sym_A']
        
        # Compute per-pixel EPE
        epe_map = np.sqrt((u_sym - u_truth)**2 + (v_sym - v_truth)**2)
        epe_valid = epe_map[valid_mask]
        
        config_data.append({
            'name': res['config_name'],
            'params': flattened_params[i],
            'epe': epe_valid
        })
    
    return config_data


def _compute_oracle_flow(results_flat: List[Dict],
                        u_truth: np.ndarray,
                        v_truth: np.ndarray,
                        valid_mask: np.ndarray) -> np.ndarray:
    """
    Select per-pixel best flow.
    
    Returns EPE array for oracle (valid pixels only).
    """
    n_configs = len(results_flat)
    
    # Stack all flows
    u_stack = np.stack([results_flat[i]['u_sym_A'] for i in range(n_configs)], axis=0)
    v_stack = np.stack([results_flat[i]['v_sym_A'] for i in range(n_configs)], axis=0)
    
    # Compute EPE stack
    epe_stack = np.sqrt((u_stack - u_truth[np.newaxis, :, :])**2 + 
                        (v_stack - v_truth[np.newaxis, :, :])**2)
    
    # Select best per pixel
    oracle_selection = np.argmin(epe_stack, axis=0)
    
    u_oracle = np.zeros_like(u_truth, dtype=np.float32)
    v_oracle = np.zeros_like(v_truth, dtype=np.float32)
    
    for i in range(n_configs):
        mask = oracle_selection == i
        u_oracle[mask] = u_stack[i][mask]
        v_oracle[mask] = v_stack[i][mask]
    
    epe_oracle_map = np.sqrt((u_oracle - u_truth)**2 + (v_oracle - v_truth)**2)
    epe_oracle = epe_oracle_map[valid_mask]
    
    return epe_oracle


def _create_epe_bins(max_epe: float, n_bins: int = 4) -> np.ndarray:
    """
    Create exponential bins for EPE histogram.
    
    Shows n_bins bins ending at the bin containing max_epe.
    
    Args:
        max_epe: Maximum EPE value in data
        n_bins: Number of bins to show (default: 4)
    
    Returns:
        bin edges array (length n_bins + 1)
    """
    # Find which power of 2 bin contains max_epe
    # max_epe in [2^p, 2^(p+1)) means it's in bin labeled 2^p
    max_power = int(np.floor(np.log2(max_epe + 1e-10)))
    
    # Show n_bins bins ending at max_power
    min_power = max_power - n_bins + 1
    
    # Create bin edges from min_power to max_power+1 (need max_power+1 for upper edge)
    bins = [2.0 ** p for p in range(min_power, max_power + 2)]
    
    return np.array(bins)


def _compute_histograms(config_data: List[Dict],
                       epe_oracle: np.ndarray,
                       u_ensemble: Optional[np.ndarray],
                       v_ensemble: Optional[np.ndarray],
                       u_truth: np.ndarray,
                       v_truth: np.ndarray,
                       valid_mask: np.ndarray,
                       bins: np.ndarray,
                       ensemble_label: str) -> List[Dict]:
    """
    Compute histograms for all configs + oracle + ensemble.
    
    Returns list of row dicts with keys: name, params, counts, max_occupied_bin, max_bin_count
    """
    min_bin = bins[0]
    n_bins = len(bins) - 1
    all_rows = []
    
    # Oracle row
    oracle_counts = np.zeros(n_bins)
    epe_filtered = epe_oracle[epe_oracle >= min_bin]
    counts, _ = np.histogram(epe_filtered, bins=bins)
    oracle_counts = counts
    
    all_rows.append({
        'name': 'Oracle',
        'params': None,
        'counts': oracle_counts,
        'max_occupied_bin': np.max(np.where(oracle_counts > 0)[0]) if np.any(oracle_counts > 0) else -1,
        'max_bin_count': np.max(oracle_counts) if np.any(oracle_counts > 0) else 0
    })
    
    # Ensemble row (if provided)
    if u_ensemble is not None and v_ensemble is not None:
        epe_ensemble = np.sqrt((u_ensemble - u_truth)**2 + (v_ensemble - v_truth)**2)
        epe_ensemble_valid = epe_ensemble[valid_mask]
        epe_filtered = epe_ensemble_valid[epe_ensemble_valid >= min_bin]
        counts, _ = np.histogram(epe_filtered, bins=bins)
        ensemble_counts = counts
        
        all_rows.append({
            'name': ensemble_label,
            'params': None,
            'counts': ensemble_counts,
            'max_occupied_bin': np.max(np.where(ensemble_counts > 0)[0]) if np.any(ensemble_counts > 0) else -1,
            'max_bin_count': np.max(ensemble_counts) if np.any(ensemble_counts > 0) else 0
        })
    
    # Config rows
    for config in config_data:
        epe_filtered = config['epe'][config['epe'] >= min_bin]
        counts, _ = np.histogram(epe_filtered, bins=bins)
        
        max_occupied = np.max(np.where(counts > 0)[0]) if np.any(counts > 0) else -1
        max_count = np.max(counts) if np.any(counts > 0) else 0
        
        all_rows.append({
            'name': config['name'],
            'params': config['params'],
            'counts': counts,
            'max_occupied_bin': max_occupied,
            'max_bin_count': max_count
        })
    
    return all_rows


def _sort_rows_by_quality(all_rows: List[Dict]) -> List[Dict]:
    """
    Sort rows: Oracle first, Ensemble second (if exists), then configs by max occupied bin.
    
    Returns sorted list.
    """
    oracle_row = all_rows[0]
    
    # Check if second row is Ensemble
    if len(all_rows) > 1 and all_rows[1]['name'] not in ['Oracle'] and all_rows[1]['params'] is None:
        ensemble_row = all_rows[1]
        config_part = all_rows[2:]
        has_ensemble = True
    else:
        ensemble_row = None
        config_part = all_rows[1:]
        has_ensemble = False
    
    # Sort configs by quality (lower max_occupied_bin = better)
    # For ties, use max_bin_count (lower = better, fewer errors at higher bins)
    config_part.sort(key=lambda x: (x['max_occupied_bin'], x['max_bin_count']))
    
    # Reassemble: Oracle, [Ensemble], Configs
    if has_ensemble:
        special_part = [oracle_row, ensemble_row]
    else:
        special_part = [oracle_row]
    
    sorted_rows = config_part + special_part
    
    return sorted_rows


def _build_parameter_rank_matrix(sorted_rows: List[Dict],
                                 flat_param_names: List[str]) -> Tuple[np.ndarray, int, Dict]:
    """
    Build (n_rows × n_params) matrix of parameter ranks.
    
    Returns:
        (param_matrix, max_rank, param_ranks)
    """
    n_rows = len(sorted_rows)
    
    if len(flat_param_names) == 0:
        return np.array([]).reshape(n_rows, 0), 1, {}
    
    # Get all unique values for each parameter
    param_values = {p: set() for p in flat_param_names}
    for row in sorted_rows:
        if row['params'] is not None:
            for p in flat_param_names:
                val = row['params'].get(p, None)
                if val is not None:
                    param_values[p].add(make_hashable(val))
    
    # Create rank mapping for each parameter
    param_ranks = {}
    max_rank = 0
    for p in flat_param_names:
        sorted_vals = sorted(list(param_values[p]))
        param_ranks[p] = {val: rank+1 for rank, val in enumerate(sorted_vals)}
        max_rank = max(max_rank, len(sorted_vals))
    
    # Build parameter matrix (rows × params)
    param_matrix = np.full((n_rows, len(flat_param_names)), np.nan)
    
    for i, row in enumerate(sorted_rows):
        if row['params'] is not None:
            for j, p in enumerate(flat_param_names):
                val = row['params'].get(p, None)
                if val is not None:
                    val_hash = make_hashable(val)
                    param_matrix[i, j] = param_ranks[p][val_hash]
    
    return param_matrix, max_rank, param_ranks


def _plot_parameter_heatmap(ax_param, ax_cbar, param_matrix: np.ndarray,
                           flat_param_names: List[str], sorted_rows: List[Dict],
                           max_rank: int):
    """Plot left heatmap showing parameter ranks."""
    if len(flat_param_names) == 0:
        ax_param.text(0.5, 0.5, 'No parameters to display', 
                     ha='center', va='center', transform=ax_param.transAxes)
        ax_cbar.axis('off')
        return
    
    # Use discrete colormap for ranks (1-indexed)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    
    # Get distinct colors from tab20
    base_colors = plt.cm.tab20.colors
    colors = [base_colors[i] for i in range(max_rank)]
    cmap_param = ListedColormap(colors)
    
    # Create boundary norm for discrete bins
    # Boundaries at 0.5, 1.5, 2.5, 3.5, 4.5 for ranks 1, 2, 3, 4
    boundaries = [i + 0.5 for i in range(max_rank + 1)]
    norm = BoundaryNorm(boundaries, cmap_param.N)
    
    # Create display matrix - mark special rows as NaN
    param_display = param_matrix.copy()
    for i, row in enumerate(sorted_rows):
        if row['params'] is None:
            param_display[i, :] = np.nan
    
    # Plot heatmap (ranks are 1-indexed: 1, 2, 3, ...)
    im_param = ax_param.imshow(param_display, cmap=cmap_param, norm=norm,
                               aspect='auto', interpolation='nearest')
    
    # Add parameter value annotations
    for row in range(len(sorted_rows)):
        if sorted_rows[row]['params'] is None:
            # Oracle/Ensemble rows - add text labels centered
            mid_col = len(flat_param_names) / 2
            ax_param.text(mid_col - 0.5, row, sorted_rows[row]['name'], 
                        ha='center', va='center',
                        fontsize=12, fontweight='bold', color='black')
        else:
            # Config rows - show parameter values
            for col, param_name in enumerate(flat_param_names):
                if not np.isnan(param_display[row, col]):
                    # Get actual parameter value
                    param_val = sorted_rows[row]['params'].get(param_name, None)
                    
                    if param_val is not None:
                        # Format value
                        if isinstance(param_val, bool):
                            val_str = 'T' if param_val else 'F'
                        elif isinstance(param_val, int):
                            val_str = str(param_val)
                        elif isinstance(param_val, float):
                            val_str = f"{param_val:.2g}"
                        else:
                            val_str = str(param_val)[:5]  # Truncate long strings
                        
                        # Determine text color based on rank
                        rank = int(param_display[row, col])
                        # Use lighter colors for ranks 3-4, darker for 1-2
                        if rank <= 2:
                            text_color = 'white'
                        else:
                            text_color = 'black'
                        
                        # Add text
                        ax_param.text(col, row, val_str,
                                    ha='center', va='center',
                                    fontsize=9, fontweight='bold',
                                    color=text_color)
    
    # Set ticks - clean up parameter names
    clean_param_names = [name.replace('config.', '') for name in flat_param_names]
    ax_param.set_xticks(range(len(flat_param_names)))
    ax_param.set_xticklabels(clean_param_names, rotation=45, ha='right', fontsize=8)
    ax_param.set_yticks(range(len(sorted_rows)))
    # Don't show config names on y-axis - they're too long and cluttered
    ax_param.set_yticklabels([])  # Empty labels
    
    ax_param.set_title('Config Parameters', fontsize=12, fontweight='bold')
    
    # Parameter colorbar (on left side) - categorical ticks only
    cbar_param = plt.colorbar(im_param, cax=ax_cbar, 
                              ticks=range(1, max_rank + 1),
                              boundaries=boundaries)
    cbar_param.set_label('Parameter Rank', fontsize=10)
    ax_cbar.yaxis.set_ticks_position('left')
    ax_cbar.yaxis.set_label_position('left')


def _plot_epe_histogram_heatmap(ax_epe, ax_cbar, count_matrix: np.ndarray,
                                bins: np.ndarray, sorted_rows: List[Dict],
                                oracle_idx: int):
    """Plot right heatmap showing EPE error bins."""
    n_bins = len(bins) - 1
    n_rows = len(sorted_rows)
    
    # Apply log10 for visualization
    log_count_matrix = np.log10(count_matrix + 1)
    
    # Find min/max of non-zero values
    non_zero_values = log_count_matrix[log_count_matrix > 0]
    if len(non_zero_values) > 0:
        log_min_nonzero = np.min(non_zero_values)
        log_max = np.max(log_count_matrix)
    else:
        log_min_nonzero = 0
        log_max = 1
    
    # Create custom colormap: white for zero, RdBu_r for non-zero values
    from matplotlib.colors import ListedColormap
    import matplotlib.cm as cm
    
    # Create a masked array where zeros are masked
    masked_log_matrix = np.ma.masked_where(log_count_matrix == 0, log_count_matrix)
    
    # Use RdBu_r colormap
    cmap = cm.get_cmap('RdBu_r')
    # Set color for masked (zero) values to white
    cmap.set_bad(color='white')
    
    # Plot with colormap scaled to non-zero range
    im_epe = ax_epe.imshow(masked_log_matrix, cmap=cmap, aspect='auto',
                          interpolation='nearest', 
                          vmin=log_min_nonzero, vmax=log_max)
    
    # Add rank annotations for each column (bin)
    # Rank 1 = lowest count (best), higher ranks = more errors
    for col in range(n_bins):
        # Get counts for this column
        col_counts = count_matrix[:, col]
        
        # Rank: lowest count = rank 1
        # Use lexsort with negative row indices as tiebreaker
        row_indices = np.arange(n_rows)
        sorted_indices = np.lexsort((-row_indices, col_counts))
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(1, len(col_counts) + 1)
        
        # Add text annotations with better contrast
        for row in range(n_rows):
            rank = ranks[row]
            count = int(count_matrix[row, col])
            
            # Format label as "rank (count)"
            label = f"{rank}\n({count})"
            
            # Determine text color based on cell value
            if count_matrix[row, col] == 0:
                # White/empty cells - use black text
                text_color = 'black'
            else:
                # For colored cells, dark backgrounds get white text
                if log_count_matrix[row, col] > 3.0:
                    text_color = 'white'
                else:
                    text_color = 'black'
            
            # Add text with outline for better visibility
            ax_epe.text(col, row, label, 
                       ha='center', va='center',
                       fontsize=8, fontweight='bold', color=text_color,
                       path_effects=[
                           path_effects.withStroke(linewidth=2.5, 
                                                 foreground='white' if text_color == 'black' else 'black')
                       ])
    
    # Set ticks - show as powers of 2
    bin_labels = []
    for i in range(n_bins):
        # Convert bin edge to power of 2
        power = int(np.round(np.log2(bins[i])))
        bin_labels.append(f"2^{power}")
    
    ax_epe.set_xticks(range(n_bins))
    ax_epe.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax_epe.set_yticks(range(n_rows))
    ax_epe.set_yticklabels([''] * n_rows)  # Already labeled on left side
    
    ax_epe.set_xlabel('EPE Bin (pixels)', fontsize=11)
    ax_epe.set_title('Error Distribution', fontsize=12, fontweight='bold')
    
    # EPE colorbar
    cbar_epe = plt.colorbar(im_epe, cax=ax_cbar)
    cbar_epe.set_label('log₁₀(count + 1)', fontsize=10)


# ============================================================================
# Oracle Comparison Helpers
# ============================================================================

def _get_top_configs(results_flat: List[Dict],
                    ensemble_selection_valid: np.ndarray,
                    u_truth: np.ndarray,
                    v_truth: np.ndarray,
                    valid_mask: np.ndarray,
                    flow_type: str,
                    cutoff: float) -> List[Dict]:
    """
    Get all configs with selection frequency and EPE.
    
    Returns list of dicts sorted by selection frequency.
    """
    # Find top configs by selection frequency
    unique_configs, counts = np.unique(ensemble_selection_valid, return_counts=True)
    sorted_indices = np.argsort(-counts)
    top_config_indices = unique_configs[sorted_indices]
    top_config_counts = counts[sorted_indices]
    
    n_configs = len(results_flat)
    
    # Get all config data
    all_configs = []
    for i in range(n_configs):
        config_idx = i
        selection_pct = 0
        if config_idx in top_config_indices:
            idx_in_top = np.where(top_config_indices == config_idx)[0][0]
            selection_pct = 100 * top_config_counts[idx_in_top] / len(ensemble_selection_valid)
        
        # Get flows for this config
        if flow_type == 'forward':
            u = results_flat[config_idx]['u_AB']
            v = results_flat[config_idx]['v_AB']
        else:  # symmetric
            u = results_flat[config_idx]['u_sym_A']
            v = results_flat[config_idx]['v_sym_A']
        
        # Compute EPE
        epe = np.sqrt((u - u_truth)**2 + (v - v_truth)**2)
        epe_valid = np.maximum(epe[valid_mask], cutoff)
        
        all_configs.append({
            'index': config_idx,
            'name': results_flat[config_idx]['config_name'],
            'selection_pct': selection_pct,
            'epe': epe_valid
        })
    
    # Sort by selection frequency
    all_configs.sort(key=lambda x: x['selection_pct'], reverse=True)
    
    return all_configs


def _plot_oracle_vs_ensemble(ax, epe_oracle: np.ndarray, epe_ensemble: np.ndarray,
                             bins: np.ndarray, stats_oracle: Dict, stats_ensemble: Dict):
    """Plot top panel: Oracle vs Ensemble step plots with log scales."""
    color_oracle = 'green'
    color_ensemble = 'blue'
    
    # Clipping thresholds
    x_clip = 0.01  # EPE below this is "perfect"
    y_clip = 0.01  # Percentage below this is noise
    
    # Count pixels below x_clip
    pct_below_clip_oracle = 100 * np.sum(epe_oracle < x_clip) / len(epe_oracle)
    pct_below_clip_ensemble = 100 * np.sum(epe_ensemble < x_clip) / len(epe_ensemble)
    pct_shown = 100 - max(pct_below_clip_oracle, pct_below_clip_ensemble)
    
    # Compute histograms
    oracle_hist, _ = np.histogram(epe_oracle, bins=bins)
    ensemble_hist, _ = np.histogram(epe_ensemble, bins=bins)
    
    # Normalize to percentages
    oracle_pct = 100 * oracle_hist / len(epe_oracle)
    ensemble_pct = 100 * ensemble_hist / len(epe_ensemble)
    
    # Plot step plots with alpha for overlap visibility
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.step(bin_centers, oracle_pct, where='mid', color=color_oracle, 
            linewidth=2, label='Oracle', alpha=0.7)
    ax.step(bin_centers, ensemble_pct, where='mid', color=color_ensemble, 
            linewidth=2, label='Ensemble', alpha=0.7)
    
    # Add statistics
    ax.axvline(stats_oracle['median'], color=color_oracle, linestyle='--', 
               linewidth=1.5, alpha=0.7)
    ax.axvline(stats_ensemble['median'], color=color_ensemble, linestyle='--', 
               linewidth=1.5, alpha=0.7)
    
    # Log scales with clipping
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(left=x_clip)
    ax.set_ylim(bottom=y_clip)
    
    # Set tight upper y-limit based on data AFTER x-clipping
    visible_mask = bin_centers >= x_clip
    max_pct_visible = max(oracle_pct[visible_mask].max(), ensemble_pct[visible_mask].max())
    ax.set_ylim(top=max_pct_visible * 2)  # Small margin above max
    
    ax.set_xlabel('EPE (px)', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Oracle vs Ensemble', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Add clipping info text box
    clip_text = f'Showing {pct_shown:.1f}% of pixels'
    ax.text(0.98, 0.98, clip_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9, family='monospace')


def _plot_config_contributions(ax, all_configs: List[Dict], bins: np.ndarray):
    """Plot middle panel: Stacked area showing config contributions."""
    n_configs = len(all_configs)
    config_colors = plt.cm.tab10(np.arange(n_configs))
    
    # Clipping threshold
    x_clip = 0.01
    
    # Compute histograms for all configs
    histograms = []
    for config in all_configs:
        hist, _ = np.histogram(config['epe'], bins=bins)
        pct = 100 * hist / len(config['epe'])
        histograms.append(pct)
    
    histograms = np.array(histograms)
    
    # Normalize to 100% at each bin
    col_sums = histograms.sum(axis=0)
    col_sums[col_sums == 0] = 1  # Avoid division by zero
    normalized = 100 * histograms / col_sums[np.newaxis, :]
    
    # Stacked area plot
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.stackplot(bin_centers, *normalized, colors=config_colors, alpha=0.8)
    
    # Log x-scale with clipping
    ax.set_xscale('log')
    ax.set_xlim(left=x_clip)
    
    ax.set_xlabel('EPE (px)', fontsize=12)
    ax.set_ylabel('Contribution (%)', fontsize=12)
    ax.set_title('Config Contributions (Normalized)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, which='both')


def _plot_top_configs(ax, all_configs: List[Dict], bins: np.ndarray, n_top: int = 3):
    """Plot bottom panel: Step plots for top N configs with log scales."""
    config_colors = plt.cm.tab10(np.arange(len(all_configs)))
    
    # Clipping thresholds
    x_clip = 0.01
    y_clip = 0.01
    
    # Track max percentage for y-limit and pixels shown
    max_pct_visible = 0
    min_pct_below_clip = 100
    
    # Linewidths: thicker for less-selected configs (harder to see)
    linewidths = [1.5, 2.0, 2.5, 3.0, 3.5]  # Increasing thickness
    
    # Precompute bin centers for visibility mask
    bin_centers = (bins[:-1] + bins[1:]) / 2
    visible_mask = bin_centers >= x_clip
    
    for i in range(min(n_top, len(all_configs))):
        config = all_configs[i]
        hist, _ = np.histogram(config['epe'], bins=bins)
        pct = 100 * hist / len(config['epe'])
        
        # Track max for y-limit (only visible bins)
        if pct[visible_mask].max() > max_pct_visible:
            max_pct_visible = pct[visible_mask].max()
        
        # Track pixels below clip
        pct_below_x = 100 * np.sum(config['epe'] < x_clip) / len(config['epe'])
        if pct_below_x < min_pct_below_clip:
            min_pct_below_clip = pct_below_x
        
        label = f"{config['name']} ({config['selection_pct']:.1f}%)"
        lw = linewidths[min(i, len(linewidths) - 1)]
        ax.step(bin_centers, pct, where='mid', color=config_colors[i], 
                linewidth=lw, label=label, alpha=0.7)
    
    # Log scales with clipping
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(left=x_clip)
    ax.set_ylim(bottom=y_clip, top=max_pct_visible * 2)  # Tight upper limit
    
    ax.set_xlabel('EPE (px)', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(f'Top {n_top} Configs by Selection', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Add clipping info text box
    pct_shown = 100 - min_pct_below_clip
    clip_text = f'Showing {pct_shown:.1f}% of pixels'
    ax.text(0.98, 0.98, clip_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9, family='monospace')


# ============================================================================
# Metric Correlation Helpers
# ============================================================================

def _compute_config_level_correlations(sweep_df: pd.DataFrame,
                                       metric_names: List[str]) -> np.ndarray:
    """
    Compute config-level correlations between metrics and EPE.
    
    Returns: (n_metrics × 2) array [forward_corr, symmetric_corr]
    """
    metric_cols = ['mean_' + m for m in metric_names]
    epe_cols = ['mean_epe_forward', 'mean_epe_symmetric']
    
    corr_data = sweep_df[metric_cols + epe_cols].copy()
    full_corr = corr_data.corr()
    config_level_corr = full_corr.loc[metric_cols, epe_cols].values
    
    return config_level_corr


def _compute_per_pixel_correlations(results_flat: List[Dict],
                                    u_truth: np.ndarray,
                                    v_truth: np.ndarray,
                                    valid_mask: np.ndarray,
                                    metric_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-pixel correlations averaged across all pixels.
    
    OPTIMIZED: Vectorized correlation computation (no pixel loop).
    
    Returns: (per_metric_corr_fwd, per_metric_corr_sym)
    """
    n_configs = len(results_flat)
    H, W = valid_mask.shape
    
    # Compute EPE stacks
    epe_forward_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    epe_symmetric_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    for i in range(n_configs):
        u_AB = results_flat[i]['u_AB']
        v_AB = results_flat[i]['v_AB']
        u_sym_A = results_flat[i]['u_sym_A']
        v_sym_A = results_flat[i]['v_sym_A']
        
        epe_forward_stack[i] = np.sqrt((u_AB - u_truth)**2 + (v_AB - v_truth)**2)
        epe_symmetric_stack[i] = np.sqrt((u_sym_A - u_truth)**2 + (v_sym_A - v_truth)**2)
    
    # Stack all metrics
    metric_stacks = {}
    for metric_name in metric_names:
        stack = np.zeros((n_configs, H, W), dtype=np.float32)
        for i in range(n_configs):
            stack[i] = results_flat[i][metric_name]
        metric_stacks[metric_name] = stack
    
    # Compute per-pixel correlations (VECTORIZED)
    per_metric_corr_fwd = []
    per_metric_corr_sym = []
    
    for metric_name in metric_names:
        metric_stack = metric_stacks[metric_name]
        
        # Extract valid pixels: (n_configs, n_valid_pixels)
        metric_valid = metric_stack[:, valid_mask]  # Shape: (n_configs, n_valid)
        epe_fwd_valid = epe_forward_stack[:, valid_mask]  # Shape: (n_configs, n_valid)
        epe_sym_valid = epe_symmetric_stack[:, valid_mask]  # Shape: (n_configs, n_valid)
        
        # Vectorized correlation computation
        # Center the data (subtract mean across configs for each pixel)
        metric_centered = metric_valid - metric_valid.mean(axis=0, keepdims=True)
        epe_fwd_centered = epe_fwd_valid - epe_fwd_valid.mean(axis=0, keepdims=True)
        epe_sym_centered = epe_sym_valid - epe_sym_valid.mean(axis=0, keepdims=True)
        
        # Compute standard deviations
        metric_std = metric_valid.std(axis=0)
        epe_fwd_std = epe_fwd_valid.std(axis=0)
        epe_sym_std = epe_sym_valid.std(axis=0)
        
        # Compute correlations: sum(centered_x * centered_y) / (n * std_x * std_y)
        # Using einsum for efficient computation
        n_samples = metric_valid.shape[0]
        
        corr_fwd = np.sum(metric_centered * epe_fwd_centered, axis=0) / (n_samples * metric_std * epe_fwd_std + 1e-10)
        corr_sym = np.sum(metric_centered * epe_sym_centered, axis=0) / (n_samples * metric_std * epe_sym_std + 1e-10)
        
        # Set correlations to NaN where there's no variation
        corr_fwd[metric_std < 1e-10] = np.nan
        corr_sym[metric_std < 1e-10] = np.nan
        
        # Average across pixels
        per_metric_corr_fwd.append(np.nanmean(corr_fwd))
        per_metric_corr_sym.append(np.nanmean(corr_sym))
    
    return np.array(per_metric_corr_fwd), np.array(per_metric_corr_sym)


def _compute_ensemble_pixel_correlations(results_flat: List[Dict],
                                        ensemble_selection: np.ndarray,
                                        u_truth: np.ndarray,
                                        v_truth: np.ndarray,
                                        valid_mask: np.ndarray,
                                        metric_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-pixel correlations using ensemble-selected configs.
    
    OPTIMIZED: Vectorized extraction using advanced indexing (no coordinate loop).
    
    Returns: (ensemble_corr_fwd, ensemble_corr_sym)
    """
    H, W = valid_mask.shape
    n_configs = len(results_flat)
    
    # Build stacks for all configs
    u_AB_stack = np.stack([results_flat[i]['u_AB'] for i in range(n_configs)])  # (n_configs, H, W)
    v_AB_stack = np.stack([results_flat[i]['v_AB'] for i in range(n_configs)])
    u_sym_stack = np.stack([results_flat[i]['u_sym_A'] for i in range(n_configs)])
    v_sym_stack = np.stack([results_flat[i]['v_sym_A'] for i in range(n_configs)])
    
    metric_stacks = {}
    for metric_name in metric_names:
        metric_stacks[metric_name] = np.stack([results_flat[i][metric_name] for i in range(n_configs)])
    
    # Get ensemble selections at valid pixels
    ensemble_sel_valid = ensemble_selection[valid_mask]  # (n_valid,)
    
    # Get coordinates of valid pixels
    y_coords, x_coords = np.where(valid_mask)  # (n_valid,) each
    
    # Extract values using advanced indexing: stack[config_idx, y, x]
    u_AB_ensemble = u_AB_stack[ensemble_sel_valid, y_coords, x_coords]
    v_AB_ensemble = v_AB_stack[ensemble_sel_valid, y_coords, x_coords]
    u_sym_ensemble = u_sym_stack[ensemble_sel_valid, y_coords, x_coords]
    v_sym_ensemble = v_sym_stack[ensemble_sel_valid, y_coords, x_coords]
    
    # Compute EPE (vectorized)
    u_truth_valid = u_truth[valid_mask]
    v_truth_valid = v_truth[valid_mask]
    
    ensemble_epe_fwd = np.sqrt((u_AB_ensemble - u_truth_valid)**2 + (v_AB_ensemble - v_truth_valid)**2)
    ensemble_epe_sym = np.sqrt((u_sym_ensemble - u_truth_valid)**2 + (v_sym_ensemble - v_truth_valid)**2)
    
    # Extract metrics (vectorized)
    ensemble_metrics = {}
    for metric_name in metric_names:
        ensemble_metrics[metric_name] = metric_stacks[metric_name][ensemble_sel_valid, y_coords, x_coords]
    
    # Compute correlations
    ensemble_corr_fwd = []
    ensemble_corr_sym = []
    
    for metric_name in metric_names:
        metric_vals = ensemble_metrics[metric_name]
        
        if np.std(metric_vals) > 1e-10:
            corr_fwd = np.corrcoef(metric_vals, ensemble_epe_fwd)[0, 1]
            corr_sym = np.corrcoef(metric_vals, ensemble_epe_sym)[0, 1]
        else:
            corr_fwd = 0.0
            corr_sym = 0.0
        
        ensemble_corr_fwd.append(corr_fwd)
        ensemble_corr_sym.append(corr_sym)
    
    return np.array(ensemble_corr_fwd), np.array(ensemble_corr_sym)


def _plot_correlation_comparison(fig, config_corr: np.ndarray,
                                 pixel_corr_fwd: np.ndarray,
                                 pixel_corr_sym: np.ndarray,
                                 ensemble_corr_fwd: np.ndarray,
                                 ensemble_corr_sym: np.ndarray,
                                 clean_labels: List[str]):
    """Plot three-panel correlation comparison."""
    # Create subplots
    axes = fig.subplots(1, 3)
    
    n_metrics = len(clean_labels)
    y_pos = np.arange(n_metrics)
    
    # Panel 1: Config-level
    ax1 = axes[0]
    ax1.barh(y_pos - 0.15, config_corr[:, 0], 0.3, label='Forward', color='blue', alpha=0.7)
    ax1.barh(y_pos + 0.15, config_corr[:, 1], 0.3, label='Symmetric', color='green', alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(clean_labels, fontsize=10)
    ax1.set_xlabel('Correlation with EPE', fontsize=12)
    ax1.set_title('Config-Level Correlations', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0, color='black', linewidth=0.8)
    
    # Panel 2: Per-pixel (best config per metric)
    ax2 = axes[1]
    ax2.barh(y_pos - 0.15, pixel_corr_fwd, 0.3, label='Forward', color='blue', alpha=0.7)
    ax2.barh(y_pos + 0.15, pixel_corr_sym, 0.3, label='Symmetric', color='green', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(clean_labels, fontsize=10)
    ax2.set_xlabel('Correlation with EPE', fontsize=12)
    ax2.set_title('Per-Pixel Correlations\n(Averaged Across Pixels)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvline(0, color='black', linewidth=0.8)
    
    # Panel 3: Ensemble selection
    ax3 = axes[2]
    ax3.barh(y_pos - 0.15, ensemble_corr_fwd, 0.3, label='Forward', color='blue', alpha=0.7)
    ax3.barh(y_pos + 0.15, ensemble_corr_sym, 0.3, label='Symmetric', color='green', alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(clean_labels, fontsize=10)
    ax3.set_xlabel('Correlation with EPE', fontsize=12)
    ax3.set_title('Residual Correlations\n(Post-Selection)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axvline(0, color='black', linewidth=0.8)
    
    fig.tight_layout()


# ============================================================================
# Main Figure Generation Functions
# ============================================================================

def generate_epe_distribution(results_full: List[Dict], 
                              u_truth: np.ndarray,
                              v_truth: np.ndarray,
                              valid_mask: np.ndarray,
                              output_path: Path,
                              u_ensemble: np.ndarray = None,
                              v_ensemble: np.ndarray = None,
                              ensemble_selection: np.ndarray = None,
                              sweep_params: List[str] = None,
                              ensemble_label: str = "Ensemble"):
    """
    Generate EPE distribution as dual heatmap.
    
    Left: Parameter ranks (color = rank of parameter value)
    Right: EPE error bins (color = log10(counts) for EPE >= 0.5 px)
    
    Args:
        u_ensemble, v_ensemble: Ensemble flow fields (optional, adds Ensemble row)
        ensemble_selection: Per-pixel config indices selected by ensemble (optional, filters configs)
        sweep_params: List of parameter names to include. If None, infers from results_full.
        ensemble_label: Label for ensemble row (e.g., "Ensemble (fixed)", "Ensemble (optimized)")
    """
    # Prepare data
    results_flat = _ensure_flattened(results_full)
    algorithm = _extract_algorithm(results_flat)
    flattened_params, flat_param_names = _prepare_params(results_flat, sweep_params)
    
    # Compute EPE for all configs
    config_data = _prepare_config_epe_data(results_flat, u_truth, v_truth, valid_mask, flattened_params)
    
    # Filter configs to only those that appear in ensemble_selection
    if ensemble_selection is not None:
        selected_indices = set(np.unique(ensemble_selection[valid_mask]))
        config_data_filtered = [c for i, c in enumerate(config_data) if i in selected_indices]
        print(f"   Filtered to {len(config_data_filtered)}/{len(config_data)} configs present in ensemble")
    else:
        config_data_filtered = config_data
    
    # Compute oracle
    epe_oracle = _compute_oracle_flow(results_flat, u_truth, v_truth, valid_mask)
    
    # Define EPE bins
    max_epe = max(np.max(epe_oracle), max(np.max(c['epe']) for c in config_data_filtered))
    bins = _create_epe_bins(max_epe)
    
    # Compute histograms
    all_rows = _compute_histograms(config_data_filtered, epe_oracle, u_ensemble, v_ensemble,
                                   u_truth, v_truth, valid_mask, bins, ensemble_label)
    
    # Sort rows
    sorted_rows = _sort_rows_by_quality(all_rows)
    n_rows = len(sorted_rows)
    n_bins = len(bins) - 1
    
    # Build parameter rank matrix
    param_matrix, max_rank, param_ranks = _build_parameter_rank_matrix(sorted_rows, flat_param_names)
    
    # Build EPE count matrix
    count_matrix = np.array([row['counts'] for row in sorted_rows])
    
    # Find oracle index in sorted rows
    oracle_idx = -1
    for i, row in enumerate(sorted_rows):
        if row['name'] == 'Oracle':
            oracle_idx = i
            break
    
    # Create figure
    fig = plt.figure(figsize=(16, max(8, n_rows * 0.5)))
    gs = gridspec.GridSpec(1, 5, figure=fig, 
                          width_ratios=[0.3, len(flat_param_names), 0.2, n_bins, 0.3],
                          wspace=0.05)
    
    ax_param_cbar = fig.add_subplot(gs[0, 0])
    ax_param = fig.add_subplot(gs[0, 1])
    ax_epe = fig.add_subplot(gs[0, 3])
    ax_epe_cbar = fig.add_subplot(gs[0, 4])
    
    # Plot heatmaps
    _plot_parameter_heatmap(ax_param, ax_param_cbar, param_matrix, flat_param_names, 
                           sorted_rows, max_rank)
    _plot_epe_histogram_heatmap(ax_epe, ax_epe_cbar, count_matrix, bins, sorted_rows, oracle_idx)
    
    # Title and save
    n_configs_shown = len([r for r in sorted_rows if r['params'] is not None])
    plt.suptitle(f'Configuration Parameters vs Error Distribution ({algorithm.upper()}, {n_configs_shown} configs shown)',
                fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_oracle_comparison_single_flow(
    epe_oracle: np.ndarray,
    epe_ensemble: np.ndarray,
    ensemble_selection_valid: np.ndarray,
    results_full: list,
    valid_mask: np.ndarray,
    u_truth: np.ndarray,
    v_truth: np.ndarray,
    flow_type: str,
    cutoff: float,
    bins: np.ndarray,
    output_path: Path
):
    """
    Generate oracle comparison for a single flow type (forward or symmetric).
    
    Creates 3-row figure:
    - Row 1: Oracle vs Ensemble (step plots)
    - Row 2: Config contributions (stacked area, 100%)
    - Row 3: Top 3 configs (step plots)
    """
    # Flatten results
    results_flat = _ensure_flattened(results_full)
    
    # Get all config data
    all_configs = _get_top_configs(results_flat, ensemble_selection_valid,
                                   u_truth, v_truth, valid_mask, flow_type, cutoff)
    
    # Compute statistics
    def compute_stats(data):
        return {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data)
        }
    
    stats_oracle = compute_stats(epe_oracle)
    stats_ensemble = compute_stats(epe_ensemble)
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    
    # Plot three panels
    _plot_oracle_vs_ensemble(axes[0], epe_oracle, epe_ensemble, bins, stats_oracle, stats_ensemble)
    _plot_config_contributions(axes[1], all_configs, bins)
    _plot_top_configs(axes[2], all_configs, bins, n_top=3)
    
    # Title and save
    flow_name = flow_type.capitalize()
    fig.suptitle(f'Oracle Comparison: {flow_name} Flow', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_oracle_comparison_histograms(
    u_oracle_fwd: np.ndarray,
    v_oracle_fwd: np.ndarray,
    u_best_fwd: np.ndarray,
    v_best_fwd: np.ndarray,
    u_ensemble_fwd: np.ndarray,
    v_ensemble_fwd: np.ndarray,
    u_oracle_sym: np.ndarray,
    v_oracle_sym: np.ndarray,
    u_best_sym: np.ndarray,
    v_best_sym: np.ndarray,
    u_ensemble_sym: np.ndarray,
    v_ensemble_sym: np.ndarray,
    u_truth: np.ndarray,
    v_truth: np.ndarray,
    valid_mask: np.ndarray,
    ensemble_selection: np.ndarray,
    results_full: list,
    output_path: Path
):
    """
    Generate oracle comparison figures (separate for forward and symmetric).
    
    Creates two figures:
    - oracle_comparison_forward.png
    - oracle_comparison_symmetric.png
    """
    results_flat = _ensure_flattened(results_full)
    
    # Compute per-pixel EPE
    epe_oracle_fwd = np.sqrt((u_oracle_fwd - u_truth)**2 + (v_oracle_fwd - v_truth)**2)
    epe_ensemble_fwd = np.sqrt((u_ensemble_fwd - u_truth)**2 + (v_ensemble_fwd - v_truth)**2)
    
    epe_oracle_sym = np.sqrt((u_oracle_sym - u_truth)**2 + (v_oracle_sym - v_truth)**2)
    epe_ensemble_sym = np.sqrt((u_ensemble_sym - u_truth)**2 + (v_ensemble_sym - v_truth)**2)
    
    # Extract valid pixels
    epe_oracle_fwd_valid = epe_oracle_fwd[valid_mask]
    epe_ensemble_fwd_valid = epe_ensemble_fwd[valid_mask]
    
    epe_oracle_sym_valid = epe_oracle_sym[valid_mask]
    epe_ensemble_sym_valid = epe_ensemble_sym[valid_mask]
    
    ensemble_selection_valid = ensemble_selection[valid_mask]
    
    # Apply cutoff
    cutoff = 1e-3
    epe_oracle_fwd_valid = np.maximum(epe_oracle_fwd_valid, cutoff)
    epe_ensemble_fwd_valid = np.maximum(epe_ensemble_fwd_valid, cutoff)
    
    epe_oracle_sym_valid = np.maximum(epe_oracle_sym_valid, cutoff)
    epe_ensemble_sym_valid = np.maximum(epe_ensemble_sym_valid, cutoff)
    
    # Determine bin ranges (use max across all data)
    max_epe_fwd = max(np.max(epe_oracle_fwd_valid), np.max(epe_ensemble_fwd_valid))
    max_epe_sym = max(np.max(epe_oracle_sym_valid), np.max(epe_ensemble_sym_valid))
    
    # Add config EPEs to max calculation
    for config in results_flat:
        u_fwd = config['u_AB']
        v_fwd = config['v_AB']
        epe_fwd = np.sqrt((u_fwd - u_truth)**2 + (v_fwd - v_truth)**2)
        max_epe_fwd = max(max_epe_fwd, np.max(epe_fwd[valid_mask]))
        
        u_sym = config['u_sym_A']
        v_sym = config['v_sym_A']
        epe_sym = np.sqrt((u_sym - u_truth)**2 + (v_sym - v_truth)**2)
        max_epe_sym = max(max_epe_sym, np.max(epe_sym[valid_mask]))
    
    # Create log-spaced bins
    n_bins = 50
    bins_fwd = np.logspace(np.log10(cutoff), np.log10(max_epe_fwd), n_bins)
    bins_sym = np.logspace(np.log10(cutoff), np.log10(max_epe_sym), n_bins)
    
    # Generate forward flow figure
    output_path_fwd = output_path.parent / (output_path.stem + '_forward' + output_path.suffix)
    generate_oracle_comparison_single_flow(
        epe_oracle_fwd_valid,
        epe_ensemble_fwd_valid,
        ensemble_selection_valid,
        results_flat,
        valid_mask,
        u_truth,
        v_truth,
        'forward',
        cutoff,
        bins_fwd,
        output_path_fwd
    )
    
    # Generate symmetric flow figure
    output_path_sym = output_path.parent / (output_path.stem + '_symmetric' + output_path.suffix)
    generate_oracle_comparison_single_flow(
        epe_oracle_sym_valid,
        epe_ensemble_sym_valid,
        ensemble_selection_valid,
        results_flat,
        valid_mask,
        u_truth,
        v_truth,
        'symmetric',
        cutoff,
        bins_sym,
        output_path_sym
    )


def generate_metric_correlations(sweep_df: pd.DataFrame,
                                 output_path: Path):
    """
    Generate metric correlation heatmap from sweep_df.
    
    This is a simple wrapper that creates a correlation heatmap
    of all numeric columns in sweep_df.
    """
    # Select numeric columns
    numeric_cols = sweep_df.select_dtypes(include=[np.number]).columns
    
    # Compute correlation matrix
    corr_matrix = sweep_df[numeric_cols].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(numeric_cols)))
    ax.set_yticks(np.arange(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=90, ha='right', fontsize=8)
    ax.set_yticklabels(numeric_cols, fontsize=8)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20, fontsize=11)
    
    # Title
    ax.set_title('Metric Correlations', fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_metric_correlations_comparison(
    results_full: list,
    sweep_df: pd.DataFrame,
    u_truth: np.ndarray,
    v_truth: np.ndarray,
    valid_mask: np.ndarray,
    ensemble_selection: np.ndarray,
    output_path: Path
):
    """
    Generate three-panel comparison of metric correlations with EPE.
    
    Left panel: Correlation across 8 config means (config-level)
    Middle panel: Per-pixel correlations using each metric's best config
    Right panel: Per-pixel correlations using ensemble-selected configs
    """
    # Flatten results
    results_flat = _ensure_flattened(results_full)
    
    # Define metrics
    metric_names = [
        'traction_A', 'traction_B',
        'consistency_A', 'consistency_B',
        'photometric_A', 'photometric_B',
        'displacements_sensitivity_A2B', 'displacements_sensitivity_B2A'
    ]
    
    clean_labels = [
        'Traction (A→B)',
        'Traction (B→A)',
        'Consistency (A→B)',
        'Consistency (B→A)',
        'Photometric (A→B)',
        'Photometric (B→A)',
        'Perturbation Sensitivity (A→B)',
        'Perturbation Sensitivity (B→A)',
    ]
    
    # Compute correlations
    config_corr = _compute_config_level_correlations(sweep_df, metric_names)
    pixel_corr_fwd, pixel_corr_sym = _compute_per_pixel_correlations(
        results_flat, u_truth, v_truth, valid_mask, metric_names)
    ensemble_corr_fwd, ensemble_corr_sym = _compute_ensemble_pixel_correlations(
        results_flat, ensemble_selection, u_truth, v_truth, valid_mask, metric_names)
    
    # Create figure
    fig = plt.figure(figsize=(18, 6))
    _plot_correlation_comparison(fig, config_corr, pixel_corr_fwd, pixel_corr_sym,
                                 ensemble_corr_fwd, ensemble_corr_sym, clean_labels)
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_flow_decomposition_figure(
    u_AB: np.ndarray,
    v_AB: np.ndarray,
    u_BA: np.ndarray,
    v_BA: np.ndarray,
    magnitude: int,
    valid_mask: np.ndarray,
    figures_dir: Path,
    flow_label: str = "ensemble"
):
    """
    Generate flow decomposition figures for both directions.
    
    Creates two 1×3 figures:
    - flow_decomposition_AB.png: Forward flow (A→B) analysis
    - flow_decomposition_BA.png: Backward flow (B→A) analysis
    
    Each figure shows: magnitude+quivers | curl | occlusion
    
    Args:
        u_AB, v_AB: Forward flow (A→B)
        u_BA, v_BA: Backward flow (B→A)
        magnitude: Scale for deformation analysis (integer)
        valid_mask: Boolean mask for valid pixels (shows as yellow boundary)
        figures_dir: Output directory for figures
        flow_label: Label for figure title (e.g., "ensemble", "oracle", "best")
    """
    H, W = u_AB.shape
    
    # Compute deformation for forward flow (lives in Frame B)
    deform_AB = compute_flow_deformation(u_AB, v_AB, magnitude)
    
    # Pad curl to match full size for visualization
    curl_AB_padded = np.zeros((H, W), dtype=np.float32)
    ch, cw = deform_AB['curl'].shape
    offset_h = (H - ch) // 2
    offset_w = (W - cw) // 2
    curl_AB_padded[offset_h:offset_h+ch, offset_w:offset_w+cw] = deform_AB['curl']
    
    # Generate forward flow figure
    try:
        # Try with valid_mask parameter (new version)
        fig_AB = plot_flow_decomposition(
            u_AB, v_AB, curl_AB_padded, deform_AB['occlusion'],
            valid_mask=valid_mask,
            title=f"Flow A→B ({flow_label}): magnitude | curl | occlusion",
            output_path=figures_dir / "flow_decomposition_AB.png"
        )
    except TypeError:
        # Fall back without valid_mask (old version)
        fig_AB = plot_flow_decomposition(
            u_AB, v_AB, curl_AB_padded, deform_AB['occlusion'],
            title=f"Flow A→B ({flow_label}): magnitude | curl | occlusion",
            output_path=figures_dir / "flow_decomposition_AB.png"
        )
    plt.close(fig_AB)
    
    # Compute deformation for backward flow (lives in Frame A)
    deform_BA = compute_flow_deformation(u_BA, v_BA, magnitude)
    
    # Pad curl
    curl_BA_padded = np.zeros((H, W), dtype=np.float32)
    ch, cw = deform_BA['curl'].shape
    curl_BA_padded[offset_h:offset_h+ch, offset_w:offset_w+cw] = deform_BA['curl']
    
    # Generate backward flow figure
    try:
        # Try with valid_mask parameter (new version)
        fig_BA = plot_flow_decomposition(
            u_BA, v_BA, curl_BA_padded, deform_BA['occlusion'],
            valid_mask=valid_mask,
            title=f"Flow B→A ({flow_label}): magnitude | curl | occlusion",
            output_path=figures_dir / "flow_decomposition_BA.png"
        )
    except TypeError:
        # Fall back without valid_mask (old version)
        fig_BA = plot_flow_decomposition(
            u_BA, v_BA, curl_BA_padded, deform_BA['occlusion'],
            title=f"Flow B→A ({flow_label}): magnitude | curl | occlusion",
            output_path=figures_dir / "flow_decomposition_BA.png"
        )
    plt.close(fig_BA)


def generate_all_sweep_figures(
    results_full: List[Dict],
    sweep_df: pd.DataFrame,
    frame1: np.ndarray,
    u_truth: np.ndarray,
    v_truth: np.ndarray,
    valid_mask: np.ndarray,
    u_oracle_fwd: np.ndarray,
    v_oracle_fwd: np.ndarray,
    u_oracle_sym: np.ndarray,
    v_oracle_sym: np.ndarray,
    u_best_fwd: np.ndarray,
    v_best_fwd: np.ndarray,
    u_best_sym: np.ndarray,
    v_best_sym: np.ndarray,
    u_ensemble_fwd: np.ndarray,
    v_ensemble_fwd: np.ndarray,
    u_ensemble_sym: np.ndarray,
    v_ensemble_sym: np.ndarray,
    ensemble_selection: np.ndarray,
    figures_dir: Path,
    sweep_params: List[str] = None,
    ensemble_source: str = "fixed",
    u_ensemble_BA: np.ndarray = None,
    v_ensemble_BA: np.ndarray = None,
    magnitude: int = 1
):
    """
    Generate all standard sweep figures.
    
    Args:
        sweep_params: List of swept parameter names to show in EPE distribution.
                     If None, infers from results_full (may include metrics).
        ensemble_source: How ensemble weights were determined ("fixed", "optimized", etc.)
        u_ensemble_BA, v_ensemble_BA: Backward ensemble flow for deformation analysis
        magnitude: Scale for deformation analysis (integer, typically 1-3)
    """
    print("   Generating figures...")
    
    generate_epe_distribution(
        results_full, u_truth, v_truth, valid_mask,
        figures_dir / "epe_distribution.png",
        u_ensemble=u_ensemble_sym, v_ensemble=v_ensemble_sym,
        ensemble_selection=ensemble_selection,
        sweep_params=sweep_params,
        ensemble_label=f"Ensemble ({ensemble_source})"
    )
    print("      ✅ EPE distribution")
    
    generate_oracle_comparison_histograms(
        u_oracle_fwd, v_oracle_fwd,
        u_best_fwd, v_best_fwd,
        u_ensemble_fwd, v_ensemble_fwd,
        u_oracle_sym, v_oracle_sym,
        u_best_sym, v_best_sym,
        u_ensemble_sym, v_ensemble_sym,
        u_truth, v_truth, valid_mask,
        ensemble_selection,
        results_full,
        figures_dir / "oracle_comparison.png"
    )
    print("      ✅ Oracle comparison (forward + symmetric)")
    
    # DISABLED: Metric correlations heatmap is too cluttered and not useful
    # generate_metric_correlations(
    #     sweep_df,
    #     figures_dir / "metric_correlations.png"
    # )
    # print("      ✅ Metric correlations")
    
    generate_metric_correlations_comparison(
        results_full,
        sweep_df,
        u_truth, v_truth, valid_mask,
        ensemble_selection,
        figures_dir / "metric_correlations_comparison.png"
    )
    print("      ✅ Metric correlations comparison (config vs pixel level)")
    
    # Flow decomposition figures
    if u_ensemble_BA is not None and v_ensemble_BA is not None:
        generate_flow_decomposition_figure(
            u_ensemble_fwd, v_ensemble_fwd,
            u_ensemble_BA, v_ensemble_BA,
            magnitude,
            valid_mask,
            figures_dir,
            flow_label=f"ensemble {ensemble_source}"
        )
        print("      ✅ Flow decomposition (AB + BA)")
    else:
        # If no backward flow provided, just do forward
        deform_AB = compute_flow_deformation(u_ensemble_fwd, v_ensemble_fwd, magnitude)
        
        H, W = u_ensemble_fwd.shape
        curl_padded = np.zeros((H, W), dtype=np.float32)
        ch, cw = deform_AB['curl'].shape
        offset = magnitude
        curl_padded[offset:offset+ch, offset:offset+cw] = deform_AB['curl']
        
        try:
            # Try with valid_mask parameter (new version)
            fig = plot_flow_decomposition(
                u_ensemble_fwd, v_ensemble_fwd, curl_padded, deform_AB['occlusion'],
                valid_mask=valid_mask,
                title=f"Flow A→B (ensemble {ensemble_source})",
                output_path=figures_dir / "flow_decomposition_AB.png"
            )
        except TypeError:
            # Fall back without valid_mask (old version)
            fig = plot_flow_decomposition(
                u_ensemble_fwd, v_ensemble_fwd, curl_padded, deform_AB['occlusion'],
                title=f"Flow A→B (ensemble {ensemble_source})",
                output_path=figures_dir / "flow_decomposition_AB.png"
            )
        plt.close(fig)
        print("      ✅ Flow decomposition (AB only)")


if __name__ == "__main__":
    print("✅ Sweep figures module loaded")
