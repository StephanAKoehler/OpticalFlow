#!/usr/bin/env python3
# File: scripts/cycle_consistency.py
"""
Analyze cycle consistency: apply forward flows then backward flows,
measure how close points return to their starting positions.

Usage:
    python scripts/cycle_consistency.py config.toml
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tomli

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.data_loader import load_movie_sequence


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
# Flow Composition
# =============================================================================

def interpolate_flow(u: np.ndarray, v: np.ndarray, 
                     x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate flow field at arbitrary (x, y) positions.
    
    Args:
        u, v: (H, W) flow components
        x, y: (H, W) query positions
        
    Returns:
        u_interp, v_interp: (H, W) interpolated flow
        valid: (H, W) bool mask of in-bounds positions
    """
    H, W = u.shape
    
    # Check bounds
    valid = (x >= 0) & (x < W - 1) & (y >= 0) & (y < H - 1)
    
    # Clamp for interpolation (invalid positions will be masked anyway)
    x_clamped = np.clip(x, 0, W - 1.001).astype(np.float32)
    y_clamped = np.clip(y, 0, H - 1.001).astype(np.float32)
    
    # Use cv2.remap for bilinear interpolation
    map_x = x_clamped.astype(np.float32)
    map_y = y_clamped.astype(np.float32)
    
    u_interp = cv2.remap(u.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)
    v_interp = cv2.remap(v.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)
    
    return u_interp, v_interp, valid


def compose_flow_cycle(
    flows_forward: list[tuple[np.ndarray, np.ndarray]],
    flows_backward: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compose forward flows then backward flows, compute cycle error.
    
    Args:
        flows_forward: [(u_01, v_01), (u_12, v_12), ...] forward direction
        flows_backward: [(u_43, v_43), (u_32, v_32), ...] backward direction
        
    Returns:
        cycle_error: (H, W) magnitude of return displacement
        valid_mask: (H, W) bool mask of valid cycle points
        net_displacement: (H, W, 2) final (u, v) displacement from start
    """
    H, W = flows_forward[0][0].shape
    
    # Initialize positions at pixel grid
    y_init, x_init = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    x_pos = x_init.astype(np.float64)
    y_pos = y_init.astype(np.float64)
    valid = np.ones((H, W), dtype=bool)
    
    # Forward pass
    for u, v in flows_forward:
        u_interp, v_interp, step_valid = interpolate_flow(u, v, x_pos, y_pos)
        valid &= step_valid
        x_pos = x_pos + u_interp
        y_pos = y_pos + v_interp
    
    # Backward pass
    for u, v in flows_backward:
        u_interp, v_interp, step_valid = interpolate_flow(u, v, x_pos, y_pos)
        valid &= step_valid
        x_pos = x_pos + u_interp
        y_pos = y_pos + v_interp
    
    # Compute displacement from start
    dx = x_pos - x_init
    dy = y_pos - y_init
    
    cycle_error = np.sqrt(dx**2 + dy**2)
    cycle_error[~valid] = np.nan
    
    net_displacement = np.stack([dx, dy], axis=-1)
    
    return cycle_error, valid, net_displacement


# =============================================================================
# Config Selection
# =============================================================================

def select_config_index(results_full: list, method: str, weights: dict = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select config index per pixel using specified method.
    
    Args:
        results_full: list of config results with metrics
        method: 'oracle', 'raw_sum', 'raw_max', 'mad_sum', 'mad_max'
        weights: dict with perturbation_rms, consistency, photometric weights
        
    Returns:
        selection: (H, W) int array of selected config indices
        selection_other_winsize: (H, W) int array of best config with different winsize
        penalty_gap: (H, W) penalty difference to next larger winsize (occlusion signal)
    """
    if weights is None:
        weights = {'perturbation_rms': 1.0, 'consistency': 1.0, 'photometric': 1.0}
    
    n_configs = len(results_full)
    H, W = results_full[0]['flows']['u_AB'].shape
    
    # Stack metrics
    pert = np.stack([r['metrics']['displacements_sensitivity_A2B'] for r in results_full], axis=0)
    cons = np.stack([r['metrics']['consistency_A'] for r in results_full], axis=0)
    phot = np.stack([r['metrics']['photometric_A'] for r in results_full], axis=0)
    
    use_mad = 'mad' in method
    use_max = 'max' in method
    
    if use_mad:
        def mad_normalize(arr):
            median = np.median(arr, axis=0, keepdims=True)
            mad = np.median(np.abs(arr - median), axis=0, keepdims=True)
            mad = np.maximum(mad, 1e-10)
            return (arr - median) / mad
        pert = mad_normalize(pert)
        cons = mad_normalize(cons)
        phot = mad_normalize(phot)
    
    w_pert = weights.get('perturbation_rms', 1.0)
    w_cons = weights.get('consistency', 1.0)
    w_phot = weights.get('photometric', 1.0)
    
    if use_max:
        penalty = np.maximum(
            np.maximum(w_pert * pert**2, w_cons * cons**2),
            w_phot * phot**2
        )
    else:
        penalty = w_pert * pert**2 + w_cons * cons**2 + w_phot * phot**2
    
    # Best selection
    selection = np.argmin(penalty, axis=0)
    
    # Get winsize for each config
    winsizes = np.array([r['params'].get('winsize', r['params'].get('preset', 0)) 
                         for r in results_full])
    unique_winsizes = np.unique(winsizes)
    
    # ========================================================================
    # Best with next larger winsize (for spread and penalty gap)
    # ========================================================================
    max_winsize = unique_winsizes.max()
    winner_winsize = winsizes[selection]  # (H, W)
    
    # Get penalty of best config
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    penalty_best = penalty[selection, ii, jj]
    
    # For each pixel, find best config with next larger winsize
    selection_next_larger = np.zeros((H, W), dtype=np.int32)
    penalty_gap = np.zeros((H, W), dtype=np.float32)
    
    for ws in unique_winsizes:
        # Mask for pixels where winner has this winsize
        pixel_mask = (winner_winsize == ws)
        if not np.any(pixel_mask):
            continue
        
        if ws == max_winsize:
            # Largest winsize: use self (spread will be 0)
            config_mask = (winsizes == ws)
            # Find best config among same winsize (will be the winner)
            penalty_same = penalty[config_mask]
            best_same_idx = np.argmin(penalty_same, axis=0)
            # Map back to global config index
            config_indices = np.where(config_mask)[0]
            selection_next_larger[pixel_mask] = config_indices[best_same_idx[pixel_mask]]
            penalty_gap[pixel_mask] = 0.0
        else:
            # Find next larger winsize
            larger = unique_winsizes[unique_winsizes > ws]
            next_larger_ws = larger.min()
            
            # Mask for configs with next larger winsize
            config_mask = (winsizes == next_larger_ws)
            
            # Get minimum penalty among configs with next larger winsize
            penalty_larger = penalty[config_mask]  # (n_configs_larger, H, W)
            best_larger_idx = np.argmin(penalty_larger, axis=0)  # (H, W)
            min_penalty_larger = np.min(penalty_larger, axis=0)  # (H, W)
            
            # Map back to global config index
            config_indices = np.where(config_mask)[0]
            selection_next_larger[pixel_mask] = config_indices[best_larger_idx[pixel_mask]]
            
            # Compute gap for these pixels
            penalty_gap[pixel_mask] = min_penalty_larger[pixel_mask] - penalty_best[pixel_mask]
    
    return selection, selection_next_larger, penalty_gap


def gather_selected_flow(results_full: list, selection: np.ndarray, 
                         direction: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Gather flow for selected configs per pixel.
    
    Args:
        results_full: list of config results
        selection: (H, W) config index per pixel
        direction: 'AB' or 'BA'
        
    Returns:
        u, v: (H, W) selected flow components
    """
    n_configs = len(results_full)
    H, W = selection.shape
    
    u_key = f'u_{direction}'
    v_key = f'v_{direction}'
    
    u_stack = np.stack([r['flows'][u_key] for r in results_full], axis=0)
    v_stack = np.stack([r['flows'][v_key] for r in results_full], axis=0)
    
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    u_selected = u_stack[selection, ii, jj]
    v_selected = v_stack[selection, ii, jj]
    
    return u_selected, v_selected


# =============================================================================
# High-level Analysis
# =============================================================================

def compute_cycle_consistency(
    sweep_dir: Path,
    n_pairs: int,
    method: str,
    weights: dict = None,
) -> dict:
    """
    Compute cycle consistency for a method across all pairs.
    
    Args:
        sweep_dir: path to sweep results
        n_pairs: number of frame pairs
        method: selection method
        weights: optional weight dict
        
    Returns:
        dict with error_map, valid_mask, mean_error, median_error
    """
    # Load all pair results
    all_results = []
    for pair_idx in range(n_pairs):
        pair_dir = sweep_dir / f'pair_{pair_idx:03d}'
        results_path = pair_dir / 'results_full.pkl'
        
        if not results_path.exists():
            print(f"❌ ERROR: Missing {results_path}")
            sys.exit(1)
        
        with open(results_path, 'rb') as f:
            all_results.append(pickle.load(f))
    
    # Build forward and backward flow lists
    flows_forward = []
    flows_backward = []
    spread_maps = []  # Track spread from each pair
    penalty_gap_maps = []  # Track penalty gap (occlusion signal)
    
    for pair_idx, results_full in enumerate(all_results):
        # Select config for this pair
        selection, selection_2nd, penalty_gap = select_config_index(results_full, method, weights)
        
        # Gather flows
        u_ab, v_ab = gather_selected_flow(results_full, selection, 'AB')
        u_ba, v_ba = gather_selected_flow(results_full, selection, 'BA')
        
        # Gather 2nd best flow for spread (best with different winsize)
        u_ab_other, v_ab_other = gather_selected_flow(results_full, selection_2nd, 'AB')
        
        # Compute spread: best vs best-other-winsize
        spread_u = u_ab - u_ab_other
        spread_v = v_ab - v_ab_other
        spread_mag = np.sqrt(spread_u**2 + spread_v**2)
        spread_maps.append(spread_mag)
        penalty_gap_maps.append(penalty_gap)
        
        flows_forward.append((u_ab, v_ab))
        flows_backward.append((u_ba, v_ba))
    
    # Aggregate spread: max across pairs (worst case)
    spread_aggregate = np.maximum.reduce(spread_maps)
    penalty_gap_aggregate = np.maximum.reduce(penalty_gap_maps)
    
    # Reverse backward list for return journey
    flows_backward = flows_backward[::-1]
    
    # Compose cycle
    error_map, valid_mask, net_displacement = compose_flow_cycle(flows_forward, flows_backward)
    
    # Statistics
    valid_errors = error_map[valid_mask]
    
    return {
        'error_map': error_map,
        'valid_mask': valid_mask,
        'net_displacement': net_displacement,
        'spread_map': spread_aggregate,
        'penalty_gap_map': penalty_gap_aggregate,
        'mean_error': float(np.nanmean(valid_errors)) if len(valid_errors) > 0 else np.nan,
        'median_error': float(np.nanmedian(valid_errors)) if len(valid_errors) > 0 else np.nan,
        'valid_fraction': float(valid_mask.sum()) / valid_mask.size,
    }


def invert_flow(u_ab: np.ndarray, v_ab: np.ndarray, 
                n_iters: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute backward flow from forward flow using iterative inverse warping.
    
    Given flow_AB defined at frame A coordinates, compute flow_BA at frame B coordinates.
    
    Args:
        u_ab, v_ab: (H, W) forward flow components
        n_iters: number of iterations for convergence
        
    Returns:
        u_ba, v_ba: (H, W) backward flow at frame B coordinates
        valid: (H, W) bool mask where inverse is well-defined
    """
    H, W = u_ab.shape
    
    # Initialize: for each (x', y') in B, guess source is same position
    y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    x_prime = x_grid.astype(np.float64)
    y_prime = y_grid.astype(np.float64)
    
    # Start with identity guess
    x_src = x_prime.copy()
    y_src = y_prime.copy()
    
    # Iterate to find source position
    # We want: (x_src, y_src) + flow_AB(x_src, y_src) = (x', y')
    # So: (x_src, y_src) = (x', y') - flow_AB(x_src, y_src)
    for _ in range(n_iters):
        # Sample forward flow at current source estimate
        u_at_src, v_at_src, valid = interpolate_flow(u_ab, v_ab, x_src, y_src)
        
        # Update source estimate
        x_src = x_prime - u_at_src
        y_src = y_prime - v_at_src
    
    # Final flow sample at converged source
    u_at_src, v_at_src, valid = interpolate_flow(u_ab, v_ab, x_src, y_src)
    
    # Backward flow is negation of forward flow at source
    u_ba = -u_at_src
    v_ba = -v_at_src
    
    # Also mark invalid where source is out of bounds
    valid &= (x_src >= 0) & (x_src < W - 1) & (y_src >= 0) & (y_src < H - 1)
    
    return u_ba, v_ba, valid


def compute_oracle_cycle_consistency(
    sweep_dir: Path,
    movie,
    n_pairs: int,
) -> dict:
    """
    Compute cycle consistency using ground truth flow (oracle baseline).
    Also computes oracle spread (best EPE vs best EPE with different winsize).
    """
    flows_forward = []
    flows_backward = []
    spread_maps = []
    
    # Track combined validity
    H, W = movie.pairs[0].u_truth.shape
    combined_valid = np.ones((H, W), dtype=bool)
    
    for pair_idx in range(n_pairs):
        pair = movie.pairs[pair_idx]
        
        # Ground truth forward
        u_ab = pair.u_truth.copy()
        v_ab = pair.v_truth.copy()
        
        # Handle NaN in ground truth
        nan_mask = np.isnan(u_ab) | np.isnan(v_ab)
        u_ab_clean = np.nan_to_num(u_ab, nan=0.0)
        v_ab_clean = np.nan_to_num(v_ab, nan=0.0)
        
        # Compute backward flow via inverse warp
        u_ba, v_ba, warp_valid = invert_flow(u_ab_clean, v_ab_clean, n_iters=5)
        
        # Mark invalid where GT was NaN
        combined_valid &= ~nan_mask & warp_valid
        
        flows_forward.append((u_ab_clean, v_ab_clean))
        flows_backward.append((u_ba, v_ba))
        
        # ====================================================================
        # Compute oracle spread for this pair
        # ====================================================================
        pair_dir = sweep_dir / f'pair_{pair_idx:03d}'
        results_path = pair_dir / 'results_full.pkl'
        
        if results_path.exists():
            with open(results_path, 'rb') as f:
                results_full = pickle.load(f)
            
            n_configs = len(results_full)
            valid_mask_pair = pair.valid_mask if hasattr(pair, 'valid_mask') else ~nan_mask
            
            # Stack flows and compute EPE for each config
            u_stack = np.stack([r['flows']['u_AB'] for r in results_full], axis=0)
            v_stack = np.stack([r['flows']['v_AB'] for r in results_full], axis=0)
            
            # EPE per config per pixel
            epe_stack = np.sqrt((u_stack - u_ab)**2 + (v_stack - v_ab)**2)
            
            # Best config per pixel
            oracle_selection = np.argmin(epe_stack, axis=0)
            
            # Get winsize for each config
            winsizes = np.array([r['params'].get('winsize', r['params'].get('preset', 0)) 
                                 for r in results_full])
            unique_winsizes = np.unique(winsizes)
            
            if len(unique_winsizes) <= 1:
                # Fall back to 2nd best
                sorted_indices = np.argsort(epe_stack, axis=0)
                oracle_selection_other = sorted_indices[1]
            else:
                # Best with next larger winsize
                max_winsize = unique_winsizes.max()
                winner_winsize = winsizes[oracle_selection]
                oracle_selection_other = np.zeros((H, W), dtype=np.int32)
                
                for ws in unique_winsizes:
                    pixel_mask = (winner_winsize == ws)
                    if not np.any(pixel_mask):
                        continue
                    
                    if ws == max_winsize:
                        # Largest winsize: use self (spread will be 0)
                        config_mask = (winsizes == ws)
                        epe_same = epe_stack[config_mask]
                        best_same_idx = np.argmin(epe_same, axis=0)
                        config_indices = np.where(config_mask)[0]
                        oracle_selection_other[pixel_mask] = config_indices[best_same_idx[pixel_mask]]
                    else:
                        # Find next larger winsize
                        larger = unique_winsizes[unique_winsizes > ws]
                        next_larger_ws = larger.min()
                        
                        config_mask = (winsizes == next_larger_ws)
                        epe_larger = epe_stack[config_mask]
                        best_larger_idx = np.argmin(epe_larger, axis=0)
                        config_indices = np.where(config_mask)[0]
                        oracle_selection_other[pixel_mask] = config_indices[best_larger_idx[pixel_mask]]
            
            # Gather flows for best and best-other-winsize
            ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            u_best = u_stack[oracle_selection, ii, jj]
            v_best = v_stack[oracle_selection, ii, jj]
            u_other = u_stack[oracle_selection_other, ii, jj]
            v_other = v_stack[oracle_selection_other, ii, jj]
            
            # Oracle spread
            spread_mag = np.sqrt((u_best - u_other)**2 + (v_best - v_other)**2)
            spread_maps.append(spread_mag)
        else:
            # No results, zero spread
            spread_maps.append(np.zeros((H, W), dtype=np.float32))
    
    # Aggregate spread: max across pairs
    spread_aggregate = np.maximum.reduce(spread_maps) if spread_maps else np.zeros((H, W))
    
    flows_backward = flows_backward[::-1]
    
    error_map, valid_mask, net_displacement = compose_flow_cycle(flows_forward, flows_backward)
    
    # Combine validity
    valid_mask &= combined_valid
    error_map[~valid_mask] = np.nan
    
    valid_errors = error_map[valid_mask]
    
    return {
        'error_map': error_map,
        'valid_mask': valid_mask,
        'net_displacement': net_displacement,
        'spread_map': spread_aggregate,
        'mean_error': float(np.nanmean(valid_errors)) if len(valid_errors) > 0 else np.nan,
        'median_error': float(np.nanmedian(valid_errors)) if len(valid_errors) > 0 else np.nan,
        'valid_fraction': float(valid_mask.sum()) / valid_mask.size,
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_cycle_consistency(
    results: dict,
    output_path: Path,
    first_frame: np.ndarray = None,
    last_frame: np.ndarray = None,
):
    """
    Plot cycle error heatmaps and histograms.
    
    Layout:
        Row 0: frame 0 | frame N-1 | overlaid step histograms
        Row 1: oracle | raw_max | mad_max cycle error heatmaps
        Row 2: spread heatmaps (flow difference to other winsize)
        Row 3: penalty gap heatmaps (occlusion signal)
    
    Args:
        results: dict mapping method_name -> result dict
        output_path: where to save figure
        first_frame: first frame image
        last_frame: last frame image
    """
    # Methods to show (in order)
    methods_to_show = ['oracle', 'raw_max', 'mad_max']
    methods_to_show = [m for m in methods_to_show if m in results]
    
    # Determine shared colorbar range for heatmaps
    vmax_error = 0
    for method in methods_to_show:
        res = results[method]
        valid_errors = res['error_map'][res['valid_mask']]
        if len(valid_errors) > 0:
            vmax_error = max(vmax_error, np.percentile(valid_errors, 95))
    vmax_error = max(vmax_error, 0.1)
    
    # Spread colorbar range
    vmax_spread = 0
    for method in methods_to_show:
        if 'spread_map' in results[method]:
            spread = results[method]['spread_map']
            valid_spread = spread[results[method]['valid_mask']]
            if len(valid_spread) > 0:
                vmax_spread = max(vmax_spread, np.percentile(valid_spread, 95))
    vmax_spread = max(vmax_spread, 0.1)
    
    # Create figure
    fig = plt.figure(figsize=(12, 14))
    
    # GridSpec: 4 rows, 3 columns
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1.2, 1.2, 1.2], hspace=0.25, wspace=0.2)
    
    # Top left: first frame
    ax_frame0 = fig.add_subplot(gs[0, 0])
    if first_frame is not None:
        ax_frame0.imshow(first_frame)
        ax_frame0.set_title('Frame 0', fontsize=11)
    else:
        ax_frame0.text(0.5, 0.5, 'No image', ha='center', va='center', transform=ax_frame0.transAxes)
    ax_frame0.set_xticks([])
    ax_frame0.set_yticks([])
    
    # Top middle: last frame
    ax_frame_last = fig.add_subplot(gs[0, 1])
    if last_frame is not None:
        ax_frame_last.imshow(last_frame)
        ax_frame_last.set_title('Frame N-1', fontsize=11)
    else:
        ax_frame_last.text(0.5, 0.5, 'No image', ha='center', va='center', transform=ax_frame_last.transAxes)
    ax_frame_last.set_xticks([])
    ax_frame_last.set_yticks([])
    
    # Top right: overlaid histograms
    ax_hist = fig.add_subplot(gs[0, 2])
    
    # Collect errors and build shared bins
    all_errors = []
    for method in methods_to_show:
        res = results[method]
        valid_errors = res['error_map'][res['valid_mask']]
        all_errors.extend(valid_errors[valid_errors > 0])
    
    if len(all_errors) > 0:
        min_err = max(1e-4, np.min(all_errors))
        max_err = np.max(all_errors)
        min_power = int(np.floor(np.log2(min_err)))
        max_power = int(np.ceil(np.log2(max_err)))
        bin_edges = np.array([0] + [2**p for p in range(min_power, max_power + 1)])
    else:
        bin_edges = np.array([0, 0.01, 0.1, 1, 10])
    
    # Colors for each method
    colors = {'oracle': 'black', 'raw_max': 'tab:orange', 'mad_max': 'tab:cyan'}
    
    for method in methods_to_show:
        res = results[method]
        valid_errors = res['error_map'][res['valid_mask']]
        
        # Histogram
        counts, _ = np.histogram(valid_errors, bins=bin_edges)
        
        # Replace 0 with nan for proper log display (don't show as 1)
        counts_plot = counts.astype(float)
        counts_plot[counts_plot == 0] = np.nan
        
        # Step plot (log-log)
        ax_hist.step(bin_edges[:-1], counts_plot, where='post', 
                    color=colors.get(method, 'gray'), linewidth=2, label=method)
    
    ax_hist.set_xscale('log')
    ax_hist.set_yscale('log')
    ax_hist.set_xlabel('Cycle Error (pixels)', fontsize=11)
    ax_hist.set_ylabel('Count', fontsize=11)
    ax_hist.legend(fontsize=10, loc='lower left')
    ax_hist.set_title('Error Distribution', fontsize=11)
    ax_hist.grid(True, alpha=0.3)
    
    # Set x-axis limits to show full range
    ax_hist.set_xlim(bin_edges[0] + 1e-5, bin_edges[-1])
    
    # Row 1: 3 cycle error heatmaps
    axes_error = [fig.add_subplot(gs[1, i]) for i in range(3)]
    
    for ax, method in zip(axes_error, methods_to_show):
        res = results[method]
        error_map = res['error_map'].copy()
        error_map[~res['valid_mask']] = np.nan
        
        im_error = ax.imshow(error_map, cmap='hot', vmin=0, vmax=vmax_error)
        ax.set_title(f"{method}\nmean={res['mean_error']:.3f}, med={res['median_error']:.3f}",
                    fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Colorbar for cycle error
    cbar_ax1 = fig.add_axes([0.92, 0.52, 0.015, 0.18])
    cbar1 = fig.colorbar(im_error, cax=cbar_ax1)
    cbar1.set_label('Cycle Error (px)', fontsize=10)
    
    # Row 2: 3 spread heatmaps
    axes_spread = [fig.add_subplot(gs[2, i]) for i in range(3)]
    
    for ax, method in zip(axes_spread, methods_to_show):
        res = results[method]
        if 'spread_map' in res:
            spread_map = res['spread_map'].copy()
            spread_map[~res['valid_mask']] = np.nan
            mean_spread = np.nanmean(spread_map)
            
            im_spread = ax.imshow(spread_map, cmap='viridis', vmin=0, vmax=vmax_spread)
            ax.set_title(f"{method} spread\nmean={mean_spread:.3f}", fontsize=10)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{method} spread\n(N/A)', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Colorbar for spread
    cbar_ax2 = fig.add_axes([0.92, 0.31, 0.015, 0.18])
    if vmax_spread > 0:
        sm_spread = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, vmax_spread))
        cbar2 = fig.colorbar(sm_spread, cax=cbar_ax2)
        cbar2.set_label('Spread (px)', fontsize=10)
    
    # Row 3: 3 penalty gap heatmaps (occlusion signal) - log scale
    axes_gap = [fig.add_subplot(gs[3, i]) for i in range(3)]
    
    from matplotlib.colors import LogNorm
    
    # Fixed range for log scale (focus on sensible values)
    vmin_gap = 1e-4
    vmax_gap = 1.0
    
    for ax, method in zip(axes_gap, methods_to_show):
        res = results[method]
        if 'penalty_gap_map' in res:
            gap_map = res['penalty_gap_map'].copy()
            gap_map[~res['valid_mask']] = np.nan
            gap_map[gap_map <= 0] = np.nan  # Can't log zero/negative
            mean_gap = np.nanmean(res['penalty_gap_map'][res['valid_mask']])
            
            im_gap = ax.imshow(gap_map, cmap='plasma', norm=LogNorm(vmin=vmin_gap, vmax=vmax_gap))
            ax.set_title(f"{method} penalty gap\nmean={mean_gap:.4f}", fontsize=10)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{method} penalty gap\n(N/A)', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Colorbar for penalty gap (log scale)
    cbar_ax3 = fig.add_axes([0.92, 0.10, 0.015, 0.18])
    sm_gap = plt.cm.ScalarMappable(cmap='plasma', norm=LogNorm(vmin=vmin_gap, vmax=vmax_gap))
    cbar3 = fig.colorbar(sm_gap, cax=cbar_ax3)
    cbar3.set_label('Penalty Gap (log)', fontsize=10)
    
    fig.suptitle('Cycle Consistency: Forward + Backward Flow Error', 
                fontsize=14, fontweight='bold')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 {output_path}")


# =============================================================================
# Main
# =============================================================================

def run_cycle_analysis(config: dict, data_dir: Path = Path('data')):
    """Run cycle consistency analysis."""
    print("=" * 60)
    print("🔄 CYCLE CONSISTENCY ANALYSIS")
    print("=" * 60)
    
    # Auto-detect experiment
    movie_hash, of_hash = auto_detect_experiment(data_dir)
    
    # Paths
    movie_dir = data_dir / movie_hash
    analysis_dir = movie_dir / 'analysis' / of_hash
    sweep_dir = analysis_dir / 'sweep'
    figures_dir = analysis_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    print(f"Experiment: {analysis_dir}")
    
    # Load movie
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        movie = load_movie_sequence(movie_dir)
    n_pairs = len(movie.pairs)
    
    print(f"Pairs: {n_pairs}")
    
    # Methods to compare
    methods = ['raw_max', 'mad_max']
    weights = {'perturbation_rms': 1.0, 'consistency': 1.0, 'photometric': 1.0}
    
    results = {}
    
    # Oracle (ground truth)
    print("Computing oracle cycle consistency...")
    results['oracle'] = compute_oracle_cycle_consistency(sweep_dir, movie, n_pairs)
    print(f"   oracle: mean={results['oracle']['mean_error']:.4f}, valid={results['oracle']['valid_fraction']:.1%}")
    
    # Each method
    for method in methods:
        print(f"Computing {method} cycle consistency...")
        results[method] = compute_cycle_consistency(sweep_dir, n_pairs, method, weights)
        print(f"   {method}: mean={results[method]['mean_error']:.4f}, valid={results[method]['valid_fraction']:.1%}")
    
    # Get first and last frames for plot
    first_frame = movie.pairs[0].frame1
    last_frame = movie.pairs[-1].frame2
    
    # Plot
    output_path = figures_dir / 'cycle_consistency.png'
    plot_cycle_consistency(results, output_path, first_frame=first_frame, last_frame=last_frame)
    
    print("=" * 60)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze cycle consistency')
    parser.add_argument('config', type=Path, help='TOML config file')
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                       help='Base data directory (default: data/)')
    
    args = parser.parse_args()
    
    if not args.config.exists():
        print(f"❌ ERROR: Config file not found: {args.config}")
        sys.exit(1)
    
    with open(args.config, 'rb') as f:
        config = tomli.load(f)
    
    run_cycle_analysis(config, args.data_dir)


if __name__ == "__main__":
    main()
