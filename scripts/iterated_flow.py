# File: scripts/iterated_flow.py
"""
Iterate optical flow A→B→A→B→A→B and measure accumulated folding and drift.

Bad configs accumulate errors; good configs stay stable.
"""

import numpy as np
import cv2
import pickle
import sys
from pathlib import Path


def warp_flow(flow_u, flow_v, disp_u, disp_v):
    """
    Sample flow at displaced positions.
    
    Args:
        flow_u, flow_v: Flow field to sample
        disp_u, disp_v: Current accumulated displacement
    
    Returns:
        flow sampled at (x + disp_u, y + disp_v), NaN for out-of-bounds
    """
    H, W = flow_u.shape
    
    # Create sampling coordinates
    y_coords, x_coords = np.mgrid[0:H, 0:W].astype(np.float32)
    
    # Offset by displacement
    map_x = x_coords + disp_u
    map_y = y_coords + disp_v
    
    # Sample with border handling
    sampled_u = cv2.remap(flow_u, map_x, map_y, 
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=np.nan)
    sampled_v = cv2.remap(flow_v, map_x, map_y,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=np.nan)
    
    return sampled_u, sampled_v


def compute_folding(u, v, valid):
    """Compute fraction of valid pixels with det(J) < 0."""
    if not valid.any():
        return 1.0
    
    du_dx = np.gradient(u, axis=1)
    du_dy = np.gradient(u, axis=0)
    dv_dx = np.gradient(v, axis=1)
    dv_dy = np.gradient(v, axis=0)
    
    # det(I + J) = (1 + du/dx)(1 + dv/dy) - du/dy * dv/dx
    det_J = (1 + du_dx) * (1 + dv_dy) - du_dy * dv_dx
    
    return float((det_J[valid] < 0).mean())


def iterate_flow(u_AB, v_AB, u_BA, v_BA, valid_mask, n_round_trips=3):
    """
    Iterate flow A→B→A repeatedly and measure accumulated folding and drift.
    
    Each round trip: A→B→A (should return to origin if flows are perfect)
    
    Args:
        u_AB, v_AB: Forward flow A→B
        u_BA, v_BA: Backward flow B→A  
        valid_mask: Initial valid pixel mask
        n_round_trips: Number of A→B→A round trips (default 3)
    
    Returns:
        dict with:
            - final_folding: fraction of valid pixels with det(J) < 0
            - mean_drift: mean magnitude of accumulated displacement (should be ~0)
            - valid_fraction: fraction of pixels still valid after all iterations
    """
    H, W = u_AB.shape
    
    # Accumulated displacement from origin
    u_acc = np.zeros((H, W), dtype=np.float32)
    v_acc = np.zeros((H, W), dtype=np.float32)
    
    # Track valid pixels through iterations
    valid = valid_mask.copy()
    initial_valid_count = valid.sum()
    
    for trip in range(n_round_trips):
        # ============================================================
        # Step 1: A → B
        # ============================================================
        u_AB_here, v_AB_here = warp_flow(u_AB, v_AB, u_acc, v_acc)
        
        # Update valid mask (NaN from warping = out of bounds)
        step_valid = ~np.isnan(u_AB_here) & ~np.isnan(v_AB_here)
        valid = valid & step_valid
        
        # Accumulate displacement
        u_acc = np.where(valid, u_acc + u_AB_here, np.nan)
        v_acc = np.where(valid, v_acc + v_AB_here, np.nan)
        
        # ============================================================
        # Step 2: B → A
        # ============================================================
        u_BA_here, v_BA_here = warp_flow(u_BA, v_BA, u_acc, v_acc)
        
        # Update valid mask
        step_valid = ~np.isnan(u_BA_here) & ~np.isnan(v_BA_here)
        valid = valid & step_valid
        
        # Accumulate displacement
        u_acc = np.where(valid, u_acc + u_BA_here, np.nan)
        v_acc = np.where(valid, v_acc + v_BA_here, np.nan)
    
    # ================================================================
    # Compute final metrics over valid pixels
    # ================================================================
    
    if not valid.any():
        return {
            'final_folding': 1.0,
            'mean_drift': np.inf,
            'valid_fraction': 0.0,
        }
    
    # Folding: det(J) < 0 in accumulated flow
    final_folding = compute_folding(u_acc, v_acc, valid)
    
    # Drift: magnitude of accumulated displacement (should be ~0)
    drift_magnitude = np.sqrt(u_acc**2 + v_acc**2)
    mean_drift = float(np.nanmean(drift_magnitude[valid]))
    
    # Valid fraction (how many pixels survived all iterations)
    valid_fraction = float(valid.sum() / initial_valid_count) if initial_valid_count > 0 else 0.0
    
    return {
        'final_folding': final_folding,
        'mean_drift': mean_drift,
        'valid_fraction': valid_fraction,
    }


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
        return None, None, None
    
    u_data = np.load(u_path)
    v_data = np.load(v_path)
    u_truth = u_data[list(u_data.keys())[0]]
    v_truth = v_data[list(v_data.keys())[0]]
    
    valid_mask = ~np.isnan(u_truth) & ~np.isnan(v_truth)
    
    return u_truth, v_truth, valid_mask


def get_sequence_name(results_path: Path) -> str:
    """Extract sequence name from path."""
    # Path: data/{movie_hash}/analysis/{of_hash}/sweep/pair_000/results_full.pkl
    movie_dir = results_path.parent.parent.parent.parent.parent
    
    # Check source_info.json (standard location)
    source_info = movie_dir / 'source_info.json'
    if source_info.exists():
        import json
        with open(source_info) as f:
            info = json.load(f)
        return info.get('sequence', movie_dir.name)
    
    return movie_dir.name


def analyze_sequence(results_path: Path, n_round_trips: int = 3) -> dict:
    """
    Analyze one sequence with iterated flow metrics.
    """
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    # Load ground truth
    u_truth, v_truth, valid_mask = load_ground_truth(results_path)
    if u_truth is None:
        print(f"      ⚠️  No ground truth found")
        return None
    
    # Handle shape mismatch (results may be downsampled)
    sample_u = results[0]['flows']['u_AB']
    if sample_u.shape != u_truth.shape:
        # Resize ground truth and mask to match results
        from scipy.ndimage import zoom
        scale_y = sample_u.shape[0] / u_truth.shape[0]
        scale_x = sample_u.shape[1] / u_truth.shape[1]
        
        u_truth = zoom(u_truth, (scale_y, scale_x), order=1) * scale_x
        v_truth = zoom(v_truth, (scale_y, scale_x), order=1) * scale_y
        valid_mask = zoom(valid_mask.astype(float), (scale_y, scale_x), order=0) > 0.5
    
    n_configs = len(results)
    config_data = []
    
    for i, r in enumerate(results):
        config_name = r['metadata'].get('config_name', f'config_{i}')
        
        # Get flows
        u_AB = r['flows']['u_AB']
        v_AB = r['flows']['v_AB']
        u_BA = r['flows']['u_BA']
        v_BA = r['flows']['v_BA']
        
        # Compute EPE
        epe = np.sqrt((u_AB - u_truth)**2 + (v_AB - v_truth)**2)
        mean_epe = float(epe[valid_mask].mean())
        
        # Compute single-step folding (what we had before)
        single_folding = compute_folding(u_AB, v_AB, valid_mask)
        
        # Compute iterated metrics
        iter_metrics = iterate_flow(u_AB, v_AB, u_BA, v_BA, valid_mask, n_round_trips)
        
        config_data.append({
            'idx': i,
            'config': config_name,
            'mean_epe': mean_epe,
            'single_folding': single_folding,
            'iter_folding': iter_metrics['final_folding'],
            'iter_drift': iter_metrics['mean_drift'],
            'iter_valid_frac': iter_metrics['valid_fraction'],
        })
    
    # Build EPE rank lookup
    sorted_by_epe = sorted(config_data, key=lambda x: x['mean_epe'])
    epe_rank = {cfg['idx']: rank for rank, cfg in enumerate(sorted_by_epe, start=1)}
    
    # Compute ranks for iterated metrics
    sorted_by_iter_fold = sorted(config_data, key=lambda x: x['iter_folding'])
    sorted_by_iter_drift = sorted(config_data, key=lambda x: x['iter_drift'])
    sorted_by_single_fold = sorted(config_data, key=lambda x: x['single_folding'])
    
    # Find best config by each metric
    best_by_iter_fold = sorted_by_iter_fold[0]
    best_by_iter_drift = sorted_by_iter_drift[0]
    best_by_single_fold = sorted_by_single_fold[0]
    
    return {
        'n_configs': n_configs,
        'best_epe': sorted_by_epe[0]['mean_epe'],
        'best_config': sorted_by_epe[0]['config'],
        'config_data': config_data,
        'epe_rank': epe_rank,
        'results': {
            'single_folding': {
                'winner_idx': best_by_single_fold['idx'],
                'winner_epe_rank': epe_rank[best_by_single_fold['idx']],
                'winner_value': best_by_single_fold['single_folding'],
            },
            'iter_folding': {
                'winner_idx': best_by_iter_fold['idx'],
                'winner_epe_rank': epe_rank[best_by_iter_fold['idx']],
                'winner_value': best_by_iter_fold['iter_folding'],
            },
            'iter_drift': {
                'winner_idx': best_by_iter_drift['idx'],
                'winner_epe_rank': epe_rank[best_by_iter_drift['idx']],
                'winner_value': best_by_iter_drift['iter_drift'],
            },
        }
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python iterated_flow.py <data_dir> [n_round_trips]")
        print("  n_round_trips: number of A→B→A cycles (default 3)")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    n_round_trips = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    if not data_dir.is_dir():
        print(f"❌ Not a directory: {data_dir}")
        sys.exit(1)
    
    # Find all results files
    results_files = list(data_dir.glob('*/analysis/*/sweep/pair_*/results_full.pkl'))
    
    if not results_files:
        print(f"❌ No results found in {data_dir}")
        sys.exit(1)
    
    print(f"📂 Found {len(results_files)} result files")
    print(f"🔄 Using {n_round_trips} round trips (A→B→A × {n_round_trips})")
    print()
    
    # Analyze each sequence
    all_results = {}
    
    for results_path in sorted(results_files):
        seq_name = get_sequence_name(results_path)
        if seq_name in all_results:
            continue
        
        print(f"   📊 {seq_name}...")
        result = analyze_sequence(results_path, n_round_trips)
        
        if result is not None:
            all_results[seq_name] = result
            print(f"      ✓ n={result['n_configs']}")
    
    if not all_results:
        print("❌ No valid results")
        sys.exit(1)
    
    # ==========================================================================
    # Print results
    # ==========================================================================
    
    print()
    print("=" * 90)
    print("ITERATED FLOW ANALYSIS: EPE RANK OF CONFIG SELECTED BY EACH METRIC")
    print("=" * 90)
    
    metrics = ['single_folding', 'iter_folding', 'iter_drift']
    
    # Header
    header = f"{'Sequence':<15}"
    for m in metrics:
        header += f" | {m:>18}"
    print(header)
    print("-" * 90)
    
    # Per-sequence
    for seq_name in sorted(all_results.keys()):
        result = all_results[seq_name]
        row = f"{seq_name:<15}"
        for m in metrics:
            rank = result['results'][m]['winner_epe_rank']
            marker = "✓" if rank <= 5 else " "
            row += f" | {rank:>17}{marker}"
        print(row)
    
    # Aggregate
    print("-" * 90)
    
    # Median
    row_median = f"{'Median':<15}"
    for m in metrics:
        ranks = [all_results[s]['results'][m]['winner_epe_rank'] for s in all_results]
        row_median += f" | {np.median(ranks):>18.0f}"
    print(row_median)
    
    # Rank <= 5
    row_top5 = f"{'Rank ≤ 5':<15}"
    for m in metrics:
        ranks = [all_results[s]['results'][m]['winner_epe_rank'] for s in all_results]
        count = sum(1 for r in ranks if r <= 5)
        row_top5 += f" | {count:>18}"
    print(row_top5)
    
    # Worst
    row_worst = f"{'Worst':<15}"
    for m in metrics:
        ranks = [all_results[s]['results'][m]['winner_epe_rank'] for s in all_results]
        row_worst += f" | {max(ranks):>18}"
    print(row_worst)
    
    print("=" * 90)


if __name__ == "__main__":
    main()
