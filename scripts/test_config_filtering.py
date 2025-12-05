# File: scripts/test_config_filtering.py
"""
Test two-stage config filtering:
  Stage 1: Keep top-K configs by mean(depth × perturbation_raw)
  Stage 2: Pixel-wise selection using photometric among survivors

Usage:
    python scripts/test_config_filtering.py data/.../results_full.pkl [--k 3]
"""

import numpy as np
import pickle
import sys
import re
import json
from pathlib import Path


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


def get_config_info(results, pert_dist=1.0):
    """Extract config info with depth-scaled perturbation scores."""
    configs = []
    
    for i, r in enumerate(results):
        config_name = r['metadata'].get('config_name', '')
        match = re.search(r'win(\d+)', config_name)
        winsize = int(match.group(1)) if match else 15
        depth_scale = winsize / 2 + pert_dist
        
        # Mean perturbation (raw)
        pert_raw = np.nanmean(r['metrics']['perturbation_raw_A'])
        pert_scaled = depth_scale * pert_raw
        
        # Mean photometric
        photo = np.nanmean(r['metrics']['photometric_A'])
        
        configs.append({
            'idx': i,
            'name': config_name,
            'winsize': winsize,
            'depth_scale': depth_scale,
            'pert_raw': pert_raw,
            'pert_scaled': pert_scaled,
            'photometric': photo,
        })
    
    return configs


def compute_epe(results, indices, u_truth, v_truth, valid_mask, method='photometric'):
    """
    Compute EPE using pixel-wise selection among specified configs.
    
    Args:
        results: Full results list
        indices: List of config indices to consider
        u_truth, v_truth: Ground truth
        valid_mask: Valid pixel mask
        method: 'photometric' or 'oracle'
    """
    H, W = u_truth.shape
    n_selected = len(indices)
    
    # Build stacks for selected configs only
    u_stack = np.zeros((n_selected, H, W), dtype=np.float32)
    v_stack = np.zeros((n_selected, H, W), dtype=np.float32)
    metric_stack = np.zeros((n_selected, H, W), dtype=np.float32)
    epe_stack = np.zeros((n_selected, H, W), dtype=np.float32)
    
    for j, idx in enumerate(indices):
        r = results[idx]
        u_stack[j] = r['flows']['u_AB']
        v_stack[j] = r['flows']['v_AB']
        metric_stack[j] = r['metrics']['photometric_A']
        epe_stack[j] = np.sqrt((u_stack[j] - u_truth)**2 + (v_stack[j] - v_truth)**2)
    
    if method == 'oracle':
        # Best possible: select config with lowest EPE at each pixel
        selection = np.argmin(epe_stack, axis=0)
    else:
        # Photometric: select config with lowest photometric at each pixel
        selection = np.argmin(metric_stack, axis=0)
    
    # Gather selected EPE values
    y_idx, x_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    selected_epe = epe_stack[selection, y_idx, x_idx]
    
    # Compute mean over valid pixels
    mean_epe = np.nanmean(selected_epe[valid_mask])
    
    return mean_epe, selected_epe


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_config_filtering.py <results_full.pkl> [--k 3]")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    
    # Parse K
    K = 3
    if '--k' in sys.argv:
        k_idx = sys.argv.index('--k')
        K = int(sys.argv[k_idx + 1])
    
    # Load data
    print(f"📂 Loading {results_path}")
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
    
    # Load ground truth
    u_truth, v_truth, valid_mask = load_ground_truth(results_path)
    H, W = u_truth.shape
    print(f"   Shape: {H}x{W}, valid: {valid_mask.sum()}")
    
    # Get config info
    configs = get_config_info(results, pert_dist)
    
    # Compute single-config EPE for reference
    print("\n📊 Computing single-config EPE...")
    for c in configs:
        idx = c['idx']
        r = results[idx]
        u = r['flows']['u_AB']
        v = r['flows']['v_AB']
        epe = np.sqrt((u - u_truth)**2 + (v - v_truth)**2)
        c['epe'] = np.nanmean(epe[valid_mask])
    
    # Sort by EPE (best first)
    configs_by_epe = sorted(configs, key=lambda x: x['epe'])
    
    # Sort by pert_scaled (lowest = best)
    configs_by_pert = sorted(configs, key=lambda x: x['pert_scaled'])
    
    print("\n" + "="*80)
    print(f"CONFIG RANKINGS (top {K})")
    print("="*80)
    
    print(f"\n{'By EPE (ground truth)':^40} | {'By pert×depth (self-supervised)':^40}")
    print("-"*40 + " | " + "-"*40)
    
    for i in range(min(K, n_configs)):
        c_epe = configs_by_epe[i]
        c_pert = configs_by_pert[i]
        print(f"  {i+1}. EPE={c_epe['epe']:.4f} win{c_epe['winsize']:2d}  |  "
              f"  {i+1}. pert={c_pert['pert_scaled']:.4f} win{c_pert['winsize']:2d} (EPE={c_pert['epe']:.4f})")
    
    # Check overlap
    top_k_by_epe = set(c['idx'] for c in configs_by_epe[:K])
    top_k_by_pert = set(c['idx'] for c in configs_by_pert[:K])
    overlap = top_k_by_epe & top_k_by_pert
    print(f"\n   Overlap: {len(overlap)}/{K} configs")
    
    # Test different selection strategies
    print("\n" + "="*80)
    print("EPE COMPARISON")
    print("="*80)
    
    all_indices = list(range(n_configs))
    filtered_indices = [c['idx'] for c in configs_by_pert[:K]]
    oracle_indices = [c['idx'] for c in configs_by_epe[:K]]
    
    strategies = [
        ("All configs (40), photometric", all_indices, 'photometric'),
        ("All configs (40), oracle", all_indices, 'oracle'),
        (f"Top-{K} by pert×depth, photometric", filtered_indices, 'photometric'),
        (f"Top-{K} by pert×depth, oracle", filtered_indices, 'oracle'),
        (f"Top-{K} by EPE (cheating), photometric", oracle_indices, 'photometric'),
        (f"Top-{K} by EPE (cheating), oracle", oracle_indices, 'oracle'),
    ]
    
    baseline_epe = None
    for name, indices, method in strategies:
        epe, _ = compute_epe(results, indices, u_truth, v_truth, valid_mask, method)
        
        if baseline_epe is None:
            baseline_epe = epe
            delta = ""
        else:
            pct = (epe - baseline_epe) / baseline_epe * 100
            delta = f"  ({pct:+.1f}%)"
        
        print(f"   {name:45s}  EPE = {epe:.4f}{delta}")
    
    # Show which configs were selected
    print(f"\n📋 Top-{K} by pert×depth:")
    for c in configs_by_pert[:K]:
        rank_by_epe = next(i for i, x in enumerate(configs_by_epe) if x['idx'] == c['idx']) + 1
        print(f"   {c['name'][:40]:40s}  EPE rank: {rank_by_epe}/{n_configs}")


if __name__ == "__main__":
    main()
