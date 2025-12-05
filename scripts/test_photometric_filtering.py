# File: scripts/test_photometric_filtering.py
"""
Test two-stage ensemble selection using photometric filtering.

Stage 1: Filter to top-K configs by mean(photo_log_raw) over image
Stage 2: Per-pixel selection by photo_log_raw among top-K

Compare filtering by photometric-mean vs perturbation×depth.

Usage:
    python scripts/test_photometric_filtering.py data/.../results_full.pkl [--k 5]
"""

import numpy as np
import pickle
import sys
import re
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
    u_truth = u_data[list(u_data.keys())[0]]
    v_truth = v_data[list(v_data.keys())[0]]
    
    valid_mask = (
        ~np.isnan(u_truth) & ~np.isnan(v_truth) &
        (np.abs(u_truth) < 1e8) & (np.abs(v_truth) < 1e8)
    )
    
    return u_truth, v_truth, valid_mask


def compute_ensemble_epe(config_indices, photo_stack, u_stack, v_stack, 
                         u_truth, v_truth, valid_mask):
    """
    Compute EPE for ensemble using per-pixel photometric selection.
    
    For each pixel, select config with lowest photo_log_raw among given indices.
    """
    H, W = u_truth.shape
    n_candidates = len(config_indices)
    
    # Extract stacks for candidate configs
    photo_candidates = photo_stack[config_indices]  # (n_candidates, H, W)
    u_candidates = u_stack[config_indices]
    v_candidates = v_stack[config_indices]
    
    # Per-pixel selection: argmin of photometric
    best_local_idx = np.argmin(photo_candidates, axis=0)  # (H, W) indices into candidates
    
    # Gather selected flows
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    u_selected = u_candidates[best_local_idx, y_idx, x_idx]
    v_selected = v_candidates[best_local_idx, y_idx, x_idx]
    
    # Compute EPE
    epe = np.sqrt((u_selected - u_truth)**2 + (v_selected - v_truth)**2)
    mean_epe = np.nanmean(epe[valid_mask])
    
    return mean_epe, best_local_idx


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_photometric_filtering.py <results_full.pkl> [--k 5]")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    
    # Parse K
    K = 5
    if '--k' in sys.argv:
        k_idx = sys.argv.index('--k')
        K = int(sys.argv[k_idx + 1])
    
    print(f"📂 Loading {results_path}")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    n_configs = len(results)
    print(f"   {n_configs} configurations, K={K}")
    
    # Load ground truth
    u_truth, v_truth, valid_mask = load_ground_truth(results_path)
    H, W = u_truth.shape
    print(f"   Shape: {H}×{W}, valid: {valid_mask.sum()}")
    
    # Build stacks
    print("\n📊 Building data stacks...")
    
    u_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    v_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    epe_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    photo_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    config_data = []
    
    for i, r in enumerate(results):
        config_name = r['metadata'].get('config_name', '')
        match = re.search(r'win(\d+)', config_name)
        winsize = int(match.group(1)) if match else 15
        
        u = r['flows']['u_AB']
        v = r['flows']['v_AB']
        photo_raw = r['metrics']['photo_log_raw_A']
        pert_raw = r['metrics']['perturbation_raw_A']
        
        u_stack[i] = u
        v_stack[i] = v
        photo_stack[i] = photo_raw
        epe_stack[i] = np.sqrt((u - u_truth)**2 + (v - v_truth)**2)
        
        depth = winsize / 2 + 1.0
        
        config_data.append({
            'idx': i,
            'name': config_name,
            'winsize': winsize,
            'mean_photo': np.nanmean(photo_raw[valid_mask]),
            'mean_pert_depth': np.nanmean(pert_raw[valid_mask]) * depth,
            'mean_epe': np.nanmean(epe_stack[i][valid_mask]),
        })
    
    # Sort configs by different criteria
    by_photo_mean = sorted(config_data, key=lambda x: x['mean_photo'])
    by_pert_depth = sorted(config_data, key=lambda x: x['mean_pert_depth'])
    by_epe = sorted(config_data, key=lambda x: x['mean_epe'])
    
    # Get index sets
    all_indices = list(range(n_configs))
    top_k_photo = [c['idx'] for c in by_photo_mean[:K]]
    top_k_pert = [c['idx'] for c in by_pert_depth[:K]]
    top_k_oracle = [c['idx'] for c in by_epe[:K]]
    
    print("\n📋 Top configs by each criterion:")
    print(f"\n   By photo-mean (top-{K}):")
    for c in by_photo_mean[:K]:
        print(f"      {c['name'][:35]}: photo={c['mean_photo']:.4f}, EPE={c['mean_epe']:.4f}")
    
    print(f"\n   By pert×depth (top-{K}):")
    for c in by_pert_depth[:K]:
        print(f"      {c['name'][:35]}: pert×d={c['mean_pert_depth']:.4f}, EPE={c['mean_epe']:.4f}")
    
    print(f"\n   By EPE/oracle (top-{K}):")
    for c in by_epe[:K]:
        print(f"      {c['name'][:35]}: EPE={c['mean_epe']:.4f}")
    
    # Compute overlap
    photo_oracle_overlap = len(set(top_k_photo) & set(top_k_oracle))
    pert_oracle_overlap = len(set(top_k_pert) & set(top_k_oracle))
    photo_pert_overlap = len(set(top_k_photo) & set(top_k_pert))
    
    print(f"\n   Overlap: photo∩oracle={photo_oracle_overlap}/{K}, "
          f"pert∩oracle={pert_oracle_overlap}/{K}, "
          f"photo∩pert={photo_pert_overlap}/{K}")
    
    # Compute ensemble EPEs
    print("\n📊 Computing ensemble EPEs...")
    
    # Oracle (per-pixel best from all)
    oracle_epe = np.nanmean(np.min(epe_stack, axis=0)[valid_mask])
    
    # All configs + photometric per-pixel
    epe_all_photo, _ = compute_ensemble_epe(
        all_indices, photo_stack, u_stack, v_stack, u_truth, v_truth, valid_mask)
    
    # Top-K by photo-mean + photometric per-pixel
    epe_topk_photo, _ = compute_ensemble_epe(
        top_k_photo, photo_stack, u_stack, v_stack, u_truth, v_truth, valid_mask)
    
    # Top-K by pert×depth + photometric per-pixel
    epe_topk_pert, _ = compute_ensemble_epe(
        top_k_pert, photo_stack, u_stack, v_stack, u_truth, v_truth, valid_mask)
    
    # Top-K by oracle + photometric per-pixel (upper bound for filtering)
    epe_topk_oracle, _ = compute_ensemble_epe(
        top_k_oracle, photo_stack, u_stack, v_stack, u_truth, v_truth, valid_mask)
    
    # Best single config
    best_single_epe = min(c['mean_epe'] for c in config_data)
    
    # Print results
    print("\n" + "=" * 75)
    print(f"ENSEMBLE SELECTION RESULTS (K={K})")
    print("=" * 75)
    print(f"{'Method':<45} | {'EPE':>8} | {'Δ oracle':>10}")
    print("-" * 75)
    print(f"{'Oracle (per-pixel best)':<45} | {oracle_epe:>8.4f} | {'-':>10}")
    print(f"{'Best single config':<45} | {best_single_epe:>8.4f} | {best_single_epe - oracle_epe:>+10.4f}")
    print("-" * 75)
    print(f"{'All {0} + photo per-pixel'.format(n_configs):<45} | {epe_all_photo:>8.4f} | {epe_all_photo - oracle_epe:>+10.4f}")
    print(f"{'Top-{0} by photo-mean + photo per-pixel'.format(K):<45} | {epe_topk_photo:>8.4f} | {epe_topk_photo - oracle_epe:>+10.4f}")
    print(f"{'Top-{0} by pert×depth + photo per-pixel'.format(K):<45} | {epe_topk_pert:>8.4f} | {epe_topk_pert - oracle_epe:>+10.4f}")
    print(f"{'Top-{0} by oracle + photo per-pixel'.format(K):<45} | {epe_topk_oracle:>8.4f} | {epe_topk_oracle - oracle_epe:>+10.4f}")
    print("=" * 75)
    
    # Relative improvements
    print("\n📈 Relative to best single config:")
    improvement_all = (best_single_epe - epe_all_photo) / best_single_epe * 100
    improvement_photo = (best_single_epe - epe_topk_photo) / best_single_epe * 100
    improvement_pert = (best_single_epe - epe_topk_pert) / best_single_epe * 100
    improvement_oracle_topk = (best_single_epe - epe_topk_oracle) / best_single_epe * 100
    improvement_oracle = (best_single_epe - oracle_epe) / best_single_epe * 100
    
    print(f"   All + photo:        {improvement_all:+.1f}%")
    print(f"   Top-K photo + photo: {improvement_photo:+.1f}%")
    print(f"   Top-K pert + photo:  {improvement_pert:+.1f}%")
    print(f"   Top-K oracle + photo: {improvement_oracle_topk:+.1f}%")
    print(f"   Full oracle:         {improvement_oracle:+.1f}%")


if __name__ == "__main__":
    main()
