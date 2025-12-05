# File: scripts/compare_selection_methods.py
"""
Compare single config and ensemble selection methods.

Single config selection (pick ONE config by image-mean):
  - perturbation × depth
  - photo_log_raw
  - photo_gray_raw
  - photo_rgb (Euclidean)

Ensemble selection (per-pixel among ALL configs):
  - photo_log_raw
  - photo_gray_raw
  - photo_rgb (Euclidean)

Usage:
    python scripts/compare_selection_methods.py data/.../results_full.pkl
"""

import numpy as np
import pickle
import sys
import re
from pathlib import Path
import cv2


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


def compute_flow_smoothness(u, v, valid_mask):
    """
    Compute flow smoothness (Jacobian Frobenius norm).
    
    smoothness = mean((∂u/∂x)² + (∂u/∂y)² + (∂v/∂x)² + (∂v/∂y)²)
    
    Lower = smoother, more coherent flow = better
    Higher = turbulent, scattered flow = worse
    """
    # Compute gradients using Sobel (more robust than simple diff)
    dudx = cv2.Sobel(u.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3) / 8.0
    dudy = cv2.Sobel(u.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3) / 8.0
    dvdx = cv2.Sobel(v.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3) / 8.0
    dvdy = cv2.Sobel(v.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3) / 8.0
    
    # Jacobian Frobenius norm squared
    jacobian_sq = dudx**2 + dudy**2 + dvdx**2 + dvdy**2
    
    # Mean over valid pixels
    return np.nanmean(jacobian_sq[valid_mask])


def compute_epe_stats(epe_array, valid_mask):
    """
    Compute EPE statistics: mean, std, p50, p90, p95, p99.
    
    Returns dict with all stats.
    """
    valid_epe = epe_array[valid_mask]
    valid_epe = valid_epe[~np.isnan(valid_epe)]
    
    return {
        'mean': np.mean(valid_epe),
        'std': np.std(valid_epe),
        'p50': np.percentile(valid_epe, 50),
        'p90': np.percentile(valid_epe, 90),
        'p95': np.percentile(valid_epe, 95),
        'p99': np.percentile(valid_epe, 99),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_selection_methods.py <results_full.pkl>")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    
    print(f"📂 Loading {results_path}")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    n_configs = len(results)
    print(f"   {n_configs} configurations")
    
    # Load ground truth
    u_truth, v_truth, valid_mask = load_ground_truth(results_path)
    H, W = u_truth.shape
    print(f"   Shape: {H}×{W}, valid: {valid_mask.sum()}")
    
    # Build stacks
    print("\n📊 Building data stacks...")
    
    u_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    v_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    epe_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    # Stability metrics
    pert_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    traction_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    consistency_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    # Photometric metrics
    photo_log_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    photo_gray_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    photo_rgb_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    config_data = []
    
    for i, r in enumerate(results):
        config_name = r['metadata'].get('config_name', '')
        match = re.search(r'win(\d+)', config_name)
        winsize = int(match.group(1)) if match else 15
        depth = winsize / 2 + 1.0
        
        u = r['flows']['u_AB']
        v = r['flows']['v_AB']
        
        u_stack[i] = u
        v_stack[i] = v
        epe_stack[i] = np.sqrt((u - u_truth)**2 + (v - v_truth)**2)
        
        # Stability metrics (raw)
        pert_raw = r['metrics']['perturbation_raw_A']
        traction_raw = r['metrics']['traction_raw_A']
        consistency_raw = r['metrics']['consistency_raw_A']
        
        pert_stack[i] = pert_raw
        traction_stack[i] = traction_raw
        consistency_stack[i] = consistency_raw
        
        # Photometric metrics
        photo_log_stack[i] = r['metrics']['photo_log_raw_A']
        photo_gray_stack[i] = r['metrics']['photo_gray_raw_A']
        
        # RGB Euclidean
        photo_r = r['metrics']['photo_r_raw_A']
        photo_g = r['metrics']['photo_g_raw_A']
        photo_b = r['metrics']['photo_b_raw_A']
        photo_rgb_stack[i] = np.sqrt(photo_r**2 + photo_g**2 + photo_b**2)
        
        config_data.append({
            'idx': i,
            'name': config_name,
            'winsize': winsize,
            'depth': depth,
            'mean_epe': np.nanmean(epe_stack[i][valid_mask]),
            # Stability metrics × depth (for single config selection)
            'mean_pert_depth': np.nanmean(pert_raw[valid_mask]) * depth,
            'mean_traction_depth': np.nanmean(traction_raw[valid_mask]) * depth,
            'mean_consistency_depth': np.nanmean(consistency_raw[valid_mask]) * depth,
            # Photometric metrics (no depth scaling)
            'mean_photo_log': np.nanmean(photo_log_stack[i][valid_mask]),
            'mean_photo_gray': np.nanmean(photo_gray_stack[i][valid_mask]),
            'mean_photo_rgb': np.nanmean(photo_rgb_stack[i][valid_mask]),
        })
    
    # ========================================================================
    # Baselines
    # ========================================================================
    print("\n📊 Computing baselines...")
    
    # Oracle (per-pixel best)
    oracle_epe = np.nanmean(np.min(epe_stack, axis=0)[valid_mask])
    
    # Best/worst single config by EPE
    epes = [c['mean_epe'] for c in config_data]
    best_single_idx = np.argmin(epes)
    worst_single_idx = np.argmax(epes)
    best_single_epe = epes[best_single_idx]
    worst_single_epe = epes[worst_single_idx]
    
    # ========================================================================
    # Single config selection
    # ========================================================================
    print("\n📊 Computing single config selections...")
    
    # Stability metrics × depth
    selected_pert = min(config_data, key=lambda x: x['mean_pert_depth'])
    single_pert_epe = selected_pert['mean_epe']
    
    selected_traction = min(config_data, key=lambda x: x['mean_traction_depth'])
    single_traction_epe = selected_traction['mean_epe']
    
    selected_consistency = min(config_data, key=lambda x: x['mean_consistency_depth'])
    single_consistency_epe = selected_consistency['mean_epe']
    
    # Photometric metrics (no depth)
    selected_log = min(config_data, key=lambda x: x['mean_photo_log'])
    single_log_epe = selected_log['mean_epe']
    
    selected_gray = min(config_data, key=lambda x: x['mean_photo_gray'])
    single_gray_epe = selected_gray['mean_epe']
    
    selected_rgb = min(config_data, key=lambda x: x['mean_photo_rgb'])
    single_rgb_epe = selected_rgb['mean_epe']
    
    # ========================================================================
    # Ensemble selection (per-pixel)
    # ========================================================================
    print("\n📊 Computing ensemble selections...")
    
    def compute_ensemble(metric_stack):
        """Per-pixel selection by metric, return selected flows and EPE array."""
        best_idx = np.argmin(metric_stack, axis=0)
        y_idx, x_idx = np.mgrid[0:H, 0:W]
        u_selected = u_stack[best_idx, y_idx, x_idx]
        v_selected = v_stack[best_idx, y_idx, x_idx]
        epe = np.sqrt((u_selected - u_truth)**2 + (v_selected - v_truth)**2)
        return u_selected, v_selected, epe
    
    # Stability metrics (per-pixel)
    u_ens_pert, v_ens_pert, epe_ens_pert = compute_ensemble(pert_stack)
    u_ens_traction, v_ens_traction, epe_ens_traction = compute_ensemble(traction_stack)
    u_ens_consistency, v_ens_consistency, epe_ens_consistency = compute_ensemble(consistency_stack)
    
    # Photometric metrics (per-pixel)
    u_ens_log, v_ens_log, epe_ens_log = compute_ensemble(photo_log_stack)
    u_ens_gray, v_ens_gray, epe_ens_gray = compute_ensemble(photo_gray_stack)
    u_ens_rgb, v_ens_rgb, epe_ens_rgb = compute_ensemble(photo_rgb_stack)
    
    # Oracle ensemble (per-pixel best by EPE)
    oracle_idx = np.argmin(epe_stack, axis=0)
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    u_oracle = u_stack[oracle_idx, y_idx, x_idx]
    v_oracle = v_stack[oracle_idx, y_idx, x_idx]
    epe_oracle = np.min(epe_stack, axis=0)
    
    # ========================================================================
    # Compute EPE statistics for all methods
    # ========================================================================
    print("\n📊 Computing EPE statistics...")
    
    # Create zero EPE array for ground truth
    epe_gt = np.zeros((H, W), dtype=np.float32)
    
    # Collect all results: (name, epe_array, u, v)
    results_data = []
    
    # Baselines
    results_data.append(("Ground truth", epe_gt, u_truth, v_truth, None))
    results_data.append(("Oracle (per-pixel best)", epe_oracle, u_oracle, v_oracle, None))
    results_data.append(("Best single (by EPE)", epe_stack[best_single_idx], 
                         u_stack[best_single_idx], v_stack[best_single_idx], config_data[best_single_idx]['name']))
    results_data.append(("Worst single (by EPE)", epe_stack[worst_single_idx],
                         u_stack[worst_single_idx], v_stack[worst_single_idx], config_data[worst_single_idx]['name']))
    
    # Single config selections (stability) - include config info
    results_data.append(("S: pert×depth", epe_stack[selected_pert['idx']],
                         u_stack[selected_pert['idx']], v_stack[selected_pert['idx']], selected_pert['name']))
    results_data.append(("S: traction×depth", epe_stack[selected_traction['idx']],
                         u_stack[selected_traction['idx']], v_stack[selected_traction['idx']], selected_traction['name']))
    results_data.append(("S: consist×depth", epe_stack[selected_consistency['idx']],
                         u_stack[selected_consistency['idx']], v_stack[selected_consistency['idx']], selected_consistency['name']))
    
    # Single config selections (photometric) - include config info
    results_data.append(("S: photo_log", epe_stack[selected_log['idx']],
                         u_stack[selected_log['idx']], v_stack[selected_log['idx']], selected_log['name']))
    results_data.append(("S: photo_gray", epe_stack[selected_gray['idx']],
                         u_stack[selected_gray['idx']], v_stack[selected_gray['idx']], selected_gray['name']))
    results_data.append(("S: photo_rgb", epe_stack[selected_rgb['idx']],
                         u_stack[selected_rgb['idx']], v_stack[selected_rgb['idx']], selected_rgb['name']))
    
    # Ensemble selections (stability)
    results_data.append(("E: perturbation", epe_ens_pert, u_ens_pert, v_ens_pert, None))
    results_data.append(("E: traction", epe_ens_traction, u_ens_traction, v_ens_traction, None))
    results_data.append(("E: consistency", epe_ens_consistency, u_ens_consistency, v_ens_consistency, None))
    
    # Ensemble selections (photometric)
    results_data.append(("E: photo_log", epe_ens_log, u_ens_log, v_ens_log, None))
    results_data.append(("E: photo_gray", epe_ens_gray, u_ens_gray, v_ens_gray, None))
    results_data.append(("E: photo_rgb", epe_ens_rgb, u_ens_rgb, v_ens_rgb, None))
    
    # Compute stats for all
    all_stats = []
    for name, epe_arr, u, v, config_name in results_data:
        stats = compute_epe_stats(epe_arr, valid_mask)
        stats['smooth'] = compute_flow_smoothness(u, v, valid_mask)
        stats['name'] = name
        stats['config'] = config_name
        all_stats.append(stats)
    
    # ========================================================================
    # Print results table
    # ========================================================================
    print("\n" + "=" * 120)
    print("SELECTION METHOD COMPARISON")
    print("=" * 120)
    print(f"{'Method':<26} | {'mean':>7} | {'std':>7} | {'p50':>7} | {'p90':>7} | {'p95':>7} | {'p99':>7} | {'Smooth':>7} | Config")
    print("-" * 120)
    
    def print_row(stats):
        config_str = stats['config'] if stats['config'] else ""
        print(f"{stats['name']:<26} | {stats['mean']:>7.4f} | {stats['std']:>7.4f} | "
              f"{stats['p50']:>7.4f} | {stats['p90']:>7.4f} | {stats['p95']:>7.4f} | "
              f"{stats['p99']:>7.4f} | {stats['smooth']:>7.4f} | {config_str}")
    
    # Print baselines
    for s in all_stats[:4]:
        print_row(s)
    
    print("-" * 120)
    print("SINGLE CONFIG (S:)")
    for s in all_stats[4:10]:
        print_row(s)
    
    print("-" * 120)
    print("ENSEMBLE (E:)")
    for s in all_stats[10:]:
        print_row(s)
    
    print("=" * 120)
    
    # Summary
    oracle_stats = all_stats[1]
    best_single_stats = all_stats[2]
    best_ensemble_mean = min(s['mean'] for s in all_stats[10:])
    oracle_headroom = best_single_stats['mean'] - oracle_stats['mean']
    ensemble_captured = best_single_stats['mean'] - best_ensemble_mean
    
    print("\n📈 Summary:")
    print(f"   Oracle headroom: {oracle_headroom:.4f}")
    print(f"   Best ensemble captures: {ensemble_captured:.4f} ({ensemble_captured/oracle_headroom*100:.1f}% of headroom)")
    
    print("\n📊 Smoothness analysis (lower = smoother):")
    print(f"   Ground truth:        {all_stats[0]['smooth']:.4f}")
    print(f"   Best single config:  {best_single_stats['smooth']:.4f} (ratio to GT: {best_single_stats['smooth']/all_stats[0]['smooth']:.2f}×)")
    best_ens_log_stats = all_stats[13]  # E: photo_log
    print(f"   Ensemble (log):      {best_ens_log_stats['smooth']:.4f} (ratio to GT: {best_ens_log_stats['smooth']/all_stats[0]['smooth']:.2f}×)")
    print(f"   Oracle:              {oracle_stats['smooth']:.4f} (ratio to GT: {oracle_stats['smooth']/all_stats[0]['smooth']:.2f}×)")


if __name__ == "__main__":
    main()
