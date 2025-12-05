# File: scripts/traction_gated_test.py
"""
Test traction-gated ensemble selection across all sequences.

Usage:
    python scripts/traction_gated_test.py <data_dir> [fidelity_threshold]
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.self_supervised import (
    traction_gated_ensemble, 
    traction_gated_fallback_ensemble,
    traction_weighted_ensemble,
    smooth_traction_ensemble,
    get_pollution_depth
)


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
    movie_dir = results_path.parent.parent.parent.parent.parent
    source_info = movie_dir / 'source_info.json'
    if source_info.exists():
        with open(source_info) as f:
            info = json.load(f)
        return info.get('sequence', movie_dir.name)
    return movie_dir.name


def get_algorithm(results_path: Path) -> str:
    """Detect algorithm from results file."""
    import pickle
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    config_name = results[0]['metadata'].get('config_name', '')
    if config_name.startswith('FB_') or config_name.startswith('farneback'):
        return 'farneback'
    elif config_name.startswith('F_') or config_name.startswith('DIS'):
        return 'dis'
    return 'unknown'


def analyze_sequence(results_path: Path, fidelity_threshold: float = 0.5, 
                     traction_exponent: float = 2.0):
    """Analyze one sequence with all traction-based ensemble methods."""
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    u_truth, v_truth, valid_mask = load_ground_truth(results_path)
    if u_truth is None:
        return None
    
    n_configs = len(results)
    H, W = u_truth.shape
    
    # Handle shape mismatch
    sample_u = results[0]['flows']['u_AB']
    if sample_u.shape != u_truth.shape:
        from scipy.ndimage import zoom
        scale_y = sample_u.shape[0] / u_truth.shape[0]
        scale_x = sample_u.shape[1] / u_truth.shape[1]
        u_truth = zoom(u_truth, (scale_y, scale_x), order=1)
        v_truth = zoom(v_truth, (scale_y, scale_x), order=1)
        valid_mask = zoom(valid_mask.astype(float), (scale_y, scale_x), order=0) > 0.5
        H, W = u_truth.shape
    
    # Build stacks for comparison methods
    u_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    v_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    epe_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    photo_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    pert_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    for i, r in enumerate(results):
        u = r['flows']['u_AB']
        v = r['flows']['v_AB']
        u_stack[i] = u
        v_stack[i] = v
        epe_stack[i] = np.sqrt((u - u_truth)**2 + (v - v_truth)**2)
        photo_stack[i] = r['metrics']['photo_log_raw_A']
        
        # Normalize perturbation
        config_name = r['metadata'].get('config_name', f'config_{i}')
        depth = get_pollution_depth(config_name)
        pert_stack[i] = r['metrics']['perturbation_raw_A'] * depth
    
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    
    # Oracle (best possible)
    oracle_idx = np.argmin(epe_stack, axis=0)
    epe_oracle = epe_stack[oracle_idx, y_idx, x_idx]
    
    # Best single config
    mean_epes = [epe_stack[i][valid_mask].mean() for i in range(n_configs)]
    best_single_idx = int(np.argmin(mean_epes))
    best_single_rank = 1  # By definition
    epe_best_single = epe_stack[best_single_idx]
    
    # Photo-only ensemble
    photo_idx = np.argmin(photo_stack, axis=0)
    epe_photo = epe_stack[photo_idx, y_idx, x_idx]
    
    # Per-pixel perturbation selection (no photo, no traction gating)
    pert_idx = np.argmin(pert_stack, axis=0)
    u_pert_perpixel = u_stack[pert_idx, y_idx, x_idx]
    v_pert_perpixel = v_stack[pert_idx, y_idx, x_idx]
    epe_pert_perpixel = np.sqrt((u_pert_perpixel - u_truth)**2 + (v_pert_perpixel - v_truth)**2)
    
    # 1. Original: traction-gated with per-pixel pert fallback
    gated = traction_gated_ensemble(results, delta_mag=1.0, 
                                     fidelity_threshold=fidelity_threshold)
    epe_gated = np.sqrt((gated['u'] - u_truth)**2 + (gated['v'] - v_truth)**2)
    
    # 2a. Gated with global fallback - PHOTO metric
    fb_photo = traction_gated_fallback_ensemble(results, delta_mag=1.0,
                                                 fidelity_threshold=fidelity_threshold,
                                                 fallback_metric='photo',
                                                 traction_exponent=traction_exponent)
    epe_fb_photo = np.sqrt((fb_photo['u'] - u_truth)**2 + (fb_photo['v'] - v_truth)**2)
    
    # 2b. Gated with global fallback - PERT metric
    fb_pert = traction_gated_fallback_ensemble(results, delta_mag=1.0,
                                                fidelity_threshold=fidelity_threshold,
                                                fallback_metric='pert',
                                                traction_exponent=traction_exponent)
    epe_fb_pert = np.sqrt((fb_pert['u'] - u_truth)**2 + (fb_pert['v'] - v_truth)**2)
    
    # 3. Smooth ensemble (spatially-varying soft fallback)
    smooth = smooth_traction_ensemble(results, delta_mag=1.0,
                                       traction_exponent=traction_exponent)
    epe_smooth = np.sqrt((smooth['u'] - u_truth)**2 + (smooth['v'] - v_truth)**2)
    
    # Get fallback config info and ranks
    sorted_epes = sorted(enumerate(mean_epes), key=lambda x: x[1])
    
    fb_photo_config = results[fb_photo['fallback_idx']]['metadata'].get('config_name', '')
    fb_photo_rank = next(i+1 for i, (idx, _) in enumerate(sorted_epes) if idx == fb_photo['fallback_idx'])
    
    fb_pert_config = results[fb_pert['fallback_idx']]['metadata'].get('config_name', '')
    fb_pert_rank = next(i+1 for i, (idx, _) in enumerate(sorted_epes) if idx == fb_pert['fallback_idx'])
    
    return {
        'oracle': float(epe_oracle[valid_mask].mean()),
        'best_single': float(epe_best_single[valid_mask].mean()),
        'best_single_idx': best_single_idx,
        'photo': float(epe_photo[valid_mask].mean()),
        'pert_perpixel': float(epe_pert_perpixel[valid_mask].mean()),
        'gated_pert': float(epe_gated[valid_mask].mean()),
        'fb_photo': float(epe_fb_photo[valid_mask].mean()),
        'fb_pert': float(epe_fb_pert[valid_mask].mean()),
        'smooth': float(epe_smooth[valid_mask].mean()),
        'smooth_temp': smooth['temperature'],
        'fb_photo_config': fb_photo_config,
        'fb_pert_config': fb_pert_config,
        'fb_photo_idx': fb_photo['fallback_idx'],
        'fb_pert_idx': fb_pert['fallback_idx'],
        'fb_photo_rank': fb_photo_rank,
        'fb_pert_rank': fb_pert_rank,
        'n_configs': n_configs,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/traction_gated_test.py <data_dir> [fidelity_threshold] [traction_exponent]")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    fidelity_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    traction_exponent = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    
    # Find all results files
    results_files = list(data_dir.glob('*/analysis/*/sweep/pair_*/results_full.pkl'))
    
    if not results_files:
        print(f"❌ No results found in {data_dir}")
        sys.exit(1)
    
    print(f"📂 Found {len(results_files)} result files")
    print(f"📊 Traction fidelity threshold = {fidelity_threshold}")
    print(f"📊 Traction exponent = {traction_exponent}")
    print()
    
    # Deduplicate by sequence name AND algorithm
    seen = set()
    unique_files = []
    for f in sorted(results_files):
        seq_name = get_sequence_name(f)
        algorithm = get_algorithm(f)
        key = (seq_name, algorithm)
        if key not in seen:
            seen.add(key)
            unique_files.append((seq_name, algorithm, f))
    
    # Group by algorithm
    from collections import defaultdict
    by_algorithm = defaultdict(list)
    for seq_name, algorithm, f in unique_files:
        by_algorithm[algorithm].append((seq_name, f))
    
    for algorithm in sorted(by_algorithm.keys()):
        files = by_algorithm[algorithm]
        n_configs = None
        
        print()
        print(f"{'='*100}")
        print(f"ALGORITHM: {algorithm.upper()} ({len(files)} sequences)")
        print(f"{'='*100}")
        
        # Results table
        print()
        print("=" * 230)
        print(f"{'Sequence':<12} | {'BestSingle':>10} | {'Photo':>10} | {'PertPix':>10} | {'FB_pert':>10} | {'Smooth':>10} | {'Oracle':>10} | {'PertPix vsBest':>14} | {'FBpert vsBest':>13} | {'FBpert Rank':>11}")
        print("-" * 230)
        
        all_results = []
        
        for seq_name, results_path in sorted(files):
            try:
                r = analyze_sequence(results_path, fidelity_threshold, traction_exponent)
                
                if r is None:
                    print(f"{seq_name:<12} | {'No GT':>10}")
                    continue
                
                n_configs = r['n_configs']
                best = r['best_single']
                photo = r['photo']
                pert_perpixel = r['pert_perpixel']
                fb_pert = r['fb_pert']
                smooth = r['smooth']
                oracle = r['oracle']
                
                # Improvements (positive = better than best single)
                pertpix_vs_best = (best - pert_perpixel) / best * 100
                fbpert_vs_best = (best - fb_pert) / best * 100
                smooth_vs_best = (best - smooth) / best * 100
                
                pertpix_sign = '+' if pertpix_vs_best > 0 else ''
                fbpert_sign = '+' if fbpert_vs_best > 0 else ''
                smooth_sign = '+' if smooth_vs_best > 0 else ''
                
                print(f"{seq_name:<12} | {best:>10.4f} | {photo:>10.4f} | {pert_perpixel:>10.4f} | {fb_pert:>10.4f} | {smooth:>10.4f} | {oracle:>10.4f} | {pertpix_sign}{pertpix_vs_best:>13.1f}% | {fbpert_sign}{fbpert_vs_best:>12.1f}% | {r['fb_pert_rank']:>5}/{r['n_configs']:<5}")
                
                all_results.append({
                    'seq': seq_name,
                    **r,
                    'pertpix_vs_best': pertpix_vs_best,
                    'fbpert_vs_best': fbpert_vs_best,
                    'smooth_vs_best': smooth_vs_best,
                })
                
            except Exception as e:
                print(f"{seq_name:<12} | ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        print("-" * 230)
        
        # Averages
        if all_results:
            avg_pertpix_vs_best = np.mean([r['pertpix_vs_best'] for r in all_results])
            avg_fbpert_vs_best = np.mean([r['fbpert_vs_best'] for r in all_results])
            avg_smooth_vs_best = np.mean([r['smooth_vs_best'] for r in all_results])
            
            pertpix_sign = '+' if avg_pertpix_vs_best > 0 else ''
            fbpert_sign = '+' if avg_fbpert_vs_best > 0 else ''
            smooth_sign = '+' if avg_smooth_vs_best > 0 else ''
            
            print(f"{'AVERAGE':<12} | {'':<10} | {'':<10} | {'':<10} | {'':<10} | {'':<10} | {'':<10} | {pertpix_sign}{avg_pertpix_vs_best:>13.1f}% | {fbpert_sign}{avg_fbpert_vs_best:>12.1f}%")
        
        print("=" * 230)
        
        # Summary of fallback selection (for FB_pert method)
        print()
        print("FB_pert fallback configs selected:")
        for r in all_results:
            fb_short = r['fb_pert_config'].split('_', 1)[1] if '_' in r['fb_pert_config'] else r['fb_pert_config']
            fb_match = "✓" if r['fb_pert_idx'] == r['best_single_idx'] else ""
            print(f"  {r['seq']:<12}: rank {r['fb_pert_rank']:>2}/{r['n_configs']} {fb_short} {fb_match}")


if __name__ == "__main__":
    main()
