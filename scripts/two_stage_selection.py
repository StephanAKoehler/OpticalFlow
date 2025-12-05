#!/usr/bin/env python3
# File: scripts/two_stage_selection.py
"""
Two-stage config selection:
  Stage 1: Self-supervised metric → top 25% configs
  Stage 2: Flow quality metric → pick best among those

Reports EPE rank of final selection.

Usage:
    python scripts/two_stage_selection.py data/
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np


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
    
    valid_mask = (
        ~np.isnan(u_truth) & ~np.isnan(v_truth) &
        (np.abs(u_truth) < 1e8) & (np.abs(v_truth) < 1e8)
    )
    
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


def parse_config_name(config_name: str) -> dict:
    """
    Parse config name like 'farneback_w21_p5_i3_s0.5_e0.6_fl1_fs1'
    or 'dis_fs0_vd3_gd16_gs2_gi20' into dict.
    """
    parts = config_name.split('_')
    algorithm = parts[0]
    config = {'algorithm': algorithm}
    
    if algorithm == 'farneback':
        for part in parts[1:]:
            if part.startswith('w'):
                config['winsize'] = int(part[1:])
            elif part.startswith('p'):
                config['pyr_scale'] = int(part[1:]) / 10.0
            elif part.startswith('i'):
                config['iterations'] = int(part[1:])
            elif part.startswith('s'):
                config['poly_sigma'] = float(part[1:])
            elif part.startswith('e'):
                config['poly_n_encoded'] = float(part[1:])
            elif part.startswith('fl'):
                config['flags'] = int(part[2:])
            elif part.startswith('fs'):
                config['fast_pyramids'] = int(part[2:])
    elif algorithm == 'dis':
        for part in parts[1:]:
            if part.startswith('fs'):
                config['finest_scale'] = int(part[2:])
            elif part.startswith('vd'):
                config['variational_refinement_delta'] = int(part[2:])
            elif part.startswith('gd'):
                config['gradient_descent_iterations'] = int(part[2:])
            elif part.startswith('gs'):
                config['patch_stride'] = int(part[2:])
            elif part.startswith('gi'):
                config['patch_size'] = int(part[2:])
    
    return config


def get_pollution_depth(config_name: str) -> float:
    """
    Get empirically measured pollution depth for a config.
    
    Falls back to winsize-based estimate if src_contamination not available.
    """
    try:
        from src_contamination import get_margin
        config = parse_config_name(config_name)
        return float(get_margin(config, magnitude=1.0))
    except ImportError:
        # Fallback: estimate from winsize
        config = parse_config_name(config_name)
        if config.get('algorithm') == 'farneback':
            winsize = config.get('winsize', 15)
            # Empirical: depth ≈ 0.6 * winsize for Farneback
            return 0.6 * winsize
        elif config.get('algorithm') == 'dis':
            # DIS: approximately constant ~6px for finest_scale=0
            return 6.0
        else:
            return 10.0  # Conservative default


def compute_flow_quality(u: np.ndarray, v: np.ndarray, valid_mask: np.ndarray) -> dict:
    """
    Compute flow quality metrics.
    
    Returns dict with:
        - smoothness: mean gradient magnitude squared
        - divergence: mean |div|
        - curl: mean |curl|
        - folding: fraction of pixels with det(J) < 0
    """
    # Gradients
    du_dx = np.gradient(u, axis=1)
    du_dy = np.gradient(u, axis=0)
    dv_dx = np.gradient(v, axis=1)
    dv_dy = np.gradient(v, axis=0)
    
    # Smoothness: sum of squared gradients
    smoothness = du_dx**2 + du_dy**2 + dv_dx**2 + dv_dy**2
    
    # Divergence
    div = du_dx + dv_dy
    
    # Curl
    curl = dv_dx - du_dy
    
    # Determinant of Jacobian (for folding detection)
    # J = I + [du/dx, du/dy; dv/dx, dv/dy]
    # det(J) = (1 + du/dx)(1 + dv/dy) - du/dy * dv/dx
    det_J = (1 + du_dx) * (1 + dv_dy) - du_dy * dv_dx
    
    return {
        'smoothness': float(smoothness[valid_mask].mean()),
        'divergence': float(np.abs(div[valid_mask]).mean()),
        'curl': float(np.abs(curl[valid_mask]).mean()),
        'folding': float((det_J[valid_mask] < 0).mean()),
    }


def analyze_single_sequence(results_path: Path) -> dict:
    """
    Analyze one sequence with two-stage selection.
    
    Now includes asymmetry and kinematics metrics if available.
    Also computes combined rank score.
    """
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    n_configs = len(results)
    top_k = max(1, n_configs // 4)  # Top 25%
    
    # Load ground truth
    u_truth, v_truth, valid_mask = load_ground_truth(results_path)
    if u_truth is None:
        return None
    
    # Self-supervised metrics for stage 1
    ss_metrics = [
        'photo_log_raw_A',
        'perturbation_raw_A', 
        'traction_raw_A',
        'consistency_raw_A',
    ]
    
    # Check if new metrics are available
    has_asymmetry = 'relative_asym_A' in results[0].get('metrics', {})
    has_kinematics = 'divergence_A' in results[0].get('metrics', {})
    
    if has_asymmetry:
        ss_metrics.append('relative_asym_A')
    
    # Flow quality metrics for stage 2
    fq_metrics = ['smoothness', 'divergence', 'curl', 'folding']
    
    # Add pre-computed kinematics if available
    if has_kinematics:
        fq_metrics.extend(['kin_divergence', 'kin_curl', 'kin_folding', 'kin_shear'])
    
    # Collect per-config data
    config_data = []
    
    # Metrics that need pollution depth normalization
    ss_metrics_to_normalize = {'perturbation_raw_A', 'traction_raw_A', 'consistency_raw_A'}
    
    for i, r in enumerate(results):
        config_name = r['metadata'].get('config_name', f'config_{i}')
        depth = get_pollution_depth(config_name)
        
        u = r['flows']['u_AB']
        v = r['flows']['v_AB']
        epe = np.sqrt((u - u_truth)**2 + (v - v_truth)**2)
        mean_epe = epe[valid_mask].mean()
        
        row = {
            'config': config_name,
            'idx': i,
            'mean_epe': mean_epe,
            'pollution_depth': depth,
        }
        
        # Self-supervised metrics (normalize spatial metrics by pollution depth)
        for key in ss_metrics:
            if key in r['metrics']:
                metric = r['metrics'][key]
                if key in ss_metrics_to_normalize:
                    # Normalize: multiply by depth to compensate for window averaging
                    row[f'ss_{key}'] = (metric * depth)[valid_mask].mean()
                else:
                    # photo_log, relative_asym: no normalization
                    row[f'ss_{key}'] = metric[valid_mask].mean()
        
        # Compute flow quality metrics (on-the-fly) - normalize by depth
        fq = compute_flow_quality(u, v, valid_mask)
        row['fq_smoothness'] = fq['smoothness']  # Keep raw for now (depends on units)
        row['fq_divergence'] = fq['divergence'] * depth
        row['fq_curl'] = fq['curl'] * depth
        row['fq_folding'] = fq['folding'] * depth
        
        # Add pre-computed kinematics if available - normalize by depth
        if has_kinematics:
            row['fq_kin_divergence'] = np.abs(r['metrics']['divergence_A'][valid_mask]).mean() * depth
            row['fq_kin_curl'] = np.abs(r['metrics']['curl_A'][valid_mask]).mean() * depth
            row['fq_kin_folding'] = r['metrics']['folding_A'][valid_mask].mean() * depth
            row['fq_kin_shear'] = r['metrics']['shear_magnitude_A'][valid_mask].mean() * depth
        
        config_data.append(row)
    
    # Build EPE rank lookup
    sorted_by_epe = sorted(config_data, key=lambda x: x['mean_epe'])
    epe_rank = {cfg['idx']: rank for rank, cfg in enumerate(sorted_by_epe, start=1)}
    
    result = {
        'n_configs': n_configs,
        'top_k': top_k,
        'best_config': sorted_by_epe[0]['config'],
        'best_epe': sorted_by_epe[0]['mean_epe'],
        'two_stage': {},  # [ss_metric][fq_metric] -> EPE rank
        'stage1_only': {},  # ss_metric -> EPE rank of its top pick
        'has_asymmetry': has_asymmetry,
        'has_kinematics': has_kinematics,
        'combined_rank': {},  # Combined rank results
    }
    
    # For each SS metric
    for ss_key in ss_metrics:
        ss_col = f'ss_{ss_key}'
        if ss_col not in config_data[0]:
            continue
        
        # Stage 1: Get top 25% by this SS metric
        sorted_by_ss = sorted(config_data, key=lambda x: x[ss_col])
        top_quartile = sorted_by_ss[:top_k]
        
        # Stage 1 only: best by SS metric alone
        ss_best = sorted_by_ss[0]
        result['stage1_only'][ss_key] = epe_rank[ss_best['idx']]
        
        result['two_stage'][ss_key] = {}
        
        # Stage 2: Among top quartile, pick best by each FQ metric
        for fq_key in fq_metrics:
            fq_col = f'fq_{fq_key}'
            
            if fq_col not in config_data[0]:
                continue
            
            # Pick config with lowest FQ value among top quartile
            fq_best = min(top_quartile, key=lambda x: x[fq_col])
            
            result['two_stage'][ss_key][fq_key] = epe_rank[fq_best['idx']]
    
    # ==========================================================================
    # Combined rank analysis
    # ==========================================================================
    
    # Compute ranks for each metric
    def compute_metric_ranks(config_data, metric_col):
        sorted_by_metric = sorted(enumerate(config_data), key=lambda x: x[1][metric_col])
        ranks = {}
        for rank, (idx, cfg) in enumerate(sorted_by_metric, start=1):
            ranks[cfg['idx']] = rank
        return ranks
    
    # photo_log ranks
    if 'ss_photo_log_raw_A' in config_data[0]:
        photo_ranks = compute_metric_ranks(config_data, 'ss_photo_log_raw_A')
    else:
        photo_ranks = None
    
    # perturbation ranks
    if 'ss_perturbation_raw_A' in config_data[0]:
        pert_ranks = compute_metric_ranks(config_data, 'ss_perturbation_raw_A')
    else:
        pert_ranks = None
    
    # kin_folding ranks (if available)
    if 'fq_kin_folding' in config_data[0]:
        folding_ranks = compute_metric_ranks(config_data, 'fq_kin_folding')
    else:
        folding_ranks = None
    
    # Combined scores
    if photo_ranks and pert_ranks:
        # Photo + Pert (2-way)
        combined_2way = {}
        for cfg in config_data:
            idx = cfg['idx']
            combined_2way[idx] = photo_ranks[idx] + pert_ranks[idx]
        
        best_2way_idx = min(combined_2way, key=combined_2way.get)
        result['combined_rank']['photo+pert'] = {
            'winner_idx': best_2way_idx,
            'winner_epe_rank': epe_rank[best_2way_idx],
            'winner_score': combined_2way[best_2way_idx],
        }
        
        # Photo + Pert + Folding (3-way)
        if folding_ranks:
            combined_3way = {}
            for cfg in config_data:
                idx = cfg['idx']
                combined_3way[idx] = photo_ranks[idx] + pert_ranks[idx] + folding_ranks[idx]
            
            best_3way_idx = min(combined_3way, key=combined_3way.get)
            result['combined_rank']['photo+pert+folding'] = {
                'winner_idx': best_3way_idx,
                'winner_epe_rank': epe_rank[best_3way_idx],
                'winner_score': combined_3way[best_3way_idx],
            }
            
            # Also try: Photo + Folding (2-way without pert)
            combined_pf = {}
            for cfg in config_data:
                idx = cfg['idx']
                combined_pf[idx] = photo_ranks[idx] + folding_ranks[idx]
            
            best_pf_idx = min(combined_pf, key=combined_pf.get)
            result['combined_rank']['photo+folding'] = {
                'winner_idx': best_pf_idx,
                'winner_epe_rank': epe_rank[best_pf_idx],
                'winner_score': combined_pf[best_pf_idx],
            }
            
            # Pert + Folding
            combined_pertf = {}
            for cfg in config_data:
                idx = cfg['idx']
                combined_pertf[idx] = pert_ranks[idx] + folding_ranks[idx]
            
            best_pertf_idx = min(combined_pertf, key=combined_pertf.get)
            result['combined_rank']['pert+folding'] = {
                'winner_idx': best_pertf_idx,
                'winner_epe_rank': epe_rank[best_pertf_idx],
                'winner_score': combined_pertf[best_pertf_idx],
            }
            
            # =================================================================
            # Intersection filter approach: 
            # top 25% pert AND top 25% folding → pick best photo
            # =================================================================
            top_k = max(1, n_configs // 4)  # 25%
            
            # Get top 25% by perturbation
            sorted_by_pert = sorted(config_data, key=lambda x: x['ss_perturbation_raw_A'])
            top_pert_idxs = {cfg['idx'] for cfg in sorted_by_pert[:top_k]}
            
            # Get top 25% by folding
            sorted_by_folding = sorted(config_data, key=lambda x: x['fq_kin_folding'])
            top_folding_idxs = {cfg['idx'] for cfg in sorted_by_folding[:top_k]}
            
            # Intersection
            survivors = top_pert_idxs & top_folding_idxs
            
            if survivors:
                # Among survivors, pick best photo_log
                survivor_cfgs = [cfg for cfg in config_data if cfg['idx'] in survivors]
                best_survivor = min(survivor_cfgs, key=lambda x: x['ss_photo_log_raw_A'])
                
                result['combined_rank']['pert∩fold→photo'] = {
                    'winner_idx': best_survivor['idx'],
                    'winner_epe_rank': epe_rank[best_survivor['idx']],
                    'n_survivors': len(survivors),
                }
            else:
                # No intersection - fall back to union or just photo
                result['combined_rank']['pert∩fold→photo'] = {
                    'winner_idx': -1,
                    'winner_epe_rank': 999,
                    'n_survivors': 0,
                }
            
            # Also try: top 50% pert AND top 50% folding → pick best photo
            top_k_50 = max(1, n_configs // 2)  # 50%
            
            top_pert_50 = {cfg['idx'] for cfg in sorted_by_pert[:top_k_50]}
            top_folding_50 = {cfg['idx'] for cfg in sorted_by_folding[:top_k_50]}
            survivors_50 = top_pert_50 & top_folding_50
            
            if survivors_50:
                survivor_cfgs_50 = [cfg for cfg in config_data if cfg['idx'] in survivors_50]
                best_survivor_50 = min(survivor_cfgs_50, key=lambda x: x['ss_photo_log_raw_A'])
                
                result['combined_rank']['pert∩fold(50%)→photo'] = {
                    'winner_idx': best_survivor_50['idx'],
                    'winner_epe_rank': epe_rank[best_survivor_50['idx']],
                    'n_survivors': len(survivors_50),
                }
    
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python two_stage_selection.py <data_dir>")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    
    if not data_dir.is_dir():
        print(f"❌ Not a directory: {data_dir}")
        sys.exit(1)
    
    # Find all results files
    results_files = list(data_dir.glob('*/analysis/*/sweep/pair_*/results_full.pkl'))
    
    if not results_files:
        print(f"❌ No results found in {data_dir}")
        sys.exit(1)
    
    print(f"📂 Found {len(results_files)} result files")
    print()
    
    # Analyze each sequence (deduplicate by name)
    all_results = {}
    
    for results_path in sorted(results_files):
        seq_name = get_sequence_name(results_path)
        if seq_name in all_results:
            continue
        
        result = analyze_single_sequence(results_path)
        if result:
            all_results[seq_name] = result
            print(f"   ✓ {seq_name}: n={result['n_configs']}, top25%={result['top_k']}")
    
    if not all_results:
        print("❌ No valid results")
        sys.exit(1)
    
    n_sequences = len(all_results)
    first_result = next(iter(all_results.values()))
    ss_metrics = list(first_result['two_stage'].keys())
    fq_metrics = list(first_result['two_stage'][ss_metrics[0]].keys())
    n_configs = first_result['n_configs']
    
    # Report which extra metrics are available
    has_asym = first_result.get('has_asymmetry', False)
    has_kin = first_result.get('has_kinematics', False)
    if has_asym or has_kin:
        print(f"📊 Extra metrics available: asymmetry={has_asym}, kinematics={has_kin}")
        print()
    
    # Calculate column width based on number of metrics
    col_width = 12
    table_width = 20 + (len(fq_metrics) + 1) * (col_width + 3) + 5
    
    # ==========================================================================
    # Per-sequence results
    # ==========================================================================
    
    for seq_name in sorted(all_results.keys()):
        result = all_results[seq_name]
        
        print()
        print("=" * table_width)
        print(f"{seq_name} (best possible: rank 1, EPE={result['best_epe']:.4f})")
        print("=" * table_width)
        
        # Header
        header = f"{'SS \\ FQ':<20}"
        for fq in fq_metrics:
            fq_short = fq[:col_width]
            header += f" | {fq_short:>{col_width}}"
        header += f" | {'SS only':>{col_width}}"
        print(header)
        print("-" * table_width)
        
        for ss in ss_metrics:
            ss_short = ss.replace('_raw_A', '')[:18]
            row = f"{ss_short:<20}"
            for fq in fq_metrics:
                rank = result['two_stage'][ss].get(fq, -1)
                if rank == -1:
                    row += f" | {'N/A':>{col_width}}"
                else:
                    marker = "✓" if rank <= 5 else " "
                    row += f" | {rank:>{col_width-1}}{marker}"
            ss_only = result['stage1_only'][ss]
            marker = "✓" if ss_only <= 5 else " "
            row += f" | {ss_only:>{col_width-1}}{marker}"
            print(row)
    
    # ==========================================================================
    # Aggregate summary
    # ==========================================================================
    
    print()
    print("=" * table_width)
    print("AGGREGATE: MEDIAN EPE RANK ACROSS ALL SEQUENCES")
    print("=" * table_width)
    
    # Header
    header = f"{'SS \\ FQ':<20}"
    for fq in fq_metrics:
        fq_short = fq[:col_width]
        header += f" | {fq_short:>{col_width}}"
    header += f" | {'SS only':>{col_width}}"
    print(header)
    print("-" * table_width)
    
    for ss in ss_metrics:
        ss_short = ss.replace('_raw_A', '')[:18]
        row = f"{ss_short:<20}"
        for fq in fq_metrics:
            ranks = [all_results[s]['two_stage'][ss].get(fq, 999) for s in all_results]
            ranks = [r for r in ranks if r != 999]  # Filter missing
            if ranks:
                median = np.median(ranks)
                row += f" | {median:>{col_width}.0f}"
            else:
                row += f" | {'N/A':>{col_width}}"
        ss_only_ranks = [all_results[s]['stage1_only'][ss] for s in all_results]
        ss_only_median = np.median(ss_only_ranks)
        row += f" | {ss_only_median:>{col_width}.0f}"
        print(row)
    
    print("=" * table_width)
    
    # ==========================================================================
    # Best combinations
    # ==========================================================================
    
    print()
    print("=" * table_width)
    print(f"RANK <= 5 COUNT (out of {n_sequences} sequences)")
    print("=" * table_width)
    
    # Header
    header = f"{'SS \\ FQ':<20}"
    for fq in fq_metrics:
        fq_short = fq[:col_width]
        header += f" | {fq_short:>{col_width}}"
    header += f" | {'SS only':>{col_width}}"
    print(header)
    print("-" * table_width)
    
    best_combo = None
    best_count = -1
    
    for ss in ss_metrics:
        ss_short = ss.replace('_raw_A', '')[:18]
        row = f"{ss_short:<20}"
        for fq in fq_metrics:
            ranks = [all_results[s]['two_stage'][ss].get(fq, 999) for s in all_results]
            ranks = [r for r in ranks if r != 999]
            if ranks:
                count = sum(1 for r in ranks if r <= 5)
                row += f" | {count:>{col_width}}"
                if count > best_count:
                    best_count = count
                    best_combo = (ss, fq)
            else:
                row += f" | {'N/A':>{col_width}}"
        ss_only_ranks = [all_results[s]['stage1_only'][ss] for s in all_results]
        ss_only_count = sum(1 for r in ss_only_ranks if r <= 5)
        row += f" | {ss_only_count:>{col_width}}"
        if ss_only_count > best_count:
            best_count = ss_only_count
            best_combo = (ss, 'none')
        print(row)
    
    print("=" * table_width)
    
    # ==========================================================================
    # Worst case analysis
    # ==========================================================================
    
    print()
    print("=" * table_width)
    print("WORST RANK (lower is better)")
    print("=" * table_width)
    
    # Header
    header = f"{'SS \\ FQ':<20}"
    for fq in fq_metrics:
        fq_short = fq[:col_width]
        header += f" | {fq_short:>{col_width}}"
    header += f" | {'SS only':>{col_width}}"
    print(header)
    print("-" * table_width)
    
    for ss in ss_metrics:
        ss_short = ss.replace('_raw_A', '')[:18]
        row = f"{ss_short:<20}"
        for fq in fq_metrics:
            ranks = [all_results[s]['two_stage'][ss].get(fq, 999) for s in all_results]
            ranks = [r for r in ranks if r != 999]
            if ranks:
                worst = max(ranks)
                row += f" | {worst:>{col_width}}"
            else:
                row += f" | {'N/A':>{col_width}}"
        ss_only_ranks = [all_results[s]['stage1_only'][ss] for s in all_results]
        ss_only_worst = max(ss_only_ranks)
        row += f" | {ss_only_worst:>{col_width}}"
        print(row)
    
    print("=" * table_width)
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    
    print()
    print("=" * 80)
    print("BEST COMBINATION")
    print("=" * 80)
    
    if best_combo is None:
        print("   No valid combinations found")
    elif best_combo[1] == 'none':
        print(f"   {best_combo[0]} alone: {best_count}/{n_sequences} sequences with rank <= 5")
    else:
        print(f"   {best_combo[0]} → {best_combo[1]}: {best_count}/{n_sequences} sequences with rank <= 5")
    
    # Compare to single-stage
    print()
    print("Does two-stage help?")
    for ss in ss_metrics:
        ss_short = ss.replace('_raw_A', '')
        ss_only_ranks = [all_results[s]['stage1_only'][ss] for s in all_results]
        ss_only_median = np.median(ss_only_ranks)
        
        best_fq = None
        best_fq_median = 999
        for fq in fq_metrics:
            ranks = [all_results[s]['two_stage'][ss].get(fq, 999) for s in all_results]
            ranks = [r for r in ranks if r != 999]
            if ranks:
                median = np.median(ranks)
                if median < best_fq_median:
                    best_fq_median = median
                    best_fq = fq
        
        if best_fq is not None:
            improvement = ss_only_median - best_fq_median
            sign = "+" if improvement > 0 else ""
            print(f"   {ss_short}: SS-only median={ss_only_median:.0f}, best two-stage ({best_fq})={best_fq_median:.0f} ({sign}{improvement:.0f})")
        else:
            print(f"   {ss_short}: SS-only median={ss_only_median:.0f}, no two-stage data")

    # ==========================================================================
    # Combined Rank Analysis
    # ==========================================================================
    
    # Check if combined_rank data is available
    has_combined = any('combined_rank' in all_results[s] and all_results[s]['combined_rank'] 
                       for s in all_results)
    
    if has_combined:
        print()
        print("=" * 80)
        print("COMBINED RANK SELECTION (equal weights, no hyperparameters)")
        print("=" * 80)
        
        # Get all combination types
        combo_types = set()
        for s in all_results:
            if 'combined_rank' in all_results[s]:
                combo_types.update(all_results[s]['combined_rank'].keys())
        combo_types = sorted(combo_types)
        
        # Per-sequence results
        print()
        print(f"{'Sequence':<15}", end="")
        for combo in combo_types:
            print(f" | {combo:>20}", end="")
        print()
        print("-" * (15 + len(combo_types) * 24))
        
        for seq_name in sorted(all_results.keys()):
            result = all_results[seq_name]
            row = f"{seq_name:<15}"
            for combo in combo_types:
                if combo in result.get('combined_rank', {}):
                    rank = result['combined_rank'][combo]['winner_epe_rank']
                    marker = "✓" if rank <= 5 else " "
                    row += f" | {rank:>19}{marker}"
                else:
                    row += f" | {'N/A':>20}"
            print(row)
        
        # Aggregate stats
        print()
        print("-" * (15 + len(combo_types) * 24))
        
        # Median
        row_median = f"{'Median':<15}"
        for combo in combo_types:
            ranks = [all_results[s]['combined_rank'][combo]['winner_epe_rank'] 
                     for s in all_results 
                     if combo in all_results[s].get('combined_rank', {})]
            if ranks:
                row_median += f" | {np.median(ranks):>20.0f}"
            else:
                row_median += f" | {'N/A':>20}"
        print(row_median)
        
        # Rank <= 5 count
        row_top5 = f"{'Rank ≤ 5':<15}"
        for combo in combo_types:
            ranks = [all_results[s]['combined_rank'][combo]['winner_epe_rank'] 
                     for s in all_results 
                     if combo in all_results[s].get('combined_rank', {})]
            if ranks:
                count = sum(1 for r in ranks if r <= 5)
                row_top5 += f" | {count:>20}"
            else:
                row_top5 += f" | {'N/A':>20}"
        print(row_top5)
        
        # Worst rank
        row_worst = f"{'Worst':<15}"
        for combo in combo_types:
            ranks = [all_results[s]['combined_rank'][combo]['winner_epe_rank'] 
                     for s in all_results 
                     if combo in all_results[s].get('combined_rank', {})]
            if ranks:
                row_worst += f" | {max(ranks):>20}"
            else:
                row_worst += f" | {'N/A':>20}"
        print(row_worst)
        
        print("=" * 80)


if __name__ == "__main__":
    main()
