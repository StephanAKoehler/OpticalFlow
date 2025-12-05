#!/usr/bin/env python3
# File: scripts/predict_best_config.py
"""
Analyze whether self-supervised metrics can predict best config.

For each sequence:
- Find the best config by EPE (ground truth)
- Sort configs by each metric
- Report where the true best lands in each metric's ranking

Usage:
    python scripts/predict_best_config.py data/
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


def analyze_single_sequence(results_path: Path) -> dict:
    """
    Analyze one sequence.
    
    Returns dict with epe_ranks: for each metric, if we pick the config
    that metric says is best, what EPE rank does that config actually have?
    
    Also returns voting results for consensus approach.
    """
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    n_configs = len(results)
    
    # Load ground truth
    u_truth, v_truth, valid_mask = load_ground_truth(results_path)
    if u_truth is None:
        return None
    
    # Metrics to analyze
    metric_keys = [
        'photo_log_raw_A',
        'perturbation_raw_A', 
        'traction_raw_A',
        'consistency_raw_A',
        'photo_gray_raw_A',
        'photometric_A',
    ]
    
    # Collect per-config data
    config_data = []
    
    for i, r in enumerate(results):
        config_name = r['metadata'].get('config_name', f'config_{i}')
        
        u = r['flows']['u_AB']
        v = r['flows']['v_AB']
        epe = np.sqrt((u - u_truth)**2 + (v - v_truth)**2)
        mean_epe = epe[valid_mask].mean()
        
        row = {
            'config': config_name,
            'idx': i,
            'mean_epe': mean_epe,
        }
        
        for key in metric_keys:
            if key in r['metrics']:
                metric = r['metrics'][key]
                row[f'mean_{key}'] = metric[valid_mask].mean()
        
        config_data.append(row)
    
    # Sort by EPE to get ground truth ranking
    sorted_by_epe = sorted(config_data, key=lambda x: x['mean_epe'])
    
    # Build EPE rank lookup: config idx -> EPE rank
    epe_rank = {cfg['idx']: rank for rank, cfg in enumerate(sorted_by_epe, start=1)}
    
    epe_best = sorted_by_epe[0]
    
    result = {
        'n_configs': n_configs,
        'best_config': epe_best['config'],
        'best_epe': epe_best['mean_epe'],
        'epe_ranks': {},  # EPE rank of the config each metric picks as best
        'voting': {},     # Voting results for different thresholds
    }
    
    # Per-metric best picks
    for key in metric_keys:
        mean_key = f'mean_{key}'
        if mean_key not in config_data[0]:
            continue
        
        # Find config with lowest metric value (metric's pick for "best")
        metric_best = min(config_data, key=lambda x: x[mean_key])
        
        # What's that config's actual EPE rank?
        result['epe_ranks'][key] = epe_rank[metric_best['idx']]
    
    # Voting analysis for different top-K thresholds
    for top_pct in [10, 20, 30]:
        top_k = max(1, int(n_configs * top_pct / 100))
        
        # Count votes for each config
        votes = {cfg['idx']: 0 for cfg in config_data}
        
        for key in metric_keys:
            mean_key = f'mean_{key}'
            if mean_key not in config_data[0]:
                continue
            
            # Sort by this metric and get top-K
            sorted_by_metric = sorted(config_data, key=lambda x: x[mean_key])
            top_k_configs = sorted_by_metric[:top_k]
            
            for cfg in top_k_configs:
                votes[cfg['idx']] += 1
        
        # Find config with most votes
        max_votes = max(votes.values())
        winners = [idx for idx, v in votes.items() if v == max_votes]
        
        # If tie, pick the one with lowest average metric rank
        if len(winners) > 1:
            # Break tie by picking lowest mean EPE among tied (but we don't know EPE!)
            # Instead, break by lowest sum of metric values
            def avg_metric_value(idx):
                cfg = config_data[idx]
                vals = [cfg[f'mean_{k}'] for k in metric_keys if f'mean_{k}' in cfg]
                return sum(vals) / len(vals) if vals else float('inf')
            
            winner_idx = min(winners, key=avg_metric_value)
        else:
            winner_idx = winners[0]
        
        winner_epe_rank = epe_rank[winner_idx]
        winner_config = config_data[winner_idx]['config']
        
        result['voting'][top_pct] = {
            'top_k': top_k,
            'winner_idx': winner_idx,
            'winner_config': winner_config,
            'winner_votes': max_votes,
            'winner_epe_rank': winner_epe_rank,
        }
    
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_best_config.py <data_dir>")
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
    
    # Analyze each sequence (deduplicate by sequence name)
    all_results = {}
    
    for results_path in sorted(results_files):
        seq_name = get_sequence_name(results_path)
        
        # Skip if we already have this sequence (take first one)
        if seq_name in all_results:
            continue
            
        result = analyze_single_sequence(results_path)
        if result:
            all_results[seq_name] = result
            print(f"   ✓ {seq_name}: best={result['best_config'][:40]} (EPE={result['best_epe']:.4f})")
    
    if not all_results:
        print("❌ No valid results")
        sys.exit(1)
    
    n_sequences = len(all_results)
    
    # Get all metric names
    first_result = next(iter(all_results.values()))
    metric_keys = list(first_result['epe_ranks'].keys())
    n_configs = first_result['n_configs']
    
    # ==========================================================================
    # Main Table: EPE rank of each metric's pick
    # ==========================================================================
    print()
    print("=" * 120)
    print(f"EPE RANK OF METRIC'S BEST PICK (1 = perfect, {n_configs} = worst)")
    print("=" * 120)
    print()
    print("Question: If I pick the config with lowest metric value, what's its actual EPE rank?")
    print()
    
    # Header
    header = f"{'Sequence':<14} | {'n':>3}"
    for key in metric_keys:
        short_key = key.replace('_raw_A', '').replace('_A', '')[:10]
        header += f" | {short_key:>10}"
    print(header)
    print("-" * 120)
    
    # Data rows
    for seq_name in sorted(all_results.keys()):
        result = all_results[seq_name]
        row = f"{seq_name:<14} | {result['n_configs']:>3}"
        for key in metric_keys:
            rank = result['epe_ranks'].get(key, -1)
            marker = "✓" if rank == 1 else " "
            row += f" | {rank:>9}{marker}"
        print(row)
    
    print("-" * 120)
    
    # Summary stats
    print()
    
    # How often does each metric find the winner?
    correct_row = f"{'Rank = 1':<14} | {'':>3}"
    for key in metric_keys:
        ranks = [all_results[s]['epe_ranks'].get(key, 999) for s in all_results]
        n_correct = sum(1 for r in ranks if r == 1)
        correct_row += f" | {n_correct:>9}/{n_sequences}"
    print(correct_row)
    
    # Top-5 (is the best at least in top 5%?)
    top5_row = f"{'Rank <= 5':<14} | {'':>3}"
    for key in metric_keys:
        ranks = [all_results[s]['epe_ranks'].get(key, 999) for s in all_results]
        n_top5 = sum(1 for r in ranks if r <= 5)
        top5_row += f" | {n_top5:>9}/{n_sequences}"
    print(top5_row)
    
    # Top-10
    top10_row = f"{'Rank <= 10':<14} | {'':>3}"
    for key in metric_keys:
        ranks = [all_results[s]['epe_ranks'].get(key, 999) for s in all_results]
        n_top10 = sum(1 for r in ranks if r <= 10)
        top10_row += f" | {n_top10:>9}/{n_sequences}"
    print(top10_row)
    
    # Median rank
    median_row = f"{'Median rank':<14} | {'':>3}"
    for key in metric_keys:
        ranks = [all_results[s]['epe_ranks'].get(key, 999) for s in all_results]
        median_rank = np.median(ranks)
        median_row += f" | {median_rank:>10.0f}"
    print(median_row)
    
    # Worst case
    worst_row = f"{'Worst rank':<14} | {'':>3}"
    for key in metric_keys:
        ranks = [all_results[s]['epe_ranks'].get(key, 999) for s in all_results]
        worst_rank = max(ranks)
        worst_row += f" | {worst_rank:>10}"
    print(worst_row)
    
    print("=" * 120)
    
    # ==========================================================================
    # Voting Table
    # ==========================================================================
    print()
    print("=" * 120)
    print("VOTING APPROACH: Pick config that most metrics put in top K%")
    print("=" * 120)
    print()
    
    # Header
    header = f"{'Sequence':<14} | {'n':>3}"
    for top_pct in [10, 20, 30]:
        header += f" | {'Top ' + str(top_pct) + '%':>12}"
    header += f" | {'Best single':>12}"
    print(header)
    print("-" * 120)
    
    # Data rows
    for seq_name in sorted(all_results.keys()):
        result = all_results[seq_name]
        row = f"{seq_name:<14} | {result['n_configs']:>3}"
        for top_pct in [10, 20, 30]:
            v = result['voting'][top_pct]
            rank = v['winner_epe_rank']
            votes = v['winner_votes']
            marker = "✓" if rank == 1 else " "
            row += f" | {rank:>3} ({votes}v){marker:>3}"
        # Compare to photo_log (best single metric)
        photo_rank = result['epe_ranks'].get('photo_log_raw_A', 999)
        row += f" | {photo_rank:>12}"
        print(row)
    
    print("-" * 120)
    
    # Summary for voting
    for top_pct in [10, 20, 30]:
        ranks = [all_results[s]['voting'][top_pct]['winner_epe_rank'] for s in all_results]
        n_correct = sum(1 for r in ranks if r == 1)
        n_top5 = sum(1 for r in ranks if r <= 5)
        median = np.median(ranks)
        worst = max(ranks)
        print(f"Top {top_pct}% voting:  Rank=1: {n_correct}/8 | Rank<=5: {n_top5}/8 | Median: {median:.0f} | Worst: {worst}")
    
    # photo_log for comparison
    ranks = [all_results[s]['epe_ranks'].get('photo_log_raw_A', 999) for s in all_results]
    n_correct = sum(1 for r in ranks if r == 1)
    n_top5 = sum(1 for r in ranks if r <= 5)
    median = np.median(ranks)
    worst = max(ranks)
    print(f"photo_log:      Rank=1: {n_correct}/8 | Rank<=5: {n_top5}/8 | Median: {median:.0f} | Worst: {worst}")
    
    print("=" * 120)
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    # Find best metric by median rank
    median_ranks = {}
    for key in metric_keys:
        ranks = [all_results[s]['epe_ranks'].get(key, 999) for s in all_results]
        median_ranks[key] = np.median(ranks)
    
    best_metric = min(median_ranks, key=median_ranks.get)
    best_median = median_ranks[best_metric]
    
    print(f"   Best metric by median rank: {best_metric}")
    print(f"   Median rank: {best_median:.0f} out of {n_configs}")
    print()
    
    if best_median <= 5:
        print("   ✓ Good: Metric's pick typically in top 5")
    elif best_median <= 10:
        print("   ~ Moderate: Metric's pick typically in top 10") 
    elif best_median <= n_configs / 4:
        print("   ~ Weak: Metric's pick typically in top quartile")
    else:
        print("   ✗ Poor: Metric doesn't reliably find good configs")


if __name__ == "__main__":
    main()
