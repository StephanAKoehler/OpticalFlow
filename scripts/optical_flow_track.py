#!/usr/bin/env python3
# File: scripts/optical_flow_track.py
"""
Optical Flow Parameter Sweep

Runs OF algorithm sweep on movie frames and computes metrics.
No optimization - that's handled by optimize_weights.py.

Usage:
    python scripts/optical_flow_track.py config.toml --movie-hash abc123
    python scripts/optical_flow_track.py config.toml --movie-hash abc123 --data-dir data/
    python scripts/optical_flow_track.py config.toml --movie-hash abc123 --no-cache
"""

import sys
import hashlib
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import tomli
import tomli_w

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.data_loader import load_movie_sequence, MovieSequence, FramePair
from src.core.data_structures import (
    create_sweep_config,
    expand_sweep_configs,
    flatten_for_visualization
)
from src.core.sweep import compute_sweep, upsample_and_compute_oracle
from src.evaluation.ground_truth import compute_epe


# =============================================================================
# Configuration Extraction
# =============================================================================

OF_SECTIONS = ['parameter_sweep', 'perturbations']


def extract_of_config(config: dict) -> dict:
    """
    Extract OF-relevant sections from full config.
    
    Args:
        config: Full TOML config dict
        
    Returns:
        Dict with only OF sections
    """
    of_config = {}
    
    for section in OF_SECTIONS:
        if section in config:
            of_config[section] = config[section]
    
    return of_config


def compute_of_hash(of_config: dict) -> str:
    """
    Compute deterministic hash from OF config.
    
    Args:
        of_config: OF sections only
        
    Returns:
        12-character hash string
    """
    config_str = tomli_w.dumps(of_config)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def validate_of_config(of_config: dict):
    """
    Validate OF config has required sections.
    
    Exits with error if validation fails.
    """
    if 'parameter_sweep' not in of_config:
        print("❌ ERROR: Missing [parameter_sweep] section in config")
        sys.exit(1)
    
    if 'algorithm' not in of_config['parameter_sweep']:
        print("❌ ERROR: [parameter_sweep] must have 'algorithm' field")
        sys.exit(1)
    
    # Validate perturbations
    pert = of_config.get('perturbations', {})
    directions = pert.get('directions', 2)
    magnitude = pert.get('magnitude', 1)
    
    if directions not in [2, 4, 8]:
        print(f"❌ ERROR: [perturbations] directions must be 2, 4, or 8, got {directions}")
        sys.exit(1)
    
    if not isinstance(magnitude, int) or magnitude < 1:
        print(f"❌ ERROR: [perturbations] magnitude must be int >= 1, got {magnitude}")
        sys.exit(1)


def get_epe_power(config: dict) -> float:
    """Extract epe_power from config, with validation."""
    eval_config = config.get('evaluation', {})
    epe_power = eval_config.get('epe_power')
    
    if epe_power is None:
        print("❌ ERROR: [evaluation] section must specify epe_power")
        sys.exit(1)
    
    return float(epe_power)


# =============================================================================
# Perturbation Generation
# =============================================================================

def generate_perturbation_deltas(directions: int, magnitude: int) -> list:
    """
    Generate perturbation vectors.
    
    Args:
        directions: Number of directions (2, 4, or 8)
        magnitude: Magnitude in pixels
        
    Returns:
        List of (dx, dy) tuples
    """
    if directions == 2:
        return [(magnitude, 0), (0, magnitude)]
    elif directions == 4:
        return [
            (magnitude, 0),
            (0, magnitude),
            (magnitude, magnitude),
            (magnitude, -magnitude)
        ]
    elif directions == 8:
        return [
            (magnitude, 0),
            (0, magnitude),
            (magnitude, magnitude),
            (magnitude, -magnitude),
            (-magnitude, 0),
            (0, -magnitude),
            (-magnitude, magnitude),
            (-magnitude, -magnitude)
        ]
    else:
        print(f"❌ ERROR: Invalid directions={directions}")
        sys.exit(1)


# =============================================================================
# Boundary Margin Computation
# =============================================================================

def compute_boundary_margin(config: dict, configs: list) -> int:
    """
    Compute boundary margin based on OF window sizes and perturbation magnitude.
    
    Args:
        config: Full config dict
        configs: List of expanded OF configs
        
    Returns:
        Boundary margin in pixels
    """
    # Check if manually specified
    eval_config = config.get('evaluation', {})
    if 'boundary_margin' in eval_config:
        return eval_config['boundary_margin']
    
    # Auto-compute from configs
    from src_contamination import get_margin
    
    pert_config = config.get('perturbations', {})
    magnitude = float(pert_config.get('magnitude', 1))
    
    margins = [get_margin(c, magnitude) for c in configs]
    return max(margins)


# =============================================================================
# Single Pair Processing
# =============================================================================

def process_pair(
    pair: FramePair,
    pair_idx: int,
    configs: list,
    deltas: list,
    epe_power: float,
    output_dir: Path,
    n_workers: Optional[int] = None,
    no_cache: bool = False
) -> dict:
    """
    Process a single frame pair: run sweep and compute oracle.
    
    Args:
        pair: FramePair to process
        pair_idx: Index of this pair in sequence
        configs: List of OF parameter configs
        deltas: Perturbation vectors
        epe_power: Power for EPE computation
        output_dir: Output directory for this pair
        n_workers: Number of parallel workers
        no_cache: Force recomputation
        
    Returns:
        Dict with sweep results and oracle
    """
    pair_dir = output_dir / f'pair_{pair_idx:03d}'
    pair_dir.mkdir(parents=True, exist_ok=True)
    
    # Check cache
    results_path = pair_dir / 'results_full.pkl'
    oracle_path = pair_dir / 'oracle.npz'
    
    if results_path.exists() and oracle_path.exists() and not no_cache:
        print(f"   📦 Loading pair {pair_idx} from cache...")
        
        with open(results_path, 'rb') as f:
            results_full = pickle.load(f)
        
        oracle_data = np.load(oracle_path)
        oracle = {k: oracle_data[k] for k in oracle_data.files}
        # Convert scalar arrays back to floats
        for k in ['oracle_epe_forward', 'oracle_epe_symmetric', 
                  'oracle_epe_forward_powered', 'oracle_epe_symmetric_powered']:
            if k in oracle:
                oracle[k] = float(oracle[k])
        
        return {
            'results_full': results_full,
            'oracle': oracle,
            'pair_dir': pair_dir,
            'cached': True
        }
    
    print(f"   ⚙️  Computing pair {pair_idx}...")
    
    # Compute sweep at native resolution
    # Pass original frames for RGB photometric metrics
    results_native = compute_sweep(
        pair.frame1,
        pair.frame2,
        configs,
        deltas,
        n_workers=n_workers,
        frame1_original=pair.frame1_original,
        frame2_original=pair.frame2_original
    )
    
    # Upsample and compute oracle
    H, W = pair.metadata['H'], pair.metadata['W']
    
    if pair.has_gt:
        sweep_results = upsample_and_compute_oracle(
            results_native,
            H, W,
            pair.u_truth,
            pair.v_truth,
            pair.valid_mask,
            epe_power=epe_power
        )
        results_full = sweep_results['results_full']
        oracle = sweep_results['oracle']
    else:
        # No GT - just upsample
        from src.utils.resampling import upsample_metrics
        from src.core.data_structures import create_result_dict
        
        results_full = []
        for res_native in results_native:
            flows_full = upsample_metrics(res_native['flows'], (H, W))
            metrics_full = upsample_metrics(res_native['metrics'], (H, W))
            
            params_with_algo = {
                'algorithm': res_native['metadata']['algorithm'],
                **res_native['params']
            }
            
            res_full = create_result_dict(
                params=params_with_algo,
                flows=flows_full,
                metrics=metrics_full,
                config_name=res_native['metadata']['config_name'],
                config_id=res_native['metadata']['config_id']
            )
            res_full['metadata']['stride'] = res_native['metadata']['stride']
            results_full.append(res_full)
        
        oracle = None
    
    # Save results
    with open(results_path, 'wb') as f:
        pickle.dump(results_full, f)
    
    if oracle is not None:
        np.savez(oracle_path, **oracle)
    
    # Save sweep summary CSV
    save_sweep_summary(results_full, oracle, pair, epe_power, pair_dir)
    
    return {
        'results_full': results_full,
        'oracle': oracle,
        'pair_dir': pair_dir,
        'cached': False
    }


def save_sweep_summary(results_full: list, oracle: dict, pair: FramePair, 
                       epe_power: float, output_dir: Path):
    """Save sweep summary as CSV."""
    import csv
    
    rows = []
    
    for i, result in enumerate(results_full):
        row = {
            'config_idx': i,
            'config_name': result['metadata']['config_name'],
            'algorithm': result['metadata']['algorithm'],
        }
        
        # Add params
        for k, v in result['params'].items():
            row[f'param_{k}'] = v
        
        # Compute EPE if GT available
        if pair.has_gt:
            epe_map = compute_epe(
                result['flows']['u_AB'],
                result['flows']['v_AB'],
                pair.u_truth,
                pair.v_truth,
                pair.valid_mask,
                power=epe_power
            )
            row['mean_epe_powered'] = float(np.nanmean(epe_map))
            row['std_epe_powered'] = float(np.nanstd(epe_map[pair.valid_mask]))
        
        rows.append(row)
    
    # Write CSV
    csv_path = output_dir / 'sweep_results.csv'
    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


# =============================================================================
# Figure Generation
# =============================================================================

def generate_sweep_figures(movie: MovieSequence, analysis_dir: Path, epe_power: float):
    """
    Generate per-pair sweep visualization figures.
    
    Args:
        movie: MovieSequence with processed pairs
        analysis_dir: Analysis output directory
        epe_power: EPE power used
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    figures_dir = analysis_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Per-pair EPE summary
    if movie.metadata['has_gt']:
        n_pairs = len(movie.pairs)
        
        # Collect oracle EPEs per pair
        oracle_epes = []
        for pair_idx in range(n_pairs):
            oracle_path = analysis_dir / 'sweep' / f'pair_{pair_idx:03d}' / 'oracle.npz'
            if oracle_path.exists():
                data = np.load(oracle_path)
                if 'oracle_epe_forward_powered' in data:
                    oracle_epes.append(float(data['oracle_epe_forward_powered']))
                elif 'oracle_epe_forward' in data:
                    oracle_epes.append(float(data['oracle_epe_forward']))
        
        if oracle_epes:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(range(len(oracle_epes)), oracle_epes, color='steelblue', alpha=0.7)
            ax.set_xlabel('Pair Index')
            ax.set_ylabel(f'Oracle EPE^{epe_power}')
            ax.set_title(f'Oracle EPE per Frame Pair (mean={np.mean(oracle_epes):.4f})')
            ax.axhline(np.mean(oracle_epes), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(oracle_epes):.4f}')
            ax.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / 'oracle_epe_per_pair.png', dpi=150)
            plt.close()
            
            print(f"   📊 Saved oracle EPE figure")


# =============================================================================
# Main Sweep Function
# =============================================================================

def run_sweep(
    config: dict,
    movie_hash: str,
    data_dir: Path = Path('data'),
    no_cache: bool = False,
    n_workers: Optional[int] = None
) -> str:
    """
    Run OF parameter sweep on all frame pairs in a movie.
    
    Args:
        config: Full TOML config dict
        movie_hash: Hash of the movie to process
        data_dir: Base data directory
        no_cache: Force recomputation
        n_workers: Number of parallel workers
        
    Returns:
        of_hash: Hash identifying this OF configuration
    """
    # Extract and validate OF config
    of_config = extract_of_config(config)
    validate_of_config(of_config)
    
    # Get epe_power
    epe_power = get_epe_power(config)
    
    # Compute hash
    of_hash = compute_of_hash(of_config)
    
    # Setup paths
    movie_dir = data_dir / movie_hash
    analysis_dir = movie_dir / 'analysis' / of_hash
    sweep_dir = analysis_dir / 'sweep'
    
    print("=" * 80)
    print("⚙️  OPTICAL FLOW PARAMETER SWEEP")
    print("=" * 80)
    print(f"Movie hash: {movie_hash}")
    print(f"OF hash: {of_hash}")
    print(f"Output: {analysis_dir}")
    print()
    
    # Validate movie exists
    if not movie_dir.exists():
        print(f"❌ ERROR: Movie directory not found: {movie_dir}")
        sys.exit(1)
    
    # Create directories
    analysis_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir.mkdir(exist_ok=True)
    
    # Save OF config
    with open(analysis_dir / 'optical_flow.toml', 'wb') as f:
        tomli_w.dump(of_config, f)
    
    # Build sweep configs
    sweep_config = create_sweep_config(config)
    configs = expand_sweep_configs(sweep_config)
    n_configs = len(configs)
    
    # Setup perturbations
    pert_config = of_config.get('perturbations', {'directions': 2, 'magnitude': 1})
    directions = pert_config.get('directions', 2)
    magnitude = pert_config.get('magnitude', 1)
    deltas = generate_perturbation_deltas(directions, magnitude)
    
    print(f"📋 Sweep configuration:")
    print(f"   Algorithm: {sweep_config['algorithm']}")
    print(f"   Configurations: {n_configs}")
    print(f"   Perturbations: {len(deltas)} directions, magnitude={magnitude}px")
    print(f"   EPE power: {epe_power}")
    print()
    
    # Compute boundary margin
    boundary_margin = compute_boundary_margin(config, configs)
    print(f"   Boundary margin: {boundary_margin}px")
    print()
    
    # Load movie
    print("📂 Loading movie sequence...")
    movie = load_movie_sequence(movie_dir, boundary_margin=boundary_margin)
    n_pairs = len(movie.pairs)
    
    print(f"   Frames: {movie.metadata['n_frames']}")
    print(f"   Pairs: {n_pairs}")
    print(f"   Ground truth: {'Available' if movie.metadata['has_gt'] else 'Not available'}")
    print()
    
    # Process each pair
    print("=" * 80)
    print(f"🔄 PROCESSING {n_pairs} FRAME PAIRS")
    print("=" * 80)
    print()
    
    pair_results = []
    
    for pair_idx, pair in enumerate(movie.pairs):
        print(f"[Pair {pair_idx + 1}/{n_pairs}]")
        
        result = process_pair(
            pair=pair,
            pair_idx=pair_idx,
            configs=configs,
            deltas=deltas,
            epe_power=epe_power,
            output_dir=sweep_dir,
            n_workers=n_workers,
            no_cache=no_cache
        )
        pair_results.append(result)
        
        # Print oracle EPE if available
        if result['oracle'] is not None:
            oracle_epe = result['oracle'].get('oracle_epe_forward_powered', 
                                               result['oracle'].get('oracle_epe_forward'))
            cached_str = " (cached)" if result['cached'] else ""
            print(f"   Oracle EPE^{epe_power}: {oracle_epe:.6f}{cached_str}")
        
        print()
    
    # Generate figures
    print("📊 Generating figures...")
    generate_sweep_figures(movie, analysis_dir, epe_power)
    print()
    
    # Summary
    print("=" * 80)
    print("✅ SWEEP COMPLETE")
    print("=" * 80)
    print(f"Output: {analysis_dir}")
    print(f"Pairs processed: {n_pairs}")
    print(f"Configs per pair: {n_configs}")
    
    if movie.metadata['has_gt']:
        # Compute mean oracle EPE
        oracle_epes = []
        for result in pair_results:
            if result['oracle'] is not None:
                epe = result['oracle'].get('oracle_epe_forward_powered',
                                           result['oracle'].get('oracle_epe_forward'))
                oracle_epes.append(epe)
        
        if oracle_epes:
            print(f"Mean oracle EPE^{epe_power}: {np.mean(oracle_epes):.6f} ± {np.std(oracle_epes):.6f}")
    
    print(f"OF hash: {of_hash}")
    print("=" * 80)
    
    return of_hash


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run optical flow parameter sweep on movie frames'
    )
    parser.add_argument('config', type=Path, help='TOML configuration file')
    parser.add_argument('--movie-hash', type=str, required=True,
                       help='Hash of movie to process')
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                       help='Base data directory (default: data/)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Force recomputation, ignore cache')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Validate config exists
    if not args.config.exists():
        print(f"❌ ERROR: Config file not found: {args.config}")
        sys.exit(1)
    
    # Load config
    with open(args.config, 'rb') as f:
        config = tomli.load(f)
    
    # Run sweep
    of_hash = run_sweep(
        config=config,
        movie_hash=args.movie_hash,
        data_dir=args.data_dir,
        no_cache=args.no_cache,
        n_workers=args.workers
    )
    
    # Print hash for scripting
    print()
    print(f"OF_HASH={of_hash}")


if __name__ == "__main__":
    main()
