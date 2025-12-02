#!/usr/bin/env python3
# File: scripts/weight_optimization_v2.py
"""
In-Process Weight Optimization (v2)

Major improvements over v1:
- Runs in-process (no subprocess spawning)
- Computes flows ONCE, reuses for all trials
- 30-50x faster per trial (~0.1 sec vs 2 sec)
- Clean Python API (no stdout parsing)

Usage:
    python scripts/weight_optimization_v2.py configs/example.toml --trials 100
    python scripts/weight_optimization_v2.py configs/example.toml --analyze
"""

import optuna
import pandas as pd
import numpy as np
import toml
import sys
from pathlib import Path
from datetime import datetime
import argparse
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble.oracle import compute_oracle_selection
from src.ensemble.selection import select_ensemble
from src.evaluation.ground_truth import compute_epe
from src.cache.experiment_cache import ExperimentCache, compute_image_hash, compute_config_hash

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================================
# Helper Functions (Module Level for Multiprocessing)
# ============================================================================

import cv2
import itertools
from src.synthesis import (
    generate_pattern_from_config,
    generate_flow_from_config,
    create_frame_pair
)
from src.optical_flow.flow_computation import compute_all_flows
from src.evaluation.self_supervised import compute_metrics_from_flows
from src.optical_flow.config_naming import generate_config_name
from src.utils.resampling import downsample_metrics, compute_downsample_stride, upsample_metrics
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return toml.load(f)


def expand_parameter_sweep(sweep_config: dict) -> list:
    algorithm = sweep_config.get('algorithm', 'farneback')
    if isinstance(algorithm, list):
        if len(algorithm) != 1:
            print(f"❌ ERROR: Multiple algorithms not supported")
            sys.exit(1)
        algorithm = algorithm[0]
    
    param_names = [k for k in sweep_config.keys() if k != 'algorithm']
    param_values = [sweep_config[k] if isinstance(sweep_config[k], list) else [sweep_config[k]]
                    for k in param_names]
    
    configs = []
    for combo in itertools.product(*param_values):
        config = {'algorithm': algorithm}
        for name, value in zip(param_names, combo):
            config[name] = value
        configs.append(config)
    return configs


def generate_perturbation_deltas(angles: list, magnitude: float) -> list:
    deltas = []
    for angle_deg in angles:
        angle_rad = np.deg2rad(angle_deg)
        dx = magnitude * np.cos(angle_rad)
        dy = magnitude * np.sin(angle_rad)
        deltas.append((float(dx), float(dy)))
    return deltas


def compute_config_worker(args):
    frame_A, frame_B, config, deltas, config_idx, n_configs = args
    config_name = generate_config_name(config)
    winsize = config.get('winsize', 15)
    
    flows = compute_all_flows(frame_A, frame_B, config, deltas, verbose=False)
    results = compute_metrics_from_flows(frame_A, frame_B, flows, config, verbose=False)
    stride = compute_downsample_stride(winsize)
    results_native = downsample_metrics(results, stride)
    results_native['config_name'] = config_name
    results_native['stride'] = stride
    
    return results_native


# ============================================================================
# Setup Functions (Run Once)
# ============================================================================

def setup_optimization(config_path: str):
    """
    One-time setup: generate test case and compute all flows.
    
    This is the expensive part (2-3 minutes with cache).
    But we only do it ONCE, then reuse for all trials.
    
    Returns:
        dict with:
            - results_full: List of config dicts with flows/metrics
            - oracle: Oracle results
            - u_true, v_true: Ground truth flows
            - valid_mask: Valid pixel mask
            - config: Full config dict
            - exp_cache: ExperimentCache instance
            - exp_dir: Experiment directory path
    """
    print("=" * 80)
    print("SETUP PHASE: Loading or computing flows")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    # Load config
    config = load_config(config_path)
    print(f"📂 Loaded config: {config_path}")
    
    # Extract configs
    image_config = config['image']
    flow_config = config['flow']
    sweep_config = config['parameter_sweep']
    perturbation_config = config.get('perturbations', {'angles': [0, 45, 90, 135], 'magnitude': 1.0})
    eval_config = config.get('evaluation', {})
    
    # Expand parameter sweep
    configs = expand_parameter_sweep(sweep_config)
    n_configs = len(configs)
    print(f"   {n_configs} configurations to evaluate")
    
    # Generate perturbations
    perturbation_angles = perturbation_config['angles']
    perturbation_magnitude = perturbation_config['magnitude']
    deltas = generate_perturbation_deltas(perturbation_angles, perturbation_magnitude)
    print(f"   {len(deltas)} perturbation directions")
    print()
    
    # ========================================================================
    # Generate test data
    # ========================================================================
    
    print("🎨 Generating test data...")
    
    # Generate pattern
    pattern = generate_pattern_from_config(image_config)
    
    # Generate flow
    flow_shape = pattern.shape[:2]
    u_true, v_true = generate_flow_from_config(flow_config, flow_shape)
    
    # Apply boundary margin
    boundary_margin = eval_config.get('boundary_margin', None)
    if boundary_margin is None:
        boundary_margin = max([c.get('winsize', 15) for c in configs])
    
    H, W = pattern.shape[:2]
    valid_mask = np.ones((H, W), dtype=bool)
    valid_mask[:boundary_margin, :] = False
    valid_mask[-boundary_margin:, :] = False
    valid_mask[:, :boundary_margin] = False
    valid_mask[:, -boundary_margin:] = False
    
    # Create frame pair
    frame1_original, frame2_original, warp_valid_mask = create_frame_pair(pattern, u_true, v_true)
    
    # Convert to grayscale if needed
    if len(frame1_original.shape) == 3:
        frame1 = cv2.cvtColor(frame1_original, cv2.COLOR_RGB2GRAY)
        frame2 = cv2.cvtColor(frame2_original, cv2.COLOR_RGB2GRAY)
    else:
        frame1 = frame1_original.copy()
        frame2 = frame2_original.copy()
    
    # Invalidate flow at edges
    u_true[~valid_mask] = np.nan
    v_true[~valid_mask] = np.nan
    
    print(f"   Image size: {H}×{W}")
    print(f"   Valid pixels: {valid_mask.sum()} ({100 * valid_mask.sum() / (H*W):.1f}%)")
    print(f"   Boundary margin: {boundary_margin} px")
    print()
    
    # ========================================================================
    # Setup experiment cache
    # ========================================================================
    
    print("📂 Setting up experiment cache...")
    exp_cache = ExperimentCache()
    of_type = sweep_config['algorithm']
    if isinstance(of_type, list):
        of_type = of_type[0]
    
    exp_dir, should_compute = exp_cache.setup_experiment(
        of_type, config, frame1, frame2, u_true, v_true, valid_mask
    )
    print()
    
    # ========================================================================
    # Compute all flows (THE EXPENSIVE PART) - or skip if cached
    # ========================================================================
    
    # NOTE: weight_optimization needs full results_full, not just sweep_df
    # So we still compute even if cache exists (for now)
    # Future: could serialize results_full to cache
    
    print("⚙️  Computing optical flows for all configs...")
    if not should_compute:
        print("   (Sweep results cached, but recomputing for optimization)")
    print()
    
    # Prepare worker arguments
    n_workers = eval_config.get('n_workers', cpu_count())
    worker_args = [
        (frame1, frame2, configs[i], deltas, i, n_configs)
        for i in range(n_configs)
    ]
    
    # Compute in parallel with progress bar
    flow_start = time.time()
    with Pool(n_workers) as pool:
        results_native = list(tqdm(
            pool.imap(compute_config_worker, worker_args),
            total=n_configs,
            desc="Computing flows",
            ncols=80
        ))
    flow_time = time.time() - flow_start
    
    print(f"   ✅ Computed all flows in {flow_time:.1f} seconds")
    print()
    
    # ========================================================================
    # Upsample to full resolution
    # ========================================================================
    
    print("📊 Upsampling to full resolution...")
    
    results_full = []
    for i, res_native in enumerate(results_native):
        res_full = upsample_metrics(res_native, (H, W))
        res_full['config_name'] = res_native['config_name']
        results_full.append(res_full)
    
    print(f"   ✅ Upsampled {n_configs} configs")
    print()
    
    # ========================================================================
    # Compute oracle (once)
    # ========================================================================
    
    print("🎯 Computing oracle (theoretical best)...")
    
    oracle = compute_oracle_selection(results_full, u_true, v_true, valid_mask)
    
    print(f"   Oracle forward EPE:   {oracle['oracle_epe_forward']:.6f} px")
    print(f"   Oracle symmetric EPE: {oracle['oracle_epe_symmetric']:.6f} px")
    print()
    
    # ========================================================================
    # Setup complete
    # ========================================================================
    
    setup_time = time.time() - start_time
    
    print("=" * 80)
    print(f"✅ SETUP COMPLETE ({setup_time:.1f} seconds)")
    print("=" * 80)
    print(f"   Ready for Optuna optimization")
    print(f"   Each trial will take ~0.1 seconds (no flow recomputation)")
    print("=" * 80)
    print()
    
    return {
        'results_full': results_full,
        'oracle': oracle,
        'u_true': u_true,
        'v_true': v_true,
        'valid_mask': valid_mask,
        'config': config,
        'n_configs': n_configs,
        'setup_time': setup_time,
        'exp_cache': exp_cache,
        'exp_dir': exp_dir,
    }


# ============================================================================
# Optuna Objective (Fast Loop)
# ============================================================================

def create_objective(setup_data: dict, total_weight: float = 4.0, discrete_step: float = 0.1):
    """
    Create Optuna objective function.
    
    This is a closure that captures setup_data, so trials can run fast.
    """
    
    results_full = setup_data['results_full']
    oracle = setup_data['oracle']
    u_true = setup_data['u_true']
    v_true = setup_data['v_true']
    valid_mask = setup_data['valid_mask']
    
    def objective(trial):
        """
        Optuna objective - runs for each trial.
        
        THIS IS FAST (~0.1 sec) because flows are precomputed!
        """
        
        # Sample raw weights from continuous space
        weight_names = [
            'traction_A', 'traction_B',
            'consistency_A', 'consistency_B',
            'photometric_A', 'photometric_B',
            'displ_A', 'displ_B'
        ]
        
        raw_weights = {}
        for name in weight_names:
            raw_weights[name] = trial.suggest_float(f'raw_{name}', 0.0, 2.0)
        
        # Normalize to sum = total_weight
        weight_sum = sum(raw_weights.values())
        if weight_sum == 0:
            normalized_weights = {name: total_weight / len(weight_names) for name in weight_names}
        else:
            normalized_weights = {name: val * total_weight / weight_sum
                                  for name, val in raw_weights.items()}
        
        # Round to nearest discrete_step
        discrete_weights = {name: round(val / discrete_step) * discrete_step
                            for name, val in normalized_weights.items()}
        
        # Fix rounding errors
        actual_sum = sum(discrete_weights.values())
        error = round((total_weight - actual_sum) / discrete_step) * discrete_step
        
        if abs(error) > 1e-6:
            max_name = max(discrete_weights.keys(), key=lambda k: discrete_weights[k])
            discrete_weights[max_name] = round(discrete_weights[max_name] + error, 1)
        
        # Map to actual config names
        weights = {
            'traction_A': discrete_weights['traction_A'],
            'traction_B': discrete_weights['traction_B'],
            'consistency_A': discrete_weights['consistency_A'],
            'consistency_B': discrete_weights['consistency_B'],
            'photometric_A': discrete_weights['photometric_A'],
            'photometric_B': discrete_weights['photometric_B'],
            'displacements_N2S_A2B': discrete_weights['displ_A'],
            'displacements_N2S_B2A': discrete_weights['displ_B'],
        }
        
        # Store weights
        for name, val in weights.items():
            trial.set_user_attr(f'weight_{name}', val)
        trial.set_user_attr('weight_sum', sum(weights.values()))
        
        # ====================================================================
        # THE FAST PART - just ensemble selection (no flow computation!)
        # ====================================================================
        
        trial_start = time.time()
        
        # Select ensemble (FAST - just array operations)
        ensemble = select_ensemble(results_full, weights, valid_mask)
        
        # Compute EPE
        EPE_ensemble = compute_epe(
            ensemble['u_ensemble_forward'],
            ensemble['v_ensemble_forward'],
            u_true,
            v_true
        )
        ensemble_epe = float(np.nanmean(EPE_ensemble[valid_mask]))
        
        trial_time = time.time() - trial_start
        
        # ====================================================================
        # Store metrics
        # ====================================================================
        
        trial.set_user_attr('ensemble_epe', ensemble_epe)
        trial.set_user_attr('oracle_epe', oracle['oracle_epe_forward'])
        trial.set_user_attr('trial_time', trial_time)
        
        # Calculate gap to oracle
        oracle_gap = ensemble_epe - oracle['oracle_epe_forward']
        trial.set_user_attr('oracle_gap', oracle_gap)
        
        if oracle['oracle_epe_forward'] > 0:
            oracle_gap_pct = oracle_gap / oracle['oracle_epe_forward'] * 100
            trial.set_user_attr('oracle_gap_pct', oracle_gap_pct)
        
        # Calculate oracle capture %
        # Oracle capture = how much of the (best_single - oracle) gap we closed
        best_epe_forward = float(np.nanmin([
            np.nanmean(compute_epe(r['u_AB'], r['v_AB'], u_true, v_true)[valid_mask])
            for r in results_full
        ]))
        
        if best_epe_forward > oracle['oracle_epe_forward']:
            capture = 100 * (best_epe_forward - ensemble_epe) / (best_epe_forward - oracle['oracle_epe_forward'])
            trial.set_user_attr('oracle_capture_pct', max(0, min(100, capture)))
        
        return ensemble_epe
    
    return objective


# ============================================================================
# Export and Analysis
# ============================================================================

def export_to_csv(study: optuna.Study, csv_path: Path) -> pd.DataFrame:
    """Export study results to CSV."""
    trials_data = []
    
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        
        row = {
            'trial_number': trial.number,
            'ensemble_epe': trial.user_attrs.get('ensemble_epe', None),
            'oracle_epe': trial.user_attrs.get('oracle_epe', None),
            'oracle_gap': trial.user_attrs.get('oracle_gap', None),
            'oracle_gap_pct': trial.user_attrs.get('oracle_gap_pct', None),
            'oracle_capture_pct': trial.user_attrs.get('oracle_capture_pct', None),
            'trial_time': trial.user_attrs.get('trial_time', None),
        }
        
        # Add weights
        for key in trial.user_attrs:
            if key.startswith('weight_'):
                row[key] = trial.user_attrs[key]
        
        trials_data.append(row)
    
    df = pd.DataFrame(trials_data)
    df.to_csv(csv_path, index=False)
    
    return df


# ============================================================================
# Main
# ============================================================================

def main(config_path: str,
         n_trials: int = 50,
         study_name: str = None,
         storage_path: str = None,
         total_weight: float = 4.0,
         discrete_step: float = 0.1):
    """
    Run weight optimization.
    
    Args:
        config_path: Path to config TOML
        n_trials: Number of trials to run
        study_name: Name for Optuna study
        storage_path: SQLite database path
        total_weight: Sum constraint for weights
        discrete_step: Discretization step (0.1 = increments of 0.1)
    """
    
    print("\n" + "=" * 80)
    print("WEIGHT OPTIMIZATION V2 (IN-PROCESS)")
    print("=" * 80)
    print(f"Config:       {config_path}")
    print(f"Study:        {study_name}")
    print(f"Trials:       {n_trials}")
    print(f"Total weight: {total_weight}")
    print(f"Discrete:     {discrete_step} steps")
    print("=" * 80)
    print()
    
    # ========================================================================
    # SETUP (expensive, but only once)
    # ========================================================================
    
    setup_data = setup_optimization(config_path)
    
    # ========================================================================
    # Create Optuna study
    # ========================================================================
    
    results_dir = Path('results') / 'optimization' / study_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Check if resuming
    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"📂 Resuming study with {n_existing} existing trials")
        if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) > 0:
            print(f"   Current best EPE: {study.best_value:.6f}")
        print()
    
    # ========================================================================
    # RUN OPTIMIZATION (fast loop!)
    # ========================================================================
    
    print("🚀 Starting optimization...")
    print("   Each trial should take ~0.1 seconds")
    print()
    
    start_time = datetime.now()
    n_start = len(study.trials)
    
    objective_func = create_objective(setup_data, total_weight, discrete_step)
    
    def progress_callback(study, trial):
        """Custom progress callback."""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            n_total = n_start + n_trials
            pct = n_completed / n_total * 100
            
            elapsed = (datetime.now() - start_time).total_seconds()
            avg_time = elapsed / (n_completed - n_start) if n_completed > n_start else 0
            remaining = avg_time * (n_total - n_completed)
            
            epe = trial.user_attrs.get('ensemble_epe', float('inf'))
            oracle_epe = trial.user_attrs.get('oracle_epe', None)
            trial_time = trial.user_attrs.get('trial_time', None)
            
            status = f"Trial {trial.number:3d}/{n_total} [{pct:5.1f}%] | EPE: {epe:.6f}"
            
            if oracle_epe and oracle_epe > 0:
                gap = epe - oracle_epe
                gap_pct = gap / oracle_epe * 100
                status += f" | Gap: {gap:.6f} ({gap_pct:+.1f}%)"
            
            if trial_time:
                status += f" | Time: {trial_time:.3f}s"
            
            status += f" | Best: {study.best_value:.6f}"
            status += f" | {remaining:.0f}s left"
            
            print(status)
    
    try:
        # Use custom tqdm wrapper for fixed width
        with tqdm(total=n_trials, desc="Optimizing", ncols=80, unit="trial") as pbar:
            def update_callback(study, trial):
                pbar.set_postfix({"Best": f"#{study.best_trial.number} {study.best_value:.6f}"})
                pbar.update(1)
            
            study.optimize(
                objective_func,
                n_trials=n_trials,
                show_progress_bar=False,
                callbacks=[update_callback]
            )
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
    
    # ========================================================================
    # EXPORT AND REPORT
    # ========================================================================
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    # Export trials CSV
    results_csv = results_dir / 'trials.csv'
    df = export_to_csv(study, results_csv)
    print(f"✅ Exported {len(df)} trials to {results_csv}")
    
    # Copy results to experiment cache directory
    exp_cache = setup_data['exp_cache']
    exp_dir = setup_data['exp_dir']
    
    print(f"📂 Saving optimization results to cache...")
    import shutil
    shutil.copy2(results_csv, exp_dir / 'optimization_trials.csv')
    
    # Extract DB filename from SQLite URI (format: "sqlite:///filename.db")
    db_filename = storage_path.replace('sqlite:///', '')
    if Path(db_filename).exists():
        shutil.copy2(db_filename, exp_dir / 'optuna_study.db')
        print(f"   ✅ Saved to {exp_dir}")
    else:
        print(f"   ⚠️  Optuna DB not found: {db_filename}")
    
    print(f"\n⏱️  Total time: {total_time:.1f} seconds")
    print(f"   Setup: {setup_data['setup_time']:.1f} seconds")
    print(f"   Optimization: {total_time - setup_data['setup_time']:.1f} seconds")
    print(f"   Avg per trial: {(total_time - setup_data['setup_time']) / n_trials:.3f} seconds")
    
    # Print best results
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if len(complete_trials) > 0:
        print(f"\n🏆 Best Trial (#{study.best_trial.number}):")
        print(f"   EPE: {study.best_value:.6f}")
        
        if 'oracle_epe' in study.best_trial.user_attrs:
            oracle_epe = study.best_trial.user_attrs['oracle_epe']
            oracle_gap = study.best_value - oracle_epe
            print(f"   Oracle EPE: {oracle_epe:.6f}")
            print(f"   Gap to oracle: {oracle_gap:.6f} ({oracle_gap / oracle_epe * 100:+.1f}%)")
        
        if 'oracle_capture_pct' in study.best_trial.user_attrs:
            print(f"   Oracle capture: {study.best_trial.user_attrs['oracle_capture_pct']:.1f}%")
        
        print(f"\n   Optimal weights:")
        weight_keys = ['traction_A', 'traction_B', 'consistency_A', 'consistency_B',
                       'photometric_A', 'photometric_B', 'displacements_N2S_A2B', 'displacements_N2S_B2A']
        
        for key in weight_keys:
            user_attr_key = f'weight_{key}'
            if user_attr_key in study.best_trial.user_attrs:
                val = study.best_trial.user_attrs[user_attr_key]
                print(f"      {key:25s}: {val:.1f}")
    
    print("=" * 80)
    print()
    
    return study, df


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Optimize ensemble weights (In-Process Version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 50 trials
  python scripts/weight_optimization_v2.py configs/example.toml --trials 50

  # Resume previous run
  python scripts/weight_optimization_v2.py configs/example.toml --trials 20
        """
    )
    
    parser.add_argument('config',
                        help='Path to config TOML file')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of trials to run (default: 50)')
    parser.add_argument('--total-weight', type=float, default=4.0,
                        help='Sum constraint for weights (default: 4.0)')
    parser.add_argument('--discrete-step', type=float, default=0.1,
                        help='Discretization step (default: 0.1)')
    parser.add_argument('--study-name', default=None,
                        help='Study name (default: auto from config)')
    parser.add_argument('--storage', default=None,
                        help='Optuna storage (default: sqlite:///optuna_{study_name}.db)')
    
    args = parser.parse_args()
    
    # Generate study name from config if not provided
    if args.study_name is None:
        config_name = Path(args.config).stem
        args.study_name = f"weights_{config_name}_v2"
    
    # Generate storage path if not provided
    if args.storage is None:
        args.storage = f"sqlite:///optuna_{args.study_name}.db"
    
    study, df = main(
        config_path=args.config,
        n_trials=args.trials,
        study_name=args.study_name,
        storage_path=args.storage,
        total_weight=args.total_weight,
        discrete_step=args.discrete_step
    )
