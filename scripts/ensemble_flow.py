#!/usr/bin/env python3
# File: scripts/ensemble_flow.py
"""
Unified Ensemble Flow Pipeline

Computes optical flow parameter sweep and generates ensemble flows.
Supports both fixed weights (from config) and weight optimization.

Usage:
    # Evaluate with config weights
    ensemble_flow.py config.toml
    
    # Optimize weights (if no weights in config, or forced)
    ensemble_flow.py config.toml --optimize --trials 100
    
    # Cache control
    ensemble_flow.py config.toml --no-cache
"""

import argparse
import sys
import toml
import numpy as np
import itertools
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Core modules
from src.core.setup import (
    compute_weight_hash,
    setup_test_data,
    setup_experiment_cache,
    get_next_optim_number,
    get_latest_optim_weights,
    weights_are_equal
)
from src.core.sweep import (
    compute_sweep,
    upsample_and_compute_oracle,
    save_sweep_to_cache,
    load_sweep_from_cache
)

# Ensemble
from src.ensemble.selection import select_ensemble
from src.evaluation.ground_truth import compute_epe
from src.evaluation.error_summary import print_error_distribution
from src.optical_flow.flow_deformation import compute_bidirectional_deformation

# Visualization
from src.visualization.sweep_figures import generate_all_sweep_figures
from src.visualization.optimization_figures import generate_all_optimization_figures
from src.visualization.flow_field_comparison import visualize_flow_field_comparison


def expand_parameter_sweep(sweep_config: dict) -> list:
    """Expand parameter sweep into list of configurations."""
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


def extract_swept_params(sweep_config: dict) -> list:
    """
    Extract parameter names that are actually being swept (have multiple values).
    
    Args:
        sweep_config: The [parameter_sweep] section from TOML
        
    Returns:
        List of parameter names that have more than one value
    """
    swept_params = []
    for key, value in sweep_config.items():
        if key == 'algorithm':
            continue
        # Check if this parameter has multiple values
        if isinstance(value, list) and len(value) > 1:
            swept_params.append(key)
    return swept_params


def generate_perturbation_deltas(directions: int, magnitude: int) -> list:
    """
    Generate perturbation vectors evenly spaced over half-circle [0°, 180°).
    
    Half-circle avoids redundancy since ±δ perturbations cover both directions.
    
    Args:
        directions: Number of directions (int >= 2)
        magnitude: Perturbation magnitude in pixels (int >= 1)
        
    Returns:
        List of (dx, dy) tuples
        
    Examples:
        directions=2 → [0°, 90°]
        directions=4 → [0°, 45°, 90°, 135°]
    """
    assert isinstance(directions, int) and directions >= 2, \
        f"directions must be int >= 2, got {directions}"
    assert isinstance(magnitude, int) and magnitude >= 1, \
        f"magnitude must be int >= 1, got {magnitude}"
    
    deltas = []
    for i in range(directions):
        angle_deg = i * 180.0 / directions
        angle_rad = np.deg2rad(angle_deg)
        dx = magnitude * np.cos(angle_rad)
        dy = magnitude * np.sin(angle_rad)
        deltas.append((float(dx), float(dy)))
    return deltas


def generate_flow_field_comparison(
    results_full,
    sweep_df,
    ensemble_results,
    test_data,
    oracle,
    figures_dir
):
    """
    Generate comprehensive flow field comparison figure.
    
    Shows Oracle vs Ensemble vs Best Single in 3x3 layout.
    
    Returns:
        dict: Flow data for reuse in other figures {
            'u_oracle_fwd', 'v_oracle_fwd',
            'u_oracle_sym', 'v_oracle_sym',
            'u_best_fwd', 'v_best_fwd',
            'u_best_sym', 'v_best_sym',
            'best_idx'
        }
    """
    # Flatten results if needed
    from src.core.data_structures import flatten_for_visualization
    if len(results_full) > 0 and 'metadata' in results_full[0]:
        results_flat = [flatten_for_visualization(r) for r in results_full]
    else:
        results_flat = results_full
    
    # Find best single config
    best_idx = sweep_df['mean_epe_forward'].idxmin()
    best_config_name = sweep_df.loc[best_idx, 'config_name']
    
    # Extract best single flows (forward)
    u_best_fwd = results_flat[best_idx]['u_AB']
    v_best_fwd = results_flat[best_idx]['v_AB']
    
    # Extract best single flows (symmetric)
    u_best_sym = results_flat[best_idx]['u_sym_A']
    v_best_sym = results_flat[best_idx]['v_sym_A']
    
    # Extract ensemble flows
    u_ensemble = ensemble_results['u_ensemble_forward']
    v_ensemble = ensemble_results['v_ensemble_forward']
    ensemble_selection = ensemble_results['ensemble_selection']
    
    # Compute oracle flows and selection (FORWARD)
    n_configs = len(results_flat)
    H, W = test_data['u_truth'].shape
    
    # Stack all configs (forward)
    u_stack_fwd = np.stack([results_flat[i]['u_AB'] for i in range(n_configs)], axis=0)
    v_stack_fwd = np.stack([results_flat[i]['v_AB'] for i in range(n_configs)], axis=0)
    
    # Compute EPE for each config at each pixel (forward)
    epe_stack_fwd = np.sqrt(
        (u_stack_fwd - test_data['u_truth'][np.newaxis, :, :])**2 + 
        (v_stack_fwd - test_data['v_truth'][np.newaxis, :, :])**2
    )
    
    # Find best config per pixel (oracle selection forward)
    oracle_selection = np.argmin(epe_stack_fwd, axis=0)
    
    # Build oracle flow (forward)
    u_oracle_fwd = np.zeros((H, W), dtype=np.float32)
    v_oracle_fwd = np.zeros((H, W), dtype=np.float32)
    
    for i in range(n_configs):
        mask = oracle_selection == i
        u_oracle_fwd[mask] = u_stack_fwd[i][mask]
        v_oracle_fwd[mask] = v_stack_fwd[i][mask]
    
    # Compute oracle flows (SYMMETRIC)
    u_stack_sym = np.stack([results_flat[i]['u_sym_A'] for i in range(n_configs)], axis=0)
    v_stack_sym = np.stack([results_flat[i]['v_sym_A'] for i in range(n_configs)], axis=0)
    
    epe_stack_sym = np.sqrt(
        (u_stack_sym - test_data['u_truth'][np.newaxis, :, :])**2 + 
        (v_stack_sym - test_data['v_truth'][np.newaxis, :, :])**2
    )
    
    oracle_selection_sym = np.argmin(epe_stack_sym, axis=0)
    
    u_oracle_sym = np.zeros((H, W), dtype=np.float32)
    v_oracle_sym = np.zeros((H, W), dtype=np.float32)
    
    for i in range(n_configs):
        mask = oracle_selection_sym == i
        u_oracle_sym[mask] = u_stack_sym[i][mask]
        v_oracle_sym[mask] = v_stack_sym[i][mask]
    
    # Generate the comparison figure
    visualize_flow_field_comparison(
        u_best_fwd, v_best_fwd,
        u_ensemble, v_ensemble,
        u_oracle_fwd, v_oracle_fwd,
        test_data['u_truth'], test_data['v_truth'],
        test_data['valid_mask'],
        best_config_name=best_config_name,
        frame_A=test_data['frame1_original'],
        ensemble_selection=ensemble_selection,
        oracle_selection=oracle_selection,
        n_configs=n_configs,
        best_config_idx=best_idx,
        save_path=str(figures_dir / 'flow_field_comparison.png'),
        dpi=150
    )
    
    # Return flow data for reuse
    return {
        'u_oracle_fwd': u_oracle_fwd,
        'v_oracle_fwd': v_oracle_fwd,
        'u_oracle_sym': u_oracle_sym,
        'v_oracle_sym': v_oracle_sym,
        'u_best_fwd': u_best_fwd,
        'v_best_fwd': v_best_fwd,
        'u_best_sym': u_best_sym,
        'v_best_sym': v_best_sym,
        'best_idx': best_idx
    }


def determine_mode(config: dict, args) -> str:
    """
    Determine execution mode: 'optimize' or 'evaluate'
    
    Priority:
    1. If --optimize flag: optimize
    2. If config has weights: evaluate
    3. Else: optimize (no weights specified)
    """
    if args.optimize:
        return 'optimize'
    
    if 'ensemble' in config and 'weights' in config['ensemble']:
        weights = config['ensemble']['weights']
        # Check if weights section is non-empty
        if weights and any(v is not None for v in weights.values()):
            return 'evaluate'
    
    return 'optimize'


def main():
    parser = argparse.ArgumentParser(
        description='Unified Ensemble Flow Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with config weights
  ensemble_flow.py config.toml
  
  # Optimize weights
  ensemble_flow.py config.toml --optimize --trials 100
  
  # Force recomputation
  ensemble_flow.py config.toml --no-cache
        """
    )
    
    parser.add_argument('config', help='Path to config TOML file')
    
    # Mode control
    parser.add_argument('--optimize', action='store_true',
                       help='Force weight optimization (ignore config weights)')
    parser.add_argument('--trials', type=int,
                       help='Number of optimization trials (overrides config)')
    
    # Cache control
    parser.add_argument('--no-cache', action='store_true',
                       help='Force recomputation (ignore cached sweep)')
    parser.add_argument('--clear-cache', metavar='HASH',
                       help='Clear specific experiment cache')
    
    # Visualization
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip figure generation')
    
    # Advanced
    parser.add_argument('--cache-size', type=int, default=1000,
                       help='Flow cache size (default: 1000)')
    parser.add_argument('--cache-verbose', action='store_true',
                       help='Verbose cache logging')
    
    args = parser.parse_args()
    
    # Load config
    print("=" * 80)
    print("ENSEMBLE FLOW PIPELINE")
    print("=" * 80)
    print(f"Config: {args.config}")
    
    with open(args.config, 'r') as f:
        config = toml.load(f)
    
    # Determine mode
    mode = determine_mode(config, args)
    print(f"Mode: {mode.upper()}")
    print("=" * 80)
    print()
    
    # Execute appropriate mode
    if mode == 'optimize':
        run_optimization_mode(config, args)
    else:
        run_evaluation_mode(config, args)


def run_evaluation_mode(config: dict, args):
    """Run ensemble evaluation with config weights."""
    print("🎯 EVALUATION MODE")
    print("   Using weights from config")
    print()
    
    # Extract config sections
    sweep_config = config['parameter_sweep']
    perturbation_config = config.get('perturbations', {'directions': 2, 'magnitude': 1})
    eval_config = config.get('evaluation', {})
    ensemble_config = config.get('ensemble', {})
    weights_config = ensemble_config.get('weights', {})
    
    # ========================================================================
    # Validate perturbation config (fail early)
    # ========================================================================
    directions = perturbation_config.get('directions', 2)
    magnitude = perturbation_config.get('magnitude', 1)
    
    if not isinstance(directions, int) or directions < 2:
        print(f"❌ ERROR: [perturbations] directions must be int >= 2, got {directions}")
        sys.exit(1)
    
    if not isinstance(magnitude, int) or magnitude < 1:
        print(f"❌ ERROR: [perturbations] magnitude must be int >= 1, got {magnitude}")
        sys.exit(1)
    
    # Expand parameter sweep
    configs = expand_parameter_sweep(sweep_config)
    n_configs = len(configs)
    print(f"📋 Configurations: {n_configs}")
    
    # Extract swept parameters (those with multiple values)
    swept_params = extract_swept_params(sweep_config)
    print(f"📋 Swept parameters: {swept_params if swept_params else 'None (single config)'}")
    print()
    
    # Generate perturbations
    deltas = generate_perturbation_deltas(directions, magnitude)
    print(f"📋 Perturbations: {len(deltas)} directions, magnitude={magnitude}px")
    print()
    
    # ========================================================================
    # Setup: Images, cache, hashing
    # ========================================================================
    
    test_data = setup_test_data(config, eval_config, configs)
    cache_info = setup_experiment_cache(config, test_data, args.no_cache)
    
    exp_cache = cache_info['exp_cache']
    should_compute = cache_info['should_compute']
    
    # ========================================================================
    # Compute or load sweep
    # ========================================================================
    
    if should_compute:
        # Compute sweep
        results_native = compute_sweep(
            test_data['frame1'],
            test_data['frame2'],
            configs,
            deltas,
            n_workers=eval_config.get('n_workers', None)
        )
        
        # Get EPE power from config
        epe_power = config.get('evaluation', {}).get('epe_power', 2.0)
        
        # Upsample and compute oracle
        sweep_results = upsample_and_compute_oracle(
            results_native,
            test_data['H'], test_data['W'],
            test_data['u_truth'], test_data['v_truth'],
            test_data['valid_mask'],
            epe_power=epe_power
        )
        
        results_full = sweep_results['results_full']
        oracle = sweep_results['oracle']
        
        # Save to cache
        save_sweep_to_cache(
            exp_cache,
            results_full,
            test_data['u_truth'], test_data['v_truth'],
            test_data['valid_mask'],
            oracle['oracle_epe_forward'],
            oracle['oracle_epe_symmetric']
        )
    else:
        # Load from cache
        cached = load_sweep_from_cache(exp_cache)
        results_full = cached['results_full']
        
        if results_full is None:
            print("❌ ERROR: Full results not in cache, cannot generate ensemble")
            print("   Run with --no-cache to recompute")
            sys.exit(1)
    
    # ========================================================================
    # Generate ensemble with config weights
    # ========================================================================
    
    print("🎯 Generating ensemble with config weights...")
    
    ensemble_weights = {
        'traction_A': weights_config.get('traction_A', 0.0),
        'traction_B': weights_config.get('traction_B', 0.0),
        'consistency_A': weights_config.get('consistency_A', 0.0),
        'consistency_B': weights_config.get('consistency_B', 0.0),
        'photometric_A': weights_config.get('photometric_A', 0.0),
        'photometric_B': weights_config.get('photometric_B', 0.0),
        'displacements_N2S_A2B': weights_config.get('displacements_N2S_A2B', 0.0),
        'displacements_N2S_B2A': weights_config.get('displacements_N2S_B2A', 0.0),
    }
    
    print(f"   Weights: {ensemble_weights}")
    
    ensemble_results = select_ensemble(results_full, ensemble_weights, test_data['valid_mask'])
    
    # Compute ensemble EPE with configured power
    u_ens = ensemble_results['u_ensemble_forward']
    v_ens = ensemble_results['v_ensemble_forward']
    epe_power = config.get('evaluation', {}).get('epe_power', 2.0)
    
    epe_map_powered = compute_epe(u_ens, v_ens,
                                   test_data['u_truth'], test_data['v_truth'],
                                   test_data['valid_mask'],
                                   power=epe_power)
    ensemble_epe_powered = np.nanmean(epe_map_powered)
    
    # Also compute standard EPE for reporting
    epe_map_standard = compute_epe(u_ens, v_ens,
                                   test_data['u_truth'], test_data['v_truth'],
                                   test_data['valid_mask'],
                                   power=1.0)
    ensemble_epe_standard = np.nanmean(epe_map_standard)
    
    if epe_power != 1.0:
        print(f"   Ensemble EPE (p={epe_power}): {ensemble_epe_powered:.6f}")
        print(f"   Ensemble EPE (standard): {ensemble_epe_standard:.6f} px")
    else:
        print(f"   Ensemble EPE: {ensemble_epe_standard:.6f} px")
    print()
    
    # ========================================================================
    # Compute deformation metrics on ensemble flow
    # ========================================================================
    
    print(f"📐 Computing deformation metrics (magnitude={magnitude}px)...")
    deformation = compute_bidirectional_deformation(
        ensemble_results['u_ensemble_forward'],
        ensemble_results['v_ensemble_forward'],
        ensemble_results['u_ensemble_backward'],
        ensemble_results['v_ensemble_backward'],
        magnitude
    )
    print(f"   ✅ Computed {len(deformation)} deformation metrics")
    print()
    
    # ========================================================================
    # Save results to weights_<hash>/ directory
    # ========================================================================
    
    weight_hash = compute_weight_hash(ensemble_weights)
    weights_dir = cache_info['exp_dir'] / f"weights_{weight_hash}"
    weights_dir.mkdir(exist_ok=True)
    
    print(f"💾 Saving results to {weights_dir.name}/")
    
    # Save config with these weights
    config_copy = config.copy()
    if 'ensemble' not in config_copy:
        config_copy['ensemble'] = {}
    config_copy['ensemble']['weights'] = ensemble_weights
    
    with open(weights_dir / 'config.toml', 'w') as f:
        toml.dump(config_copy, f)
    
    # Save ensemble results
    np.savez(
        weights_dir / 'ensemble_results.npz',
        u_forward=ensemble_results['u_ensemble_forward'],
        v_forward=ensemble_results['v_ensemble_forward'],
        u_symmetric=ensemble_results['u_ensemble_symmetric'],
        v_symmetric=ensemble_results['v_ensemble_symmetric'],
        selection=ensemble_results['ensemble_selection'],
        ensemble_epe=ensemble_epe_standard  # Use standard for saving
    )
    
    # Save deformation metrics separately (keeps ensemble_results.npz clean)
    np.savez(
        weights_dir / 'deformation.npz',
        **deformation
    )
    
    # Generate figures
    if not args.no_viz:
        print(f"📊 Generating figures...")
        figures_dir = weights_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        # Load sweep_df for figure generation
        sweep_df = exp_cache.load_sweep_results()
        
        # Generate flow field comparison (new 3x3 visualization)
        # This also computes oracle and best flows for reuse
        print(f"   Generating flow field comparison...")
        flow_data = generate_flow_field_comparison(
            results_full,
            sweep_df,
            ensemble_results,
            test_data,
            oracle if should_compute else cached,
            figures_dir
        )
        
        # Generate additional sweep figures (histograms, correlations, etc.)
        print(f"   Generating additional figures...")
        generate_all_sweep_figures(
            results_full,
            sweep_df,
            test_data['frame1_original'],
            test_data['u_truth'],
            test_data['v_truth'],
            test_data['valid_mask'],
            flow_data['u_oracle_fwd'],
            flow_data['v_oracle_fwd'],
            flow_data['u_oracle_sym'],
            flow_data['v_oracle_sym'],
            flow_data['u_best_fwd'],
            flow_data['v_best_fwd'],
            flow_data['u_best_sym'],
            flow_data['v_best_sym'],
            ensemble_results['u_ensemble_forward'],
            ensemble_results['v_ensemble_forward'],
            ensemble_results['u_ensemble_symmetric'],
            ensemble_results['v_ensemble_symmetric'],
            ensemble_results['ensemble_selection'],
            figures_dir,
            sweep_params=swept_params,  # Pass swept parameters
            ensemble_source="fixed"  # Using fixed weights from config
        )
        print(f"   ✅ Figures saved")
    
    # ========================================================================
    # Compute best single config EPE for comparison
    # ========================================================================
    best_single_epe = sweep_df['mean_epe_forward'].min()
    best_single_name = sweep_df.loc[sweep_df['mean_epe_forward'].idxmin(), 'config_name']
    
    # Get oracle EPE
    if should_compute:
        oracle_epe = oracle['oracle_epe_forward']
    else:
        oracle_epe = cached['oracle_epe_forward']
    
    # Compute improvement metrics
    improvement_vs_best = 100 * (1 - ensemble_epe_standard / best_single_epe)
    
    # Oracle capture
    possible_improvement = best_single_epe - oracle_epe
    actual_improvement = best_single_epe - ensemble_epe_standard
    if possible_improvement > 1e-10:
        oracle_capture = 100 * actual_improvement / possible_improvement
    else:
        oracle_capture = 100.0
    
    # ========================================================================
    # Print error distribution table
    # ========================================================================
    if not args.no_viz:
        # Compute EPE maps for distribution analysis
        epe_map_best = compute_epe(
            flow_data['u_best_fwd'], flow_data['v_best_fwd'],
            test_data['u_truth'], test_data['v_truth'],
            test_data['valid_mask'],
            power=1.0
        )
        
        epe_map_ensemble = compute_epe(
            ensemble_results['u_ensemble_forward'], ensemble_results['v_ensemble_forward'],
            test_data['u_truth'], test_data['v_truth'],
            test_data['valid_mask'],
            power=1.0
        )
        
        epe_map_oracle = compute_epe(
            flow_data['u_oracle_fwd'], flow_data['v_oracle_fwd'],
            test_data['u_truth'], test_data['v_truth'],
            test_data['valid_mask'],
            power=1.0
        )
        
        print_error_distribution(
            epe_map_best,
            epe_map_ensemble,
            epe_map_oracle,
            test_data['valid_mask']
        )
    
    print()
    print("=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results: {weights_dir}")
    print()
    print("Performance Summary:")
    print(f"   Best single config: {best_single_epe:.6f} px  ({best_single_name})")
    print(f"   Ensemble:           {ensemble_epe_standard:.6f} px  (↓{improvement_vs_best:.1f}% vs best single)")
    print(f"   Oracle:             {oracle_epe:.6f} px  (theoretical limit)")
    print()
    print(f"   Oracle capture:     {oracle_capture:.1f}%  [(best_single - ensemble) / (best_single - oracle)]")
    print("=" * 80)


def run_optimization_mode(config: dict, args):
    """Run weight optimization, then evaluate with best weights."""
    print("🔍 OPTIMIZATION MODE")
    print("   Finding optimal weights")
    print()
    
    import optuna
    import pandas as pd
    
    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Extract config sections
    sweep_config = config['parameter_sweep']
    perturbation_config = config.get('perturbations', {'directions': 2, 'magnitude': 1})
    eval_config = config.get('evaluation', {})
    opt_config = config.get('optimization', {})
    
    # ========================================================================
    # Validate perturbation config (fail early)
    # ========================================================================
    directions = perturbation_config.get('directions', 2)
    magnitude = perturbation_config.get('magnitude', 1)
    
    if not isinstance(directions, int) or directions < 2:
        print(f"❌ ERROR: [perturbations] directions must be int >= 2, got {directions}")
        sys.exit(1)
    
    if not isinstance(magnitude, int) or magnitude < 1:
        print(f"❌ ERROR: [perturbations] magnitude must be int >= 1, got {magnitude}")
        sys.exit(1)
    
    # Optimization parameters
    n_trials = args.trials if args.trials else opt_config.get('n_trials', 50)
    study_name = opt_config.get('study_name', 'default')
    
    # Make study name deterministic for resumability
    # Use 'default' unless user specifies otherwise in config
    if study_name == 'auto':
        study_name = 'default'
    
    print(f"📋 Optimization settings:")
    print(f"   Study name: {study_name}")
    print(f"   Trials: {n_trials} (will add to existing if resuming)")
    print(f"   Constraint: weights sum to 1.0 (simplex)")
    print()
    
    # Expand parameter sweep
    configs = expand_parameter_sweep(sweep_config)
    n_configs = len(configs)
    print(f"📋 Configurations: {n_configs}")
    
    # Extract swept parameters (those with multiple values)
    swept_params = extract_swept_params(sweep_config)
    print(f"📋 Swept parameters: {swept_params if swept_params else 'None (single config)'}")
    print()
    
    # Generate perturbations
    deltas = generate_perturbation_deltas(directions, magnitude)
    print(f"📋 Perturbations: {len(deltas)} directions, magnitude={magnitude}px")
    print()
    
    # ========================================================================
    # Setup: Images, cache, hashing
    # ========================================================================
    
    test_data = setup_test_data(config, eval_config, configs)
    cache_info = setup_experiment_cache(config, test_data, args.no_cache)
    
    exp_cache = cache_info['exp_cache']
    should_compute = cache_info['should_compute']
    
    # ========================================================================
    # Compute or load sweep (expensive part - only once)
    # ========================================================================
    
    if should_compute:
        # Compute sweep
        results_native = compute_sweep(
            test_data['frame1'],
            test_data['frame2'],
            configs,
            deltas,
            n_workers=eval_config.get('n_workers', None)
        )
        
        # Get EPE power from config
        epe_power = config.get('evaluation', {}).get('epe_power', 2.0)
        
        # Upsample and compute oracle
        sweep_results = upsample_and_compute_oracle(
            results_native,
            test_data['H'], test_data['W'],
            test_data['u_truth'], test_data['v_truth'],
            test_data['valid_mask'],
            epe_power=epe_power
        )
        
        results_full = sweep_results['results_full']
        oracle = sweep_results['oracle']
        
        # Save to cache
        save_sweep_to_cache(
            exp_cache,
            results_full,
            test_data['u_truth'], test_data['v_truth'],
            test_data['valid_mask'],
            oracle['oracle_epe_forward'],
            oracle['oracle_epe_symmetric']
        )
    else:
        # Load from cache
        cached = load_sweep_from_cache(exp_cache)
        results_full = cached['results_full']
        
        if results_full is None:
            print("❌ ERROR: Full results not in cache, cannot optimize")
            print("   Run with --no-cache to recompute")
            sys.exit(1)
        
        oracle = {
            'oracle_epe_forward': cached['oracle_epe_forward'],
            'oracle_epe_symmetric': cached['oracle_epe_symmetric']
        }
    
    # ========================================================================
    # Run Optuna optimization
    # ========================================================================
    
    # Get EPE power from config
    epe_power = config.get('evaluation', {}).get('epe_power', 2.0)
    
    print("🔍 Running weight optimization...")
    print(f"   This will test {n_trials} weight combinations")
    print(f"   Optimizing with EPE power: {epe_power}")
    print()
    
    # Create optimization directory
    opt_dir = cache_info['exp_dir'] / f"optimization_{study_name}"
    opt_dir.mkdir(exist_ok=True)
    
    # Create Optuna study
    storage_path = f"sqlite:///{opt_dir / 'optuna_study.db'}"
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
        print(f"📂 Resuming existing study with {n_existing} trials")
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if complete_trials:
            print(f"   Current best EPE: {study.best_value:.6f}")
        print(f"   Will run {n_trials} additional trials")
        print()
    else:
        print(f"🆕 Starting new optimization study")
        print()
    
    # Define objective function with simplex constraint
    def objective(trial):
        # Sample 7 weights continuously, constrain 8th to make sum = 1.0
        remaining = 1.0
        weights = {}
        
        # First 7 weights: sample from [0, remaining] without step constraint
        weight_keys = ['traction_A', 'traction_B', 'consistency_A', 'consistency_B',
                      'photometric_A', 'photometric_B', 'displacements_N2S_A2B']
        
        for key in weight_keys:
            # Guard against floating-point errors making remaining slightly negative
            remaining = max(0.0, remaining)
            
            # If remaining is essentially zero, set to zero and skip
            if remaining < 1e-10:
                weights[key] = 0.0
            else:
                # Continuous sampling for smooth optimization
                w = trial.suggest_float(key, 0.0, remaining)
                weights[key] = w
                remaining -= w
        
        # 8th weight is determined by simplex constraint
        weights['displacements_N2S_B2A'] = max(0.0, remaining)
        
        # Generate ensemble
        ensemble_results = select_ensemble(results_full, weights, test_data['valid_mask'])
        
        # Compute EPE with configured power
        u_ens = ensemble_results['u_ensemble_forward']
        v_ens = ensemble_results['v_ensemble_forward']
        epe_map_powered = compute_epe(u_ens, v_ens,
                                       test_data['u_truth'], test_data['v_truth'],
                                       test_data['valid_mask'],
                                       power=epe_power)
        epe = np.nanmean(epe_map_powered)
        
        return epe
    
    # Run optimization with progress bar
    from tqdm import tqdm
    with tqdm(total=n_trials, desc="Optimizing", ncols=80, unit="trial") as pbar:
        def update_callback(study, trial):
            pbar.set_postfix({"Best": f"#{study.best_trial.number} {study.best_value:.6f}"})
            pbar.update(1)
        
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=False,
            callbacks=[update_callback]
        )
    
    print()
    print(f"✅ Optimization complete")
    print(f"   Total trials: {len(study.trials)} (ran {n_trials} this session)")
    print(f"   Best EPE: {study.best_value:.6f} px")
    print(f"   Oracle EPE: {oracle['oracle_epe_forward']:.6f} px")
    print(f"   Gap to oracle: {study.best_value - oracle['oracle_epe_forward']:.6f} px "
          f"({100 * (study.best_value / oracle['oracle_epe_forward'] - 1):+.1f}%)")
    print()
    
    # ========================================================================
    # Save optimization results
    # ========================================================================
    
    print(f"💾 Saving optimization results...")
    
    # Export trials to CSV
    trials_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            row = {'trial': trial.number, 'epe': trial.value}
            row.update(trial.params)
            trials_data.append(row)
    
    trials_df = pd.DataFrame(trials_data)
    trials_df.to_csv(opt_dir / 'trials.csv', index=False)
    
    # Save best weights
    best_weights = study.best_params
    import json
    with open(opt_dir / 'best_weights.json', 'w') as f:
        json.dump(best_weights, f, indent=2)
    
    print(f"   ✅ Saved to {opt_dir.name}/")
    
    # Generate optimization figures
    if not args.no_viz:
        opt_figures_dir = opt_dir / 'figures'
        generate_all_optimization_figures(study, oracle['oracle_epe_forward'], opt_figures_dir)
    
    print()
    
    # ========================================================================
    # Generate ensemble with best weights
    # ========================================================================
    
    print("🎯 Generating final ensemble with optimal weights...")
    print(f"   Optimal weights:")
    for key, val in best_weights.items():
        print(f"      {key:25s}: {val:.1f}")
    print()
    
    ensemble_results = select_ensemble(results_full, best_weights, test_data['valid_mask'])
    
    # Compute ensemble EPE with both powered and standard
    u_ens = ensemble_results['u_ensemble_forward']
    v_ens = ensemble_results['v_ensemble_forward']
    
    epe_map_powered = compute_epe(u_ens, v_ens,
                                   test_data['u_truth'], test_data['v_truth'],
                                   test_data['valid_mask'],
                                   power=epe_power)
    ensemble_epe_powered = np.nanmean(epe_map_powered)
    
    epe_map_standard = compute_epe(u_ens, v_ens,
                                   test_data['u_truth'], test_data['v_truth'],
                                   test_data['valid_mask'],
                                   power=1.0)
    ensemble_epe_standard = np.nanmean(epe_map_standard)
    
    # ========================================================================
    # Compute deformation metrics on ensemble flow
    # ========================================================================
    
    print(f"📐 Computing deformation metrics (magnitude={magnitude}px)...")
    deformation = compute_bidirectional_deformation(
        ensemble_results['u_ensemble_forward'],
        ensemble_results['v_ensemble_forward'],
        ensemble_results['u_ensemble_backward'],
        ensemble_results['v_ensemble_backward'],
        magnitude
    )
    print(f"   ✅ Computed {len(deformation)} deformation metrics")
    print()
    
    # ========================================================================
    # Save to weights_optim_NNN/ directory (only if weights changed)
    # ========================================================================
    
    # Check if weights changed from last optimization
    prev_weights = get_latest_optim_weights(cache_info['exp_dir'])
    
    if prev_weights is not None and weights_are_equal(best_weights, prev_weights):
        print(f"✓ Optimal weights unchanged from previous optimization")
        print(f"   Skipping new weights_optim directory")
        print()
    else:
        # Weights are new/different - create new directory
        optim_num = get_next_optim_number(cache_info['exp_dir'])
        weights_dir = cache_info['exp_dir'] / f"weights_optim_{optim_num:03d}"
        weights_dir.mkdir(exist_ok=True)
        
        if prev_weights is None:
            print(f"💾 Saving optimal ensemble to {weights_dir.name}/")
        else:
            print(f"💾 Weights improved! Saving to {weights_dir.name}/")
    
        # Save config with optimal weights
        config_copy = config.copy()
        if 'ensemble' not in config_copy:
            config_copy['ensemble'] = {}
        config_copy['ensemble']['weights'] = best_weights
        
        with open(weights_dir / 'config.toml', 'w') as f:
            toml.dump(config_copy, f)
        
        # Save ensemble results
        np.savez(
            weights_dir / 'ensemble_results.npz',
            u_forward=ensemble_results['u_ensemble_forward'],
            v_forward=ensemble_results['v_ensemble_forward'],
            u_symmetric=ensemble_results['u_ensemble_symmetric'],
            v_symmetric=ensemble_results['v_ensemble_symmetric'],
            selection=ensemble_results['ensemble_selection'],
            ensemble_epe=ensemble_epe_standard  # Use standard for saving
        )
        
        # Save deformation metrics separately
        np.savez(
            weights_dir / 'deformation.npz',
            **deformation
        )
        
        # Generate figures
        if not args.no_viz:
            print(f"📊 Generating figures...")
            figures_dir = weights_dir / 'figures'
            figures_dir.mkdir(exist_ok=True)
            
            # Load sweep_df for figure generation
            sweep_df = exp_cache.load_sweep_results()
            
            # Generate flow field comparison (new 3x3 visualization)
            # This also computes oracle and best flows for reuse
            print(f"   Generating flow field comparison...")
            flow_data = generate_flow_field_comparison(
                results_full,
                sweep_df,
                ensemble_results,
                test_data,
                oracle,
                figures_dir
            )
            
            # Generate additional sweep figures (histograms, correlations, etc.)
            print(f"   Generating additional figures...")
            generate_all_sweep_figures(
                results_full,
                sweep_df,
                test_data['frame1_original'],
                test_data['u_truth'],
                test_data['v_truth'],
                test_data['valid_mask'],
                flow_data['u_oracle_fwd'],
                flow_data['v_oracle_fwd'],
                flow_data['u_oracle_sym'],
                flow_data['v_oracle_sym'],
                flow_data['u_best_fwd'],
                flow_data['v_best_fwd'],
                flow_data['u_best_sym'],
                flow_data['v_best_sym'],
                ensemble_results['u_ensemble_forward'],
                ensemble_results['v_ensemble_forward'],
                ensemble_results['u_ensemble_symmetric'],
                ensemble_results['v_ensemble_symmetric'],
                ensemble_results['ensemble_selection'],
                figures_dir,
                sweep_params=swept_params,  # Pass swept parameters
                ensemble_source="optimized"  # Using optimized weights
            )
            print(f"   ✅ Figures saved")
        
        print()
    
    # ========================================================================
    # Compute best single config EPE for comparison
    # ========================================================================
    sweep_df = exp_cache.load_sweep_results()
    best_single_epe = sweep_df['mean_epe_forward'].min()
    best_single_name = sweep_df.loc[sweep_df['mean_epe_forward'].idxmin(), 'config_name']
    
    # Compute improvement metrics
    oracle_epe = oracle['oracle_epe_forward']
    improvement_vs_best = 100 * (1 - ensemble_epe_standard / best_single_epe)
    
    # Oracle capture: how much of the possible improvement did we get?
    # (best_single - ensemble) / (best_single - oracle)
    possible_improvement = best_single_epe - oracle_epe
    actual_improvement = best_single_epe - ensemble_epe_standard
    if possible_improvement > 1e-10:
        oracle_capture = 100 * actual_improvement / possible_improvement
    else:
        oracle_capture = 100.0  # Already at oracle
    
    # ========================================================================
    # Print error distribution table
    # ========================================================================
    if not args.no_viz:
        # Compute EPE maps for distribution analysis
        epe_map_best = compute_epe(
            flow_data['u_best_fwd'], flow_data['v_best_fwd'],
            test_data['u_truth'], test_data['v_truth'],
            test_data['valid_mask'],
            power=1.0
        )
        
        epe_map_ensemble = compute_epe(
            ensemble_results['u_ensemble_forward'], ensemble_results['v_ensemble_forward'],
            test_data['u_truth'], test_data['v_truth'],
            test_data['valid_mask'],
            power=1.0
        )
        
        epe_map_oracle = compute_epe(
            flow_data['u_oracle_fwd'], flow_data['v_oracle_fwd'],
            test_data['u_truth'], test_data['v_truth'],
            test_data['valid_mask'],
            power=1.0
        )
        
        print_error_distribution(
            epe_map_best,
            epe_map_ensemble,
            epe_map_oracle,
            test_data['valid_mask']
        )
    
    print("=" * 80)
    print("✅ OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Optimization: {opt_dir}")
    
    # Check if we created a new weights directory
    if prev_weights is not None and weights_are_equal(best_weights, prev_weights):
        # Weights unchanged - reference the existing directory
        import re
        pattern = re.compile(r'weights_optim_(\d+)')
        max_num = 0
        for item in cache_info['exp_dir'].iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    num = int(match.group(1))
                    if num > max_num:
                        max_num = num
        if max_num > 0:
            print(f"Best ensemble: {cache_info['exp_dir'] / f'weights_optim_{max_num:03d}'} (unchanged)")
    else:
        print(f"Best ensemble: {weights_dir}")
    
    print()
    print("Performance Summary:")
    print(f"   Best single config: {best_single_epe:.6f} px  ({best_single_name})")
    print(f"   Ensemble:           {ensemble_epe_standard:.6f} px  (↓{improvement_vs_best:.1f}% vs best single)")
    print(f"   Oracle:             {oracle_epe:.6f} px  (theoretical limit)")
    print()
    print(f"   Oracle capture:     {oracle_capture:.1f}%  [(best_single - ensemble) / (best_single - oracle)]")
    print("=" * 80)


if __name__ == "__main__":
    main()
