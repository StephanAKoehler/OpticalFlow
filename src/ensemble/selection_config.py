# File: src/ensemble/selection_config.py
"""
Selection configuration parsing and orchestration.

Handles parsing [selection.*] sections from TOML and running selection
for all configured methods.
"""

import sys
import json
import toml
import numpy as np
from pathlib import Path
from typing import Optional

from src.ensemble.selection import (
    select_ensemble,
    validate_selection_config,
    compute_selection_hash
)
from src.evaluation.ground_truth import compute_epe
from src.optical_flow.flow_deformation import compute_bidirectional_deformation


def parse_selection_configs(config: dict) -> dict:
    """
    Parse all [selection.*] sections from config.
    
    Args:
        config: Full TOML config dict
        
    Returns:
        Dict mapping method names to selection configs:
        {
            'mad_sum': {'normalize': 'mad', 'aggregation': 'sum', ...},
            'raw_max': {'normalize': 'none', 'aggregation': 'max', ...},
            ...
        }
        
    Raises:
        SystemExit if no selection methods defined or validation fails
    """
    selection_section = config.get('selection', {})
    
    if not selection_section:
        print("❌ ERROR: No [selection.*] sections in config")
        print("   Define at least one selection method, e.g.:")
        print("   [selection.mad_sum]")
        print("   normalize = \"mad\"")
        print("   aggregation = \"sum\"")
        print("   power = 2")
        print("   traction = 0.0")
        print("   perturbation_rms = 1.0")
        print("   consistency = 1.0")
        print("   photometric = 1.0")
        sys.exit(1)
    
    selection_configs = {}
    
    for name, method_config in selection_section.items():
        # Validate
        validate_selection_config(method_config, name)
        selection_configs[name] = method_config.copy()
    
    return selection_configs


def run_selection_method(
    method_name: str,
    selection_config: dict,
    results_full: list[dict],
    valid_mask: np.ndarray,
    exp_dir: Path,
    pair_data: dict,
    magnitude: int,
    epe_power: float,
    no_viz: bool = False
) -> dict:
    """
    Run a single selection method and save results.
    
    Args:
        method_name: Name of selection method (e.g., 'mad_sum')
        selection_config: Selection configuration dict
        results_full: List of config result dicts from sweep
        valid_mask: Boolean mask for valid pixels
        exp_dir: Experiment directory for output
        pair_data: Dict with frame1, frame2, u_truth, v_truth, etc.
        magnitude: Perturbation magnitude (for deformation metrics)
        epe_power: EPE power from [evaluation] section
        no_viz: Skip figure generation if True
        
    Returns:
        Dict with selection results including EPE if GT available
    """
    
    print(f"\n{'='*60}")
    print(f"SELECTION: {method_name}")
    print(f"{'='*60}")
    print(f"   normalize: {selection_config['normalize']}")
    print(f"   aggregation: {selection_config['aggregation']}")
    print(f"   power: {selection_config['power']}")
    print(f"   weights: traction={selection_config['traction']:.2f}, "
          f"perturbation_rms={selection_config['perturbation_rms']:.2f}, "
          f"consistency={selection_config['consistency']:.2f}, "
          f"photometric={selection_config['photometric']:.2f}")
    print()
    
    # Compute selection hash
    selection_hash = compute_selection_hash(selection_config)
    
    # Create output directory
    output_dir = exp_dir / f"selection_{method_name}_{selection_hash}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already computed
    results_path = output_dir / 'ensemble_results.npz'
    if results_path.exists():
        print(f"   📦 Loading from cache: {output_dir.name}")
        cached = np.load(results_path)
        ensemble_results = {
            'u_ensemble_forward': cached['u_forward'],
            'v_ensemble_forward': cached['v_forward'],
            'u_ensemble_symmetric': cached['u_symmetric'],
            'v_ensemble_symmetric': cached['v_symmetric'],
            'u_ensemble_backward': cached['u_backward'],
            'v_ensemble_backward': cached['v_backward'],
            'ensemble_selection': cached['selection'],
        }
    else:
        # Run selection
        print(f"   🔍 Computing ensemble selection...")
        ensemble_results = select_ensemble(results_full, selection_config, valid_mask)
        print(f"   ✅ Selection complete")
        
        # Save ensemble results
        np.savez(
            results_path,
            u_forward=ensemble_results['u_ensemble_forward'],
            v_forward=ensemble_results['v_ensemble_forward'],
            u_symmetric=ensemble_results['u_ensemble_symmetric'],
            v_symmetric=ensemble_results['v_ensemble_symmetric'],
            u_backward=ensemble_results['u_ensemble_backward'],
            v_backward=ensemble_results['v_ensemble_backward'],
            selection=ensemble_results['ensemble_selection']
        )
    
    # Save selection config
    config_path = output_dir / 'config.toml'
    with open(config_path, 'w') as f:
        toml.dump({'selection': {method_name: selection_config}}, f)
    
    # Compute deformation metrics
    print(f"   📐 Computing deformation metrics...")
    deformation = compute_bidirectional_deformation(
        ensemble_results['u_ensemble_forward'],
        ensemble_results['v_ensemble_forward'],
        ensemble_results['u_ensemble_backward'],
        ensemble_results['v_ensemble_backward'],
        magnitude
    )
    np.savez(output_dir / 'deformation.npz', **deformation)
    print(f"   ✅ Computed {len(deformation)} deformation metrics")
    
    # Compute EPE if GT available
    u_truth = pair_data.get('u_truth')
    v_truth = pair_data.get('v_truth')
    has_gt = u_truth is not None and v_truth is not None
    
    ensemble_epe = None
    ensemble_epe_std = None
    ensemble_epe_powered = None
    ensemble_epe_powered_std = None
    
    if has_gt:
        # Standard EPE (power=1) for comparison
        epe_map = compute_epe(
            ensemble_results['u_ensemble_forward'],
            ensemble_results['v_ensemble_forward'],
            u_truth,
            v_truth,
            valid_mask,
            power=1.0
        )
        ensemble_epe = float(np.nanmean(epe_map))
        ensemble_epe_std = float(np.nanstd(epe_map[valid_mask]))
        
        # Also compute powered EPE
        epe_map_powered = compute_epe(
            ensemble_results['u_ensemble_forward'],
            ensemble_results['v_ensemble_forward'],
            u_truth,
            v_truth,
            valid_mask,
            power=epe_power
        )
        ensemble_epe_powered = float(np.nanmean(epe_map_powered))
        ensemble_epe_powered_std = float(np.nanstd(epe_map_powered[valid_mask]))
        
        print(f"\n   📊 Ensemble EPE: {ensemble_epe:.6f} ± {ensemble_epe_std:.6f} px (standard)")
        print(f"   📊 Ensemble EPE^{epe_power}: {ensemble_epe_powered:.6f} ± {ensemble_epe_powered_std:.6f}")
    
    # Selection distribution
    n_configs = len(results_full)
    selection_counts = np.bincount(
        ensemble_results['ensemble_selection'][valid_mask].ravel(),
        minlength=n_configs
    )
    n_valid = valid_mask.sum()
    n_configs_used = int(np.sum(selection_counts > 0))
    
    print(f"\n   📊 Selection distribution ({n_configs_used} configs used):")
    # Show top 5 configs
    top_indices = np.argsort(selection_counts)[::-1][:5]
    for idx in top_indices:
        count = selection_counts[idx]
        pct = 100 * count / n_valid
        if count > 0:
            # Get config name if available
            if 'metadata' in results_full[idx]:
                config_name = results_full[idx]['metadata'].get('config_name', f'config_{idx}')
            else:
                config_name = results_full[idx].get('config_name', f'config_{idx}')
            print(f"      {config_name}: {count:5d} px ({pct:5.1f}%)")
    
    # Generate figures if requested
    if not no_viz:
        figures_dir = output_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        # TODO: Call visualization functions
        print(f"   📊 Figures directory: {figures_dir}")
    
    print(f"\n   ✅ Results saved to: {output_dir.name}")
    
    return {
        'method_name': method_name,
        'selection_config': selection_config,
        'selection_hash': selection_hash,
        'output_dir': output_dir,
        'ensemble_results': ensemble_results,
        'ensemble_epe': ensemble_epe,
        'ensemble_epe_std': ensemble_epe_std,
        'ensemble_epe_powered': ensemble_epe_powered,
        'ensemble_epe_powered_std': ensemble_epe_powered_std,
        'n_configs_used': n_configs_used,
        'deformation': deformation,
    }


def run_all_selections(
    config: dict,
    results_full: list[dict],
    valid_mask: np.ndarray,
    exp_dir: Path,
    pair_data: dict,
    no_viz: bool = False
) -> dict:
    """
    Run all selection methods defined in config.
    
    Args:
        config: Full TOML config
        results_full: List of config result dicts from sweep
        valid_mask: Boolean mask for valid pixels
        exp_dir: Experiment directory for output
        pair_data: Dict with frame1, frame2, u_truth, v_truth, etc.
        no_viz: Skip figure generation if True
        
    Returns:
        Dict mapping method names to their results
    """
    
    # Parse selection configs
    selection_configs = parse_selection_configs(config)
    
    # Get magnitude and epe_power from config
    perturbation_config = config.get('perturbations', {'magnitude': 1})
    magnitude = perturbation_config.get('magnitude', 1)
    
    eval_config = config.get('evaluation', {})
    epe_power = eval_config.get('epe_power', None)
    if epe_power is None:
        print("❌ ERROR: [evaluation] epe_power must be specified")
        sys.exit(1)
    
    print(f"\n📋 Running {len(selection_configs)} selection method(s)")
    
    results = {}
    
    for method_name, selection_config in selection_configs.items():
        result = run_selection_method(
            method_name=method_name,
            selection_config=selection_config,
            results_full=results_full,
            valid_mask=valid_mask,
            exp_dir=exp_dir,
            pair_data=pair_data,
            magnitude=magnitude,
            epe_power=epe_power,
            no_viz=no_viz
        )
        results[method_name] = result
    
    # Summary
    print(f"\n{'='*60}")
    print("SELECTION SUMMARY")
    print(f"{'='*60}")
    
    has_gt = pair_data.get('u_truth') is not None
    
    if has_gt:
        print(f"\n{'Method':<20} {'EPE (px)':>12}")
        print("-" * 35)
        for method_name, result in sorted(results.items(), key=lambda x: x[1]['ensemble_epe'] or float('inf')):
            epe = result['ensemble_epe']
            if epe is not None:
                print(f"{method_name:<20} {epe:>12.6f}")
            else:
                print(f"{method_name:<20} {'N/A':>12}")
    else:
        print("No ground truth available - EPE not computed")
        for method_name in results:
            print(f"   ✅ {method_name}")
    
    return results


def run_optimization_for_method(
    method_name: str,
    selection_config: dict,
    results_full: list[dict],
    u_truth: np.ndarray,
    v_truth: np.ndarray,
    valid_mask: np.ndarray,
    exp_dir: Path,
    n_trials: int,
    epe_power: float,
    show_progress: bool = True
) -> dict:
    """
    Run weight optimization for a single selection method.
    
    Args:
        method_name: Name of selection method
        selection_config: Selection config (normalize/aggregation/power fixed)
        results_full: List of config result dicts
        u_truth, v_truth: Ground truth flow
        valid_mask: Valid pixel mask
        exp_dir: Experiment directory
        n_trials: Number of Optuna trials
        epe_power: EPE power for loss function
        show_progress: Whether to show progress bar
        
    Returns:
        Dict with optimization results
    """
    
    from src.optimization.weight_optimizer import optimize_weights
    
    # Create optimization directory
    opt_dir = exp_dir / f"optimization_{method_name}"
    
    # Build selection template (fixed params, weights will be optimized)
    selection_template = {
        'normalize': selection_config['normalize'],
        'aggregation': selection_config['aggregation'],
        'power': selection_config['power'],
    }
    
    # Run optimization
    opt_results = optimize_weights(
        results_full=results_full,
        u_truth=u_truth,
        v_truth=v_truth,
        valid_mask=valid_mask,
        selection_template=selection_template,
        output_dir=opt_dir,
        n_trials=n_trials,
        epe_power=epe_power,
        show_progress=show_progress,
        method_name=method_name
    )
    
    return {
        'method_name': method_name,
        'output_dir': opt_dir,
        **opt_results
    }


if __name__ == "__main__":
    print("✅ Selection config module loaded")
    
    # Test parsing
    test_config = {
        'selection': {
            'mad_sum': {
                'normalize': 'mad',
                'aggregation': 'sum',
                'power': 2,
                'traction': 0.0,
                'perturbation_rms': 1.0,
                'consistency': 1.0,
                'photometric': 1.0,
            },
            'raw_max': {
                'normalize': 'none',
                'aggregation': 'max',
                'power': 2,
                'traction': 0.0,
                'perturbation_rms': 1.0,
                'consistency': 1.0,
                'photometric': 1.0,
            }
        },
        'evaluation': {
            'epe_power': 2
        },
        'perturbations': {
            'magnitude': 1
        }
    }
    
    selection_configs = parse_selection_configs(test_config)
    print(f"\n✅ Parsed {len(selection_configs)} selection methods:")
    for name, cfg in selection_configs.items():
        print(f"   {name}: {cfg['normalize']}_{cfg['aggregation']}")
