# File: src/visualization/ensemble_figures.py
"""
Ensemble flow figure generation.

Coordinates all visualization for optical flow tracking:
- Flow field comparisons
- Sweep analysis figures
- Optimization figures
- Error distribution tables

All functions require ground truth and fail gracefully if unavailable.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.ground_truth import compute_epe
from src.evaluation.error_summary import print_error_distribution
from src.visualization.flow_field_comparison import visualize_flow_field_comparison
from src.visualization.sweep_figures import generate_all_sweep_figures
from src.visualization.optimization_figures import generate_all_optimization_figures


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


def generate_ensemble_figures(
    results_full: List[dict],
    sweep_df,
    ensemble_results: dict,
    test_data: dict,
    oracle: dict,
    figures_dir: Path,
    swept_params: List[str],
    ensemble_source: str,
    skip_if_no_gt: bool = True
) -> Optional[dict]:
    """
    Generate all ensemble flow figures.
    
    Coordinates all visualization including:
    - Flow field comparison (3x3 grid)
    - Sweep analysis figures (histograms, correlations)
    - Error distribution tables
    
    Args:
        results_full: List of config result dicts
        sweep_df: Sweep results DataFrame
        ensemble_results: Ensemble selection results
        test_data: Test data dict with frames and GT
        oracle: Oracle computation results
        figures_dir: Output directory for figures
        swept_params: List of swept parameter names
        ensemble_source: "fixed" or "optimized"
        skip_if_no_gt: If True, skip gracefully when GT unavailable
    
    Returns:
        dict with flow_data (oracle flows, best flows) or None if skipped
    """
    
    # ========================================================================
    # Check for ground truth
    # ========================================================================
    
    has_gt = (test_data.get('u_truth') is not None and 
              test_data.get('v_truth') is not None)
    
    if not has_gt:
        if skip_if_no_gt:
            print(f"   ⚠️  Skipping figures (ground truth not available)")
            return None
        else:
            print(f"❌ ERROR: Figures require ground truth but it's not available")
            sys.exit(1)
    
    # ========================================================================
    # Generate flow field comparison
    # ========================================================================
    
    print(f"   Generating flow field comparison...")
    
    flow_data = generate_flow_field_comparison(
        results_full,
        sweep_df,
        ensemble_results,
        test_data,
        oracle,
        figures_dir
    )
    
    # ========================================================================
    # Generate sweep figures
    # ========================================================================
    
    print(f"   Generating sweep analysis figures...")
    
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
        sweep_params=swept_params,
        ensemble_source=ensemble_source
    )
    
    return flow_data


def generate_optimization_figures(
    study,
    oracle_epe: float,
    figures_dir: Path
):
    """
    Generate optimization-specific figures.
    
    Args:
        study: Optuna study object
        oracle_epe: Oracle EPE for reference
        figures_dir: Output directory
    """
    
    print(f"   Generating optimization figures...")
    
    generate_all_optimization_figures(
        study,
        oracle_epe,
        figures_dir
    )


def print_ensemble_error_distribution(
    flow_data: dict,
    ensemble_results: dict,
    test_data: dict,
    valid_mask: np.ndarray,
    epe_power: float = 1.0,
    skip_if_no_gt: bool = True
):
    """
    Print error distribution table comparing best, ensemble, and oracle.
    
    Args:
        flow_data: Flow data dict with oracle and best flows
        ensemble_results: Ensemble selection results
        test_data: Test data dict with GT
        valid_mask: Valid pixel mask
        epe_power: EPE power for computation
        skip_if_no_gt: If True, skip gracefully when GT unavailable
    """
    
    # ========================================================================
    # Check for ground truth
    # ========================================================================
    
    has_gt = (test_data.get('u_truth') is not None and 
              test_data.get('v_truth') is not None)
    
    if not has_gt:
        if skip_if_no_gt:
            print(f"   ⚠️  Skipping error distribution (ground truth not available)")
            return
        else:
            print(f"❌ ERROR: Error distribution requires ground truth")
            sys.exit(1)
    
    # ========================================================================
    # Compute EPE maps
    # ========================================================================
    
    epe_map_best = compute_epe(
        flow_data['u_best_fwd'],
        flow_data['v_best_fwd'],
        test_data['u_truth'],
        test_data['v_truth'],
        valid_mask,
        power=epe_power
    )
    
    epe_map_ensemble = compute_epe(
        ensemble_results['u_ensemble_forward'],
        ensemble_results['v_ensemble_forward'],
        test_data['u_truth'],
        test_data['v_truth'],
        valid_mask,
        power=epe_power
    )
    
    epe_map_oracle = compute_epe(
        flow_data['u_oracle_fwd'],
        flow_data['v_oracle_fwd'],
        test_data['u_truth'],
        test_data['v_truth'],
        valid_mask,
        power=epe_power
    )
    
    # ========================================================================
    # Print table
    # ========================================================================
    
    print_error_distribution(
        epe_map_best,
        epe_map_ensemble,
        epe_map_oracle,
        valid_mask
    )


def compute_performance_summary(
    ensemble_epe: float,
    sweep_df,
    oracle: dict,
    verbose: bool = True
) -> dict:
    """
    Compute performance summary statistics.
    
    Args:
        ensemble_epe: Ensemble EPE
        sweep_df: Sweep results DataFrame
        oracle: Oracle computation results
        verbose: Print summary if True
    
    Returns:
        dict with:
            - best_single_epe
            - best_single_name
            - ensemble_epe
            - oracle_epe
            - improvement_vs_best (%)
            - oracle_capture (%)
    """
    
    # Best single config
    best_single_epe = sweep_df['mean_epe_forward'].min()
    best_single_name = sweep_df.loc[sweep_df['mean_epe_forward'].idxmin(), 'config_name']
    
    # Oracle EPE
    oracle_epe = oracle['oracle_epe_forward']
    
    # Improvement vs best single
    improvement_vs_best = 100 * (1 - ensemble_epe / best_single_epe)
    
    # Oracle capture: how much of possible improvement did we get?
    possible_improvement = best_single_epe - oracle_epe
    actual_improvement = best_single_epe - ensemble_epe
    
    if possible_improvement > 1e-10:
        oracle_capture = 100 * actual_improvement / possible_improvement
    else:
        oracle_capture = 100.0  # Already at oracle
    
    summary = {
        'best_single_epe': best_single_epe,
        'best_single_name': best_single_name,
        'ensemble_epe': ensemble_epe,
        'oracle_epe': oracle_epe,
        'improvement_vs_best': improvement_vs_best,
        'oracle_capture': oracle_capture
    }
    
    if verbose:
        print()
        print("Performance Summary:")
        print(f"   Best single config: {best_single_epe:.6f} px  ({best_single_name})")
        print(f"   Ensemble:           {ensemble_epe:.6f} px  (↓{improvement_vs_best:.1f}% vs best single)")
        print(f"   Oracle:             {oracle_epe:.6f} px  (theoretical limit)")
        print()
        print(f"   Oracle capture:     {oracle_capture:.1f}%  [(best_single - ensemble) / (best_single - oracle)]")
    
    return summary


if __name__ == "__main__":
    print("✅ Ensemble figures module loaded")
    print("\n📊 This module coordinates all figure generation for ensemble flow tracking.")
    print("   - Flow field comparisons")
    print("   - Sweep analysis figures") 
    print("   - Optimization figures")
    print("   - Error distribution tables")
    print("\n⚠️  All functions require ground truth and skip gracefully if unavailable.")
