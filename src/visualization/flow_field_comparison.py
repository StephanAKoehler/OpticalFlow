#!/usr/bin/env python3
# File: src/visualization/flow_field_comparison.py
"""
Comprehensive 3×3 flow field comparison visualization.

Can be used as module or CLI script.

Layout:
┌─────────────────┬─────────────────┬─────────────────┐
│    Oracle       │   Ensemble      │  Best Single    │
├─────────────────┼─────────────────┼─────────────────┤
│ Flow Magnitude  │ Flow Magnitude  │ Flow Magnitude  │  Row 1
├─────────────────┼─────────────────┼─────────────────┤
│ EPE Error Map   │ EPE Error Map   │ EPE Error Map   │  Row 2
├─────────────────┼─────────────────┼─────────────────┤
│ Oracle Configs  │ Ensemble Config │ Agreement Map   │  Row 3
└─────────────────┴─────────────────┴─────────────────┘

Features:
- Clean layout with no text overlays
- Statistics in subplot titles
- Config index 0 = best single (marked with *)
- Agreement map shows where ensemble matches oracle
- Colorbars positioned outside grid on right

Usage as CLI:
    python -m src.visualization.flow_field_comparison results/e9e4e6/dis_f39dcf_e9e4e6
    python -m src.visualization.flow_field_comparison results/e9e4e6/dis_f39dcf_e9e4e6 --weights-dir weights_optim_002
"""

import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_flow_field_comparison(
    u_best, v_best,
    u_ensemble, v_ensemble,
    u_oracle, v_oracle,
    u_true, v_true,
    valid_mask,
    best_config_name: str,
    frame_A: np.ndarray = None,
    ensemble_selection: np.ndarray = None,
    oracle_selection: np.ndarray = None,
    n_configs: int = None,
    best_config_idx: int = None,
    save_path: str = 'flow_field_comparison.png',
    dpi: int = 150
):
    """
    Generate 2×3 flow field comparison figure focused on selection errors.
    
    Layout:
    Row 1: Config mismatch visualization (Frame A | Ensemble vs Oracle | Best vs Oracle)
    Row 2: EPE error maps filtered to EPE > threshold
    
    Args:
        u_best, v_best: Best single config flow
        u_ensemble, v_ensemble: Ensemble flow
        u_oracle, v_oracle: Oracle flow
        u_true, v_true: Ground truth flow
        valid_mask: Boolean mask for valid pixels
        best_config_name: Name of best config for title
        frame_A: Reference frame for disagreement visualization (optional)
        ensemble_selection: Config indices selected by ensemble (H, W) (optional)
        oracle_selection: Config indices selected by oracle (H, W) (optional)
        n_configs: Number of configs for colormap (optional)
        best_config_idx: Index of best config in results_full (optional)
        save_path: Output path
        dpi: Figure DPI
    """
    
    # Compute EPE maps
    epe_best = np.sqrt((u_best - u_true)**2 + (v_best - v_true)**2)
    epe_ensemble = np.sqrt((u_ensemble - u_true)**2 + (v_ensemble - v_true)**2)
    epe_oracle = np.sqrt((u_oracle - u_true)**2 + (v_oracle - v_true)**2)
    
    # Apply valid mask
    epe_best[~valid_mask] = np.nan
    epe_ensemble[~valid_mask] = np.nan
    epe_oracle[~valid_mask] = np.nan
    
    # EPE threshold
    epe_cutoff = 0.1
    
    # Count hard pixels for each method
    hard_oracle = (epe_oracle > epe_cutoff) & valid_mask
    hard_ensemble = (epe_ensemble > epe_cutoff) & valid_mask
    hard_best = (epe_best > epe_cutoff) & valid_mask
    
    n_hard_oracle = np.sum(hard_oracle)
    n_hard_ensemble = np.sum(hard_ensemble)
    n_hard_best = np.sum(hard_best)
    
    # Compute EPE statistics only for hard pixels (EPE > threshold)
    epe_oracle_hard = epe_oracle[hard_oracle]
    epe_ensemble_hard = epe_ensemble[hard_ensemble]
    epe_best_hard = epe_best[hard_best]
    
    epe_oracle_mean = np.mean(epe_oracle_hard) if len(epe_oracle_hard) > 0 else 0
    epe_oracle_std = np.std(epe_oracle_hard) if len(epe_oracle_hard) > 0 else 0
    epe_ensemble_mean = np.mean(epe_ensemble_hard) if len(epe_ensemble_hard) > 0 else 0
    epe_ensemble_std = np.std(epe_ensemble_hard) if len(epe_ensemble_hard) > 0 else 0
    epe_best_mean = np.mean(epe_best_hard) if len(epe_best_hard) > 0 else 0
    epe_best_std = np.std(epe_best_hard) if len(epe_best_hard) > 0 else 0
    
    # Create figure with 2×3 layout
    fig = plt.figure(figsize=(16, 11))
    
    # Create gridspec with space for colorbars on right
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.05, hspace=0.20,
                          left=0.05, right=0.92, top=0.88, bottom=0.08)
    
    # Create axes
    axes = []
    for row in range(2):
        axes_row = []
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            axes_row.append(ax)
        axes.append(axes_row)
    axes = np.array(axes)
    
    # ========================================================================
    # Row 1: Config Mismatch Visualization
    # ========================================================================
    
    if ensemble_selection is not None and oracle_selection is not None and frame_A is not None:
        # Prepare frame for visualization
        if frame_A.dtype == np.float32 or frame_A.dtype == np.float64:
            frame_normalized = (frame_A * 255).astype(np.uint8)
        else:
            frame_normalized = frame_A
        
        # Ensure 2D grayscale
        if len(frame_normalized.shape) == 3:
            frame_normalized = frame_normalized[:, :, 0]
        
        H, W = frame_normalized.shape
        
        # Config mismatch masks (within oracle hard pixels)
        ensemble_agreement = (ensemble_selection == oracle_selection) & valid_mask
        ensemble_mismatch = (ensemble_selection != oracle_selection) & hard_oracle
        
        best_matches_oracle = (oracle_selection == best_config_idx) & valid_mask
        best_mismatch = ~best_matches_oracle & hard_oracle
        
        n_ensemble_mismatch = np.sum(ensemble_mismatch)
        n_best_mismatch = np.sum(best_mismatch)
        
        # ====================================================================
        # Row 1, Col 0: Frame A reference (compressed to 0-127 range)
        # ====================================================================
        output_frameA = np.zeros((H, W, 3), dtype=np.uint8)
        compressed_frameA = (frame_normalized / 2).astype(np.uint8)
        output_frameA[:, :, 0] = compressed_frameA
        output_frameA[:, :, 1] = compressed_frameA
        output_frameA[:, :, 2] = compressed_frameA
        
        axes[0, 0].imshow(output_frameA)
        axes[0, 0].set_title('Frame A', fontsize=11, fontweight='bold', pad=8)
        axes[0, 0].axis('off')
        
        # ====================================================================
        # Row 1, Col 1: Ensemble vs Oracle config mismatch
        # ====================================================================
        output_ensemble = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Good regions: white
        good_mask_ensemble = ~ensemble_mismatch
        output_ensemble[good_mask_ensemble, 0] = 255
        output_ensemble[good_mask_ensemble, 1] = 255
        output_ensemble[good_mask_ensemble, 2] = 255
        
        # Mismatch regions: frame_A compressed to 0-127
        if np.any(ensemble_mismatch):
            compressed = (frame_normalized[ensemble_mismatch] / 2).astype(np.uint8)
            output_ensemble[ensemble_mismatch, 0] = compressed
            output_ensemble[ensemble_mismatch, 1] = compressed
            output_ensemble[ensemble_mismatch, 2] = compressed
        
        axes[0, 1].imshow(output_ensemble)
        axes[0, 1].set_title(f'Config Mismatch: Ensemble vs Oracle\n({n_ensemble_mismatch}/{n_hard_oracle})',
                            fontsize=11, fontweight='bold', pad=8)
        axes[0, 1].axis('off')
        
        # ====================================================================
        # Row 1, Col 2: Best Single vs Oracle config mismatch
        # ====================================================================
        output_best = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Good regions: white
        good_mask_best = ~best_mismatch
        output_best[good_mask_best, 0] = 255
        output_best[good_mask_best, 1] = 255
        output_best[good_mask_best, 2] = 255
        
        # Mismatch regions: frame_A compressed to 0-127
        if np.any(best_mismatch):
            compressed = (frame_normalized[best_mismatch] / 2).astype(np.uint8)
            output_best[best_mismatch, 0] = compressed
            output_best[best_mismatch, 1] = compressed
            output_best[best_mismatch, 2] = compressed
        
        axes[0, 2].imshow(output_best)
        axes[0, 2].set_title(f'Config Mismatch: Best Single vs Oracle\n({n_best_mismatch}/{n_hard_oracle})',
                            fontsize=11, fontweight='bold', pad=8)
        axes[0, 2].axis('off')
        
        # Draw boundary boxes on mismatch maps
        from matplotlib.patches import Rectangle
        valid_rows, valid_cols = np.where(valid_mask)
        if len(valid_rows) > 0:
            row_min, row_max = valid_rows.min(), valid_rows.max()
            col_min, col_max = valid_cols.min(), valid_cols.max()
            
            for ax_idx in [1, 2]:
                rect = Rectangle((col_min, row_min), 
                               col_max - col_min, row_max - row_min,
                               linewidth=2, edgecolor='yellow', facecolor='none',
                               linestyle='--')
                axes[0, ax_idx].add_patch(rect)
        
        # Build suptitle with all stats
        suptitle = f"Selection Error Analysis for EPE > {epe_cutoff} px\n"
        suptitle += f"Oracle: {n_hard_oracle} pixels, Ensemble: {n_hard_ensemble} pixels, Best Config: {n_hard_best} pixels"
        fig.suptitle(suptitle, fontsize=12, fontweight='bold', y=0.96)
        
    else:
        # No config data - show message
        for col in range(3):
            axes[0, col].text(0.5, 0.5, 'Config data not provided',
                            ha='center', va='center', fontsize=12, color='gray')
            axes[0, col].axis('off')
        
        # Simple suptitle
        fig.suptitle(f"Selection Error Analysis for EPE > {epe_cutoff} px", 
                    fontsize=12, fontweight='bold', y=0.96)
    
    # ========================================================================
    # Row 2: EPE Error Maps (filtered to EPE > threshold)
    # ========================================================================
    
    # Create masked EPE arrays (mask where EPE <= cutoff)
    epe_oracle_masked = np.ma.masked_where((epe_oracle <= epe_cutoff) | np.isnan(epe_oracle), epe_oracle)
    epe_ensemble_masked = np.ma.masked_where((epe_ensemble <= epe_cutoff) | np.isnan(epe_ensemble), epe_ensemble)
    epe_best_masked = np.ma.masked_where((epe_best <= epe_cutoff) | np.isnan(epe_best), epe_best)
    
    # Use blue-green colormap for errors
    from matplotlib.colors import LogNorm
    cmap_epe = plt.cm.winter.copy()
    cmap_epe.set_bad(color='white')
    
    # Compute vmax for log scale
    epe_vmax = np.nanmax([np.nanmax(epe_best), np.nanmax(epe_ensemble), np.nanmax(epe_oracle)])
    
    # Oracle EPE
    im_epe_oracle = axes[1, 0].imshow(epe_oracle_masked, cmap=cmap_epe, 
                                       norm=LogNorm(vmin=epe_cutoff, vmax=epe_vmax))
    axes[1, 0].set_title(f'Oracle\nEPE = {epe_oracle_mean:.3f} ± {epe_oracle_std:.3f} px',
                         fontsize=11, pad=5)
    axes[1, 0].axis('off')
    
    # Ensemble EPE
    im_epe_ensemble = axes[1, 1].imshow(epe_ensemble_masked, cmap=cmap_epe,
                                         norm=LogNorm(vmin=epe_cutoff, vmax=epe_vmax))
    axes[1, 1].set_title(f'Ensemble\nEPE = {epe_ensemble_mean:.3f} ± {epe_ensemble_std:.3f} px',
                         fontsize=11, pad=5)
    axes[1, 1].axis('off')
    
    # Best Single EPE
    im_epe_best = axes[1, 2].imshow(epe_best_masked, cmap=cmap_epe,
                                     norm=LogNorm(vmin=epe_cutoff, vmax=epe_vmax))
    axes[1, 2].set_title(f'Best Single\nEPE = {epe_best_mean:.3f} ± {epe_best_std:.3f} px',
                         fontsize=11, pad=5)
    axes[1, 2].axis('off')
    
    # ========================================================================
    # Add colorbar on the right side (for EPE row only)
    # ========================================================================
    
    cbar_ax_epe = fig.add_axes([0.93, 0.12, 0.015, 0.35])
    cbar_epe = fig.colorbar(im_epe_oracle, cax=cbar_ax_epe)
    cbar_epe.set_label('EPE (px)', fontsize=10)
    
    # Save
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved flow field comparison: {save_path}")
    print(f"   Hard pixels (EPE > {epe_cutoff}): Oracle={n_hard_oracle}, Ensemble={n_hard_ensemble}, Best={n_hard_best}")
    if ensemble_selection is not None:
        print(f"   Config mismatch: Ensemble={n_ensemble_mismatch}/{n_hard_oracle}, Best={n_best_mismatch}/{n_hard_oracle}")


def load_experiment_data(exp_dir: Path, weights_dir_name: str = None):
    """Load sweep and ensemble results."""
    
    # Load results_full.pkl
    results_file = exp_dir / 'results_full.pkl'
    if not results_file.exists():
        print(f"❌ ERROR: results_full.pkl not found in {exp_dir}")
        sys.exit(1)
    
    with open(results_file, 'rb') as f:
        results_full = pickle.load(f)
    
    print(f"✅ Loaded results_full.pkl ({len(results_full)} configs)")
    
    # Load sweep DataFrame
    sweep_csv = exp_dir / 'sweep_results.csv'
    if not sweep_csv.exists():
        print(f"❌ ERROR: sweep_results.csv not found")
        sys.exit(1)
    
    sweep_df = pd.read_csv(sweep_csv)
    
    # Find best single config
    best_idx = sweep_df['mean_epe_forward'].idxmin()
    best_config_name = sweep_df.loc[best_idx, 'config_name']
    best_epe = sweep_df.loc[best_idx, 'mean_epe_forward']
    print(f"✅ Best config: {best_config_name} (EPE: {best_epe:.6f})")
    
    # Load ground truth
    u_truth = np.load(exp_dir / 'u_truth.npy')
    v_truth = np.load(exp_dir / 'v_truth.npy')
    valid_mask = np.load(exp_dir / 'valid_mask.npy')
    print(f"✅ Loaded ground truth")
    
    # Load ensemble results
    if weights_dir_name is None:
        weights_dirs = sorted(exp_dir.glob('weights_optim_*'))
        if not weights_dirs:
            print(f"❌ ERROR: No weights_optim_* directories found")
            sys.exit(1)
        weights_dir = weights_dirs[-1]
        print(f"✅ Using: {weights_dir.name}")
    else:
        weights_dir = exp_dir / weights_dir_name
    
    ensemble_file = weights_dir / 'ensemble_results.npz'
    if not ensemble_file.exists():
        print(f"❌ ERROR: ensemble_results.npz not found")
        sys.exit(1)
    
    ensemble_data = np.load(ensemble_file)
    ensemble_epe = float(ensemble_data['ensemble_epe'])
    print(f"✅ Loaded ensemble results (EPE: {ensemble_epe:.6f})")
    
    return {
        'results_full': results_full,
        'sweep_df': sweep_df,
        'ensemble_data': ensemble_data,
        'u_truth': u_truth,
        'v_truth': v_truth,
        'valid_mask': valid_mask,
        'best_idx': best_idx,
        'best_config_name': best_config_name,
        'weights_dir': weights_dir
    }


def compute_oracle_flow(results_full, u_truth, v_truth, valid_mask):
    """Compute oracle (pixel-wise best) flow and config selection."""
    n_configs = len(results_full)
    H, W = u_truth.shape
    
    # Stack all configs
    u_stack = np.stack([results_full[i]['u_AB'] for i in range(n_configs)], axis=0)
    v_stack = np.stack([results_full[i]['v_AB'] for i in range(n_configs)], axis=0)
    
    # Compute EPE for each config at each pixel
    epe_stack = np.sqrt((u_stack - u_truth[np.newaxis, :, :])**2 + 
                        (v_stack - v_truth[np.newaxis, :, :])**2)
    
    # Find best config per pixel
    oracle_selection = np.argmin(epe_stack, axis=0)
    
    # Build oracle flow
    u_oracle = np.zeros((H, W), dtype=np.float32)
    v_oracle = np.zeros((H, W), dtype=np.float32)
    
    for i in range(n_configs):
        mask = oracle_selection == i
        u_oracle[mask] = u_stack[i][mask]
        v_oracle[mask] = v_stack[i][mask]
    
    # Compute oracle EPE
    epe_oracle = np.sqrt((u_oracle - u_truth)**2 + (v_oracle - v_truth)**2)
    oracle_epe = np.nanmean(epe_oracle[valid_mask])
    
    print(f"✅ Computed oracle flow (EPE: {oracle_epe:.6f})")
    
    return u_oracle, v_oracle, oracle_selection


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
    Extracted from ensemble_flow.py for better organization.
    
    Args:
        results_full: List of full result dictionaries
        sweep_df: DataFrame with sweep statistics
        ensemble_results: Dict with ensemble flows and selection
        test_data: Dict with test images and ground truth
        oracle: Dict with oracle EPE values (unused here, kept for API compat)
        figures_dir: Path to save figures
    
    Returns:
        dict: Flow data for reuse in other figures {
            'u_oracle_fwd', 'v_oracle_fwd',
            'u_oracle_sym', 'v_oracle_sym',
            'u_best_fwd', 'v_best_fwd',
            'u_best_sym', 'v_best_sym',
            'best_idx'
        }
    """
    from src.core.data_structures import flatten_for_visualization
    
    # Flatten results if needed
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
    oracle_selection_fwd = np.argmin(epe_stack_fwd, axis=0)
    
    # Build oracle flow (forward)
    u_oracle_fwd = np.zeros((H, W), dtype=np.float32)
    v_oracle_fwd = np.zeros((H, W), dtype=np.float32)
    
    for i in range(n_configs):
        mask = oracle_selection_fwd == i
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
        oracle_selection=oracle_selection_fwd,
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


def main():
    """CLI entry point for generating flow comparison figures."""
    parser = argparse.ArgumentParser(
        description='Generate comprehensive flow comparison figure from experiment results'
    )
    parser.add_argument('experiment_dir', type=Path,
                       help='Path to experiment directory (e.g., results/e9e4e6/dis_f39dcf_e9e4e6)')
    parser.add_argument('--weights-dir', type=str, default=None,
                       help='Weights directory name (default: latest weights_optim_*)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: weights_dir/figures/flow_comparison.png)')
    
    args = parser.parse_args()
    
    if not args.experiment_dir.exists():
        print(f"❌ ERROR: Directory not found: {args.experiment_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("FLOW FIELD COMPARISON")
    print("=" * 80)
    print(f"Experiment: {args.experiment_dir}")
    print()
    
    # Load data
    print("📂 Loading data...")
    data = load_experiment_data(args.experiment_dir, args.weights_dir)
    print()
    
    # Extract flows
    results_full = data['results_full']
    ensemble = data['ensemble_data']
    best_idx = data['best_idx']
    
    # Best single config
    u_best = results_full[best_idx]['u_AB']
    v_best = results_full[best_idx]['v_AB']
    
    # Ensemble
    u_ensemble = ensemble['u_forward']
    v_ensemble = ensemble['v_forward']
    
    # Ground truth
    u_truth = data['u_truth']
    v_truth = data['v_truth']
    valid_mask = data['valid_mask']
    
    # Compute oracle
    print("🔍 Computing oracle flow...")
    u_oracle, v_oracle, oracle_selection = compute_oracle_flow(results_full, u_truth, v_truth, valid_mask)
    print()
    
    # Load frame A for disagreement visualization
    frame_A = np.load(args.experiment_dir / 'frame1.npy')
    print(f"✅ Loaded frame1.npy")
    
    # Get ensemble selection from ensemble_results
    ensemble_selection = ensemble['selection'] if 'selection' in ensemble else None
    
    if ensemble_selection is None:
        print("⚠️  Warning: ensemble_selection not found in ensemble_results.npz")
        print("   Row 3 will show 'data not provided' message")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data['weights_dir'] / 'figures' / 'flow_comparison.png'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate figure
    print("🎨 Generating figure...")
    visualize_flow_field_comparison(
        u_best, v_best,
        u_ensemble, v_ensemble,
        u_oracle, v_oracle,
        u_truth, v_truth,
        valid_mask,
        best_config_name=data['best_config_name'],
        frame_A=frame_A,
        ensemble_selection=ensemble_selection,
        oracle_selection=oracle_selection,
        n_configs=len(results_full),
        best_config_idx=best_idx,
        save_path=str(output_path),
        dpi=150
    )
    
    print()
    print("=" * 80)
    print("✅ COMPLETE")
    print("=" * 80)
    print(f"Output: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
