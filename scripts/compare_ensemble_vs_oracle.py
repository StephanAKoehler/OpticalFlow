#!/usr/bin/env python3
# File: scripts/compare_ensemble_vs_oracle.py
"""
Compare Best Single Config vs Ensemble vs Oracle

Generates 3-column comparison figure:
- Column 1: Best single config (lowest EPE)
- Column 2: Ensemble (from optimization)
- Column 3: Oracle (pixel-wise best)

Rows:
- Row 1: Flow magnitude with EPE overlay
- Row 2: EPE error maps

Usage:
    compare_ensemble_vs_oracle.py path/to/experiment/dir
    compare_ensemble_vs_oracle.py path/to/experiment/dir --weights-dir weights_optim_001
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def load_experiment_data(exp_dir: Path, weights_dir_name: str = None):
    """
    Load all necessary data from experiment directory.
    
    Returns dict with:
        - sweep_results: npz with all configs
        - sweep_df: DataFrame with per-config stats
        - ensemble_results: npz with ensemble flows
        - ground_truth: u_truth, v_truth, valid_mask
        - best_config_idx: index of best single config
        - best_config_name: name of best single config
    """
    
    # Load sweep results
    sweep_file = exp_dir / 'sweep_results.npz'
    if not sweep_file.exists():
        print(f"❌ ERROR: sweep_results.npz not found in {exp_dir}")
        sys.exit(1)
    
    sweep_data = np.load(sweep_file)
    print(f"✅ Loaded sweep results from {sweep_file.name}")
    print(f"   Available keys: {list(sweep_data.keys())}")
    
    # Load sweep DataFrame
    sweep_csv = exp_dir / 'sweep_results.csv'
    if not sweep_csv.exists():
        print(f"❌ ERROR: sweep_results.csv not found in {exp_dir}")
        sys.exit(1)
    
    sweep_df = pd.read_csv(sweep_csv)
    print(f"✅ Loaded sweep DataFrame: {len(sweep_df)} configs")
    
    # Find best single config
    best_idx = sweep_df['epe_forward'].idxmin()
    best_epe = sweep_df.loc[best_idx, 'epe_forward']
    best_config_name = sweep_df.loc[best_idx, 'config_name']
    
    print(f"✅ Best single config: {best_config_name}")
    print(f"   EPE: {best_epe:.6f} px")
    
    # Load ensemble results
    if weights_dir_name is None:
        # Find latest weights_optim directory
        weights_dirs = sorted(exp_dir.glob('weights_optim_*'))
        if not weights_dirs:
            print(f"❌ ERROR: No weights_optim_* directories found in {exp_dir}")
            sys.exit(1)
        weights_dir = weights_dirs[-1]
        print(f"✅ Using latest weights directory: {weights_dir.name}")
    else:
        weights_dir = exp_dir / weights_dir_name
        if not weights_dir.exists():
            print(f"❌ ERROR: {weights_dir_name} not found in {exp_dir}")
            sys.exit(1)
        print(f"✅ Using specified weights directory: {weights_dir.name}")
    
    ensemble_file = weights_dir / 'ensemble_results.npz'
    if not ensemble_file.exists():
        print(f"❌ ERROR: ensemble_results.npz not found in {weights_dir}")
        sys.exit(1)
    
    ensemble_data = np.load(ensemble_file)
    ensemble_epe = float(ensemble_data['ensemble_epe'])
    print(f"✅ Loaded ensemble results")
    print(f"   EPE: {ensemble_epe:.6f} px")
    
    # Extract ground truth
    u_truth = sweep_data['u_truth']
    v_truth = sweep_data['v_truth']
    valid_mask = sweep_data['valid_mask']
    
    # Extract oracle EPE
    oracle_epe = float(sweep_data['oracle_epe_forward'])
    print(f"✅ Oracle EPE: {oracle_epe:.6f} px")
    
    return {
        'sweep_data': sweep_data,
        'sweep_df': sweep_df,
        'ensemble_data': ensemble_data,
        'ensemble_epe': ensemble_epe,
        'u_truth': u_truth,
        'v_truth': v_truth,
        'valid_mask': valid_mask,
        'best_config_idx': best_idx,
        'best_config_name': best_config_name,
        'best_epe': best_epe,
        'oracle_epe': oracle_epe,
    }


def compute_flow_magnitude(u, v):
    """Compute flow magnitude."""
    return np.sqrt(u**2 + v**2)


def compute_epe(u, v, u_truth, v_truth):
    """Compute EPE map."""
    return np.sqrt((u - u_truth)**2 + (v - v_truth)**2)


def create_comparison_figure(data: dict, output_path: Path):
    """
    Create 3-column comparison figure.
    
    Layout:
    ┌─────────────────────────┬─────────────────┬─────────────────┐
    │   Best Single           │    Ensemble     │     Oracle      │
    │ [config name]           │                 │                 │
    ├─────────────────────────┼─────────────────┼─────────────────┤
    │ [flow magnitude]        │ [flow mag]      │ [flow mag]      │  Row 1
    │ EPE: X.XXXX             │ EPE: X.XXXX     │ EPE: X.XXXX     │
    ├─────────────────────────┼─────────────────┼─────────────────┤
    │ [error map]             │ [error map]     │ [error map]     │  Row 2
    └─────────────────────────┴─────────────────┴─────────────────┘
    """
    
    # Extract data
    sweep_data = data['sweep_data']
    ensemble_data = data['ensemble_data']
    best_idx = data['best_config_idx']
    
    # Get flows
    u_best = sweep_data['configs_u_forward'][best_idx]
    v_best = sweep_data['configs_v_forward'][best_idx]
    
    u_ensemble = ensemble_data['u_forward']
    v_ensemble = ensemble_data['v_forward']
    
    u_oracle = sweep_data['oracle_u_forward']
    v_oracle = sweep_data['oracle_v_forward']
    
    u_truth = data['u_truth']
    v_truth = data['v_truth']
    valid_mask = data['valid_mask']
    
    # Compute magnitudes
    mag_best = compute_flow_magnitude(u_best, v_best)
    mag_ensemble = compute_flow_magnitude(u_ensemble, v_ensemble)
    mag_oracle = compute_flow_magnitude(u_oracle, v_oracle)
    
    # Compute EPE maps
    epe_best = compute_epe(u_best, v_best, u_truth, v_truth)
    epe_ensemble = compute_epe(u_ensemble, v_ensemble, u_truth, v_truth)
    epe_oracle = compute_epe(u_oracle, v_oracle, u_truth, v_truth)
    
    # Mask invalid regions
    mag_best[~valid_mask] = np.nan
    mag_ensemble[~valid_mask] = np.nan
    mag_oracle[~valid_mask] = np.nan
    epe_best[~valid_mask] = np.nan
    epe_ensemble[~valid_mask] = np.nan
    epe_oracle[~valid_mask] = np.nan
    
    # Compute EPE statistics
    epe_best_mean = np.nanmean(epe_best)
    epe_ensemble_mean = np.nanmean(epe_ensemble)
    epe_oracle_mean = np.nanmean(epe_oracle)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Column titles (only first column gets 2-line title)
    axes[0, 0].set_title(f'Best Single Config\n{data["best_config_name"]}', 
                         fontsize=12, fontweight='bold', pad=10)
    axes[0, 1].set_title('Ensemble', fontsize=12, fontweight='bold', pad=10)
    axes[0, 2].set_title('Oracle', fontsize=12, fontweight='bold', pad=10)
    
    # Compute colormaps limits
    mag_vmax = np.nanmax([np.nanmax(mag_best), np.nanmax(mag_ensemble), np.nanmax(mag_oracle)])
    epe_vmax = np.nanmax([np.nanmax(epe_best), np.nanmax(epe_ensemble), np.nanmax(epe_oracle)])
    
    # Row 1: Flow magnitude
    im_mag_best = axes[0, 0].imshow(mag_best, cmap='viridis', vmin=0, vmax=mag_vmax)
    axes[0, 0].text(0.02, 0.98, f'EPE: {epe_best_mean:.4f}', 
                    transform=axes[0, 0].transAxes,
                    fontsize=10, color='white', weight='bold',
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    axes[0, 0].set_ylabel('Flow Magnitude (px)', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    im_mag_ensemble = axes[0, 1].imshow(mag_ensemble, cmap='viridis', vmin=0, vmax=mag_vmax)
    axes[0, 1].text(0.02, 0.98, f'EPE: {epe_ensemble_mean:.4f}', 
                    transform=axes[0, 1].transAxes,
                    fontsize=10, color='white', weight='bold',
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    axes[0, 1].axis('off')
    
    im_mag_oracle = axes[0, 2].imshow(mag_oracle, cmap='viridis', vmin=0, vmax=mag_vmax)
    axes[0, 2].text(0.02, 0.98, f'EPE: {epe_oracle_mean:.4f}', 
                    transform=axes[0, 2].transAxes,
                    fontsize=10, color='white', weight='bold',
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    axes[0, 2].axis('off')
    
    # Add colorbar for row 1
    cbar_mag = fig.colorbar(im_mag_oracle, ax=axes[0, :], orientation='vertical', 
                            fraction=0.046, pad=0.04)
    cbar_mag.set_label('Flow Magnitude (px)', fontsize=10)
    
    # Row 2: EPE error maps
    im_epe_best = axes[1, 0].imshow(epe_best, cmap='hot', vmin=0, vmax=epe_vmax)
    axes[1, 0].set_ylabel('EPE (px)', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    im_epe_ensemble = axes[1, 1].imshow(epe_ensemble, cmap='hot', vmin=0, vmax=epe_vmax)
    axes[1, 1].axis('off')
    
    im_epe_oracle = axes[1, 2].imshow(epe_oracle, cmap='hot', vmin=0, vmax=epe_vmax)
    axes[1, 2].axis('off')
    
    # Add colorbar for row 2
    cbar_epe = fig.colorbar(im_epe_oracle, ax=axes[1, :], orientation='vertical',
                            fraction=0.046, pad=0.04)
    cbar_epe.set_label('EPE (px)', fontsize=10)
    
    # Add summary statistics as text
    stats_text = (
        f'Best Single: {epe_best_mean:.6f} px\n'
        f'Ensemble:    {epe_ensemble_mean:.6f} px ({100*(epe_ensemble_mean/epe_best_mean - 1):+.1f}%)\n'
        f'Oracle:      {epe_oracle_mean:.6f} px ({100*(epe_oracle_mean/epe_best_mean - 1):+.1f}%)\n'
        f'Ens vs Oracle: {100*(epe_ensemble_mean/epe_oracle_mean - 1):+.1f}%'
    )
    
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
             family='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved figure to {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Compare Best Single Config vs Ensemble vs Oracle',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use latest weights_optim directory
  compare_ensemble_vs_oracle.py experiments/abc123def456/
  
  # Specify weights directory
  compare_ensemble_vs_oracle.py experiments/abc123def456/ --weights-dir weights_optim_002
        """
    )
    
    parser.add_argument('experiment_dir', type=Path,
                       help='Path to experiment directory')
    parser.add_argument('--weights-dir', type=str, default=None,
                       help='Name of weights_optim directory (default: latest)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename (default: comparison_ensemble_oracle.png)')
    
    args = parser.parse_args()
    
    # Validate experiment directory
    if not args.experiment_dir.exists():
        print(f"❌ ERROR: Experiment directory not found: {args.experiment_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("ENSEMBLE VS ORACLE COMPARISON")
    print("=" * 80)
    print(f"Experiment: {args.experiment_dir}")
    print()
    
    # Load data
    print("📂 Loading data...")
    data = load_experiment_data(args.experiment_dir, args.weights_dir)
    print()
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        if args.weights_dir:
            output_path = args.experiment_dir / args.weights_dir / 'figures' / 'comparison_ensemble_oracle.png'
        else:
            weights_dirs = sorted(args.experiment_dir.glob('weights_optim_*'))
            output_path = weights_dirs[-1] / 'figures' / 'comparison_ensemble_oracle.png'
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate figure
    print("📊 Generating comparison figure...")
    fig = create_comparison_figure(data, output_path)
    
    print()
    print("=" * 80)
    print("✅ COMPARISON COMPLETE")
    print("=" * 80)
    print(f"Figure: {output_path}")
    print()
    print("Summary:")
    print(f"  Best single: {data['best_epe']:.6f} px ({data['best_config_name']})")
    print(f"  Ensemble:    {data['ensemble_epe']:.6f} px ({100*(data['ensemble_epe']/data['best_epe'] - 1):+.1f}%)")
    print(f"  Oracle:      {data['oracle_epe']:.6f} px ({100*(data['oracle_epe']/data['best_epe'] - 1):+.1f}%)")
    print(f"  Gap (Ens→Oracle): {100*(data['ensemble_epe']/data['oracle_epe'] - 1):+.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
