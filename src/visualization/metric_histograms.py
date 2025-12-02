 # File: src/visualization/metric_histograms.py
"""
Comprehensive histogram/violin plots for all metrics across configs.

Shows distributions of EPE and cost components to understand config behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



def hide_violin_whiskers(parts):
    """Remove vertical bars/whiskers from violin plots."""
    for partname in ('cbars', 'cmaxes', 'cmins', 'cmedians', 'cmeans'):
        if partname in parts:
            parts[partname].set_visible(False)


def plot_metric_histograms(results_dict: dict, output_path: Path):
    """
    Create 3×3 grid of metric distributions.
    
    Row 0: u component | v component | Flow magnitude
    Row 1: Forward EPE | Symmetric EPE | Backward EPE
    Row 2: Traction | Consistency | Photometric
    
    Each plot shows configs + Oracle + Ensemble as violins
    
    Args:
        results_dict: Dictionary with all pipeline results
        output_path: Where to save figure
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    # Extract data
    config_names = results_dict['config_names']
    valid_mask = results_dict['valid_mask']
    n_configs = len(config_names)
    
    results_full = results_dict['results_full']
    
    EPE_forward_stack = results_dict['EPE_forward_stack']
    EPE_symmetric_stack = results_dict['EPE_symmetric_stack']
    
    oracle_epe_forward = results_dict['oracle_epe_forward']
    oracle_epe_symmetric = results_dict['oracle_epe_symmetric']
    ensemble_epe_forward = results_dict['ensemble_epe_forward']
    ensemble_epe_symmetric = results_dict['ensemble_epe_symmetric']
    
    u_true = results_dict['u_true']
    v_true = results_dict['v_true']
    
    # Extract Oracle and Ensemble flows
    u_oracle_forward = results_dict.get('u_oracle_forward', np.zeros_like(u_true))
    v_oracle_forward = results_dict.get('v_oracle_forward', np.zeros_like(v_true))
    u_ensemble_forward = results_dict.get('u_ensemble_forward', np.zeros_like(u_true))
    v_ensemble_forward = results_dict.get('v_ensemble_forward', np.zeros_like(v_true))
    
    # Positions: n_configs + Oracle + Ensemble
    positions = np.arange(n_configs + 2)
    labels = config_names + ['Oracle', 'Ensemble']
    
    # ========================================================================
    # ROW 0: Flow Components
    # ========================================================================
    
    # === COL 0: u component ===
    data_u = []
    for result in results_full:
        u_vals = result['u_AB'][valid_mask]
        u_vals = u_vals[~np.isnan(u_vals)]
        data_u.append(u_vals)
    
    # Add Oracle
    data_u.append(u_oracle_forward[valid_mask][~np.isnan(u_oracle_forward[valid_mask])])
    # Add Ensemble
    data_u.append(u_ensemble_forward[valid_mask][~np.isnan(u_ensemble_forward[valid_mask])])
    
    parts = axes[0, 0].violinplot(data_u, positions=positions, widths=0.7,
                                  showmeans=False, showmedians=False)
    hide_violin_whiskers(parts)
    
    # Color differently
    for i, pc in enumerate(parts['bodies']):
        if i < n_configs:
            pc.set_facecolor('steelblue')
        elif i == n_configs:  # Oracle
            pc.set_facecolor('green')
        else:  # Ensemble
            pc.set_facecolor('red')
        pc.set_alpha(0.7)
    
    u_true_mean = u_true[valid_mask].mean()
    if abs(u_true_mean) > 0.001:
        axes[0, 0].axhline(u_true_mean, color='green', linestyle='--',
                          linewidth=2, label=f'GT: {u_true_mean:.2f}')
        axes[0, 0].legend(loc='upper right', fontsize=8)
    
    axes[0, 0].set_xticks(positions)
    axes[0, 0].set_xticklabels([])
    axes[0, 0].set_ylabel('u (pixels)', fontsize=10)
    axes[0, 0].set_title('u Component (Forward)', fontweight='bold', fontsize=12)
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].axhline(0, color='black', linewidth=0.5, alpha=0.3)
    
    # === COL 1: v component ===
    data_v = []
    for result in results_full:
        v_vals = result['v_AB'][valid_mask]
        v_vals = v_vals[~np.isnan(v_vals)]
        data_v.append(v_vals)
    
    # Add Oracle
    data_v.append(v_oracle_forward[valid_mask][~np.isnan(v_oracle_forward[valid_mask])])
    # Add Ensemble
    data_v.append(v_ensemble_forward[valid_mask][~np.isnan(v_ensemble_forward[valid_mask])])
    
    parts = axes[0, 1].violinplot(data_v, positions=positions, widths=0.7,
                                  showmeans=False, showmedians=False)
    hide_violin_whiskers(parts)
    
    for i, pc in enumerate(parts['bodies']):
        if i < n_configs:
            pc.set_facecolor('coral')
        elif i == n_configs:
            pc.set_facecolor('green')
        else:
            pc.set_facecolor('red')
        pc.set_alpha(0.7)
    
    v_true_mean = v_true[valid_mask].mean()
    if abs(v_true_mean) > 0.001:
        axes[0, 1].axhline(v_true_mean, color='green', linestyle='--',
                          linewidth=2, label=f'GT: {v_true_mean:.2f}')
        axes[0, 1].legend(loc='upper right', fontsize=8)
    
    axes[0, 1].set_xticks(positions)
    axes[0, 1].set_xticklabels([])
    axes[0, 1].set_ylabel('v (pixels)', fontsize=10)
    axes[0, 1].set_title('v Component (Forward)', fontweight='bold', fontsize=12)
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].axhline(0, color='black', linewidth=0.5, alpha=0.3)
    
    # === COL 2: Flow magnitude ===
    data_mag = []
    for result in results_full:
        u_vals = result['u_AB'][valid_mask]
        v_vals = result['v_AB'][valid_mask]
        mag = np.sqrt(u_vals**2 + v_vals**2)
        mag = mag[~np.isnan(mag)]
        data_mag.append(mag)
    
    # Add Oracle
    mag_oracle = np.sqrt(u_oracle_forward**2 + v_oracle_forward**2)[valid_mask]
    data_mag.append(mag_oracle[~np.isnan(mag_oracle)])
    # Add Ensemble
    mag_ensemble = np.sqrt(u_ensemble_forward**2 + v_ensemble_forward**2)[valid_mask]
    data_mag.append(mag_ensemble[~np.isnan(mag_ensemble)])
    
    parts = axes[0, 2].violinplot(data_mag, positions=positions, widths=0.7,
                                  showmeans=False, showmedians=False)
    hide_violin_whiskers(parts)
    
    for i, pc in enumerate(parts['bodies']):
        if i < n_configs:
            pc.set_facecolor('mediumpurple')
        elif i == n_configs:
            pc.set_facecolor('green')
        else:
            pc.set_facecolor('red')
        pc.set_alpha(0.7)
    
    mag_true_mean = np.sqrt(u_true**2 + v_true**2)[valid_mask].mean()
    if mag_true_mean > 0.001:
        axes[0, 2].axhline(mag_true_mean, color='green', linestyle='--',
                          linewidth=2, label=f'GT: {mag_true_mean:.2f}')
        axes[0, 2].legend(loc='upper right', fontsize=8)
    
    axes[0, 2].set_xticks(positions)
    axes[0, 2].set_xticklabels([])
    axes[0, 2].set_ylabel('Magnitude (pixels)', fontsize=10)
    axes[0, 2].set_title('Flow Magnitude', fontweight='bold', fontsize=12)
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # ========================================================================
    # ROW 1: EPE Distributions
    # ========================================================================
    
    # Compute backward EPE
    EPE_backward_stack = np.zeros((n_configs, *u_true.shape), dtype=np.float32)
    for i, result in enumerate(results_full):
        u_pred = result['u_BA']
        v_pred = result['v_BA']
        EPE_backward_stack[i] = np.sqrt((u_pred + u_true)**2 + (v_pred + v_true)**2)
    
    oracle_epe_backward = np.nanmean(EPE_backward_stack.min(axis=0)[valid_mask])
    
    u_ensemble_backward = results_dict.get('u_ensemble_backward', np.zeros_like(u_true))
    v_ensemble_backward = results_dict.get('v_ensemble_backward', np.zeros_like(v_true))
    ensemble_epe_backward = np.nanmean(
        np.sqrt((u_ensemble_backward + u_true)**2 + (v_ensemble_backward + v_true)**2)[valid_mask]
    )
    
    # Prepare EPE data arrays
    data_forward = []
    data_symmetric = []
    data_backward = []
    
    for i in range(n_configs):
        # Forward
        epe = EPE_forward_stack[i][valid_mask]
        epe = epe[~np.isnan(epe)]
        epe_nonzero = epe[epe > 0]
        data_forward.append(epe_nonzero if len(epe_nonzero) > 0 else epe)
        
        # Symmetric
        epe = EPE_symmetric_stack[i][valid_mask]
        epe = epe[~np.isnan(epe)]
        epe_nonzero = epe[epe > 0]
        data_symmetric.append(epe_nonzero if len(epe_nonzero) > 0 else epe)
        
        # Backward
        epe = EPE_backward_stack[i][valid_mask]
        epe = epe[~np.isnan(epe)]
        epe_nonzero = epe[epe > 0]
        data_backward.append(epe_nonzero if len(epe_nonzero) > 0 else epe)
    
    # Add Oracle and Ensemble EPE
    EPE_oracle_fwd = np.sqrt((u_oracle_forward - u_true)**2 + (v_oracle_forward - v_true)**2)[valid_mask]
    data_forward.append(EPE_oracle_fwd[~np.isnan(EPE_oracle_fwd)])
    
    EPE_ensemble_fwd = np.sqrt((u_ensemble_forward - u_true)**2 + (v_ensemble_forward - v_true)**2)[valid_mask]
    data_forward.append(EPE_ensemble_fwd[~np.isnan(EPE_ensemble_fwd)])
    
    # For symmetric - use same Oracle/Ensemble (placeholder)
    data_symmetric.append(EPE_oracle_fwd[~np.isnan(EPE_oracle_fwd)])
    data_symmetric.append(EPE_ensemble_fwd[~np.isnan(EPE_ensemble_fwd)])
    
    # For backward - use same Oracle/Ensemble (placeholder)  
    data_backward.append(EPE_oracle_fwd[~np.isnan(EPE_oracle_fwd)])
    data_backward.append(EPE_ensemble_fwd[~np.isnan(EPE_ensemble_fwd)])
    
    # Find global EPE range
    all_epe = []
    for i in range(n_configs):
        all_epe.extend(data_forward[i])
        all_epe.extend(data_symmetric[i])
        all_epe.extend(data_backward[i])
    
    if len(all_epe) > 0:
        epe_min = 1e-3  # Subpixel resolution threshold
        epe_max = np.nanmax(all_epe)
        use_log = epe_max > 0 and (epe_max / epe_min) > 10
    else:
        epe_min, epe_max, use_log = 1e-3, 1.0, False
    
    # === COL 0: Forward EPE ===
    parts = axes[1, 0].violinplot(data_forward, positions=positions, widths=0.7,
                                  showmeans=False, showmedians=False)
    hide_violin_whiskers(parts)
    for i, pc in enumerate(parts['bodies']):
        if i < n_configs:
            pc.set_facecolor('lightblue')
        elif i == n_configs:
            pc.set_facecolor('green')
        else:
            pc.set_facecolor('red')
        pc.set_alpha(0.7)
    
    # Add 25/50/75 percentiles below x-axis (stacked vertically, 75% on top)
    for i, pos in enumerate(positions):
        p25 = np.percentile(data_forward[i], 25)
        p50 = np.percentile(data_forward[i], 50)
        p75 = np.percentile(data_forward[i], 75)
        # Stack vertically: 75% → 50% → 25% (top to bottom)
        axes[1, 0].text(pos, 0, f'{p75:.4f}\n{p50:.4f}\n{p25:.4f}',
                       ha='center', va='top', fontsize=7,
                       color='black', weight='bold',
                       transform=axes[1, 0].get_xaxis_transform())
    
    axes[1, 0].set_xticks(positions)
    axes[1, 0].set_xticklabels([])
    axes[1, 0].set_ylabel('EPE (pixels)', fontsize=10)
    axes[1, 0].set_title('Forward EPE (A→B)', fontweight='bold', fontsize=12)
    if use_log:
        axes[1, 0].set_yscale('log')
    axes[1, 0].set_ylim([epe_min, epe_max])
    axes[1, 0].grid(axis='y', alpha=0.3, which='both')
    
    # === COL 1: Symmetric EPE ===
    parts = axes[1, 1].violinplot(data_symmetric, positions=positions, widths=0.7,
                                  showmeans=False, showmedians=False)
    hide_violin_whiskers(parts)
    for i, pc in enumerate(parts['bodies']):
        if i < n_configs:
            pc.set_facecolor('lightgreen')
        elif i == n_configs:
            pc.set_facecolor('green')
        else:
            pc.set_facecolor('red')
        pc.set_alpha(0.7)
    
    # Add 25/50/75 percentiles below x-axis (stacked vertically, 75% on top)
    for i, pos in enumerate(positions):
        p25 = np.percentile(data_symmetric[i], 25)
        p50 = np.percentile(data_symmetric[i], 50)
        p75 = np.percentile(data_symmetric[i], 75)
        # Stack vertically: 75% → 50% → 25% (top to bottom)
        axes[1, 1].text(pos, 0, f'{p75:.4f}\n{p50:.4f}\n{p25:.4f}',
                       ha='center', va='top', fontsize=7,
                       color='black', weight='bold',
                       transform=axes[1, 1].get_xaxis_transform())
    
    axes[1, 1].set_xticks(positions)
    axes[1, 1].set_xticklabels([])
    axes[1, 1].set_ylabel('EPE (pixels)', fontsize=10)
    axes[1, 1].set_title('Symmetric EPE', fontweight='bold', fontsize=12)
    if use_log:
        axes[1, 1].set_yscale('log')
    axes[1, 1].set_ylim([epe_min, epe_max])
    axes[1, 1].grid(axis='y', alpha=0.3, which='both')
    
    # === COL 2: Backward EPE ===
    parts = axes[1, 2].violinplot(data_backward, positions=positions, widths=0.7,
                                  showmeans=False, showmedians=False)
    hide_violin_whiskers(parts)
    for i, pc in enumerate(parts['bodies']):
        if i < n_configs:
            pc.set_facecolor('lightcoral')
        elif i == n_configs:
            pc.set_facecolor('green')
        else:
            pc.set_facecolor('red')
        pc.set_alpha(0.7)
    
    # Add 25/50/75 percentiles below x-axis (stacked vertically, 75% on top)
    for i, pos in enumerate(positions):
        p25 = np.percentile(data_backward[i], 25)
        p50 = np.percentile(data_backward[i], 50)
        p75 = np.percentile(data_backward[i], 75)
        # Stack vertically: 75% → 50% → 25% (top to bottom)
        axes[1, 2].text(pos, 0, f'{p75:.4f}\n{p50:.4f}\n{p25:.4f}',
                       ha='center', va='top', fontsize=7,
                       color='black', weight='bold',
                       transform=axes[1, 2].get_xaxis_transform())
    
    axes[1, 2].set_xticks(positions)
    axes[1, 2].set_xticklabels([])
    axes[1, 2].set_ylabel('EPE (pixels)', fontsize=10)
    axes[1, 2].set_title('Backward EPE (B→A)', fontweight='bold', fontsize=12)
    if use_log:
        axes[1, 2].set_yscale('log')
    axes[1, 2].set_ylim([epe_min, epe_max])
    axes[1, 2].grid(axis='y', alpha=0.3, which='both')
    
    # ========================================================================
    # ROW 2: Cost Components
    # ========================================================================
    
    # === COL 0: Traction ===
    data_traction = []
    for result in results_full:
        trac_avg = (result['traction_A'] + result['traction_B']) / 2
        trac = trac_avg[valid_mask]
        trac = trac[~np.isnan(trac)]
        data_traction.append(trac)
    
    # Add Oracle traction
    trac_oracle = results_dict.get('traction_oracle', np.zeros_like(u_true))
    oracle_trac = trac_oracle[valid_mask]
    data_traction.append(oracle_trac[~np.isnan(oracle_trac)])
    
    # Add Ensemble traction
    trac_ensemble = results_dict.get('traction_ensemble', np.zeros_like(u_true))
    ensemble_trac = trac_ensemble[valid_mask]
    data_traction.append(ensemble_trac[~np.isnan(ensemble_trac)])
    
    parts = axes[2, 0].violinplot(data_traction, positions=positions, widths=0.7,
                                  showmeans=False, showmedians=False)
    hide_violin_whiskers(parts)
    
    for i, pc in enumerate(parts['bodies']):
        if i < n_configs:
            pc.set_facecolor('plum')
        elif i == n_configs:
            pc.set_facecolor('green')
        else:
            pc.set_facecolor('red')
        pc.set_alpha(0.7)
    
    axes[2, 0].set_xticks(positions)
    axes[2, 0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[2, 0].set_ylabel('Traction (pixels)', fontsize=10)
    axes[2, 0].set_title('Traction (avg A+B)', fontweight='bold', fontsize=12)
    axes[2, 0].grid(axis='y', alpha=0.3)
    
    # === COL 1: Consistency ===
    data_consistency = []
    for result in results_full:
        cons_avg = (result['consistency_A'] + result['consistency_B']) / 2
        cons = cons_avg[valid_mask]
        cons = cons[~np.isnan(cons)]
        data_consistency.append(cons)
    
    # Add Oracle consistency
    u_oracle_forward = results_dict.get('u_oracle_forward', np.zeros_like(u_true))
    v_oracle_forward = results_dict.get('v_oracle_forward', np.zeros_like(v_true))
    u_oracle_backward = results_dict.get('u_oracle_backward', np.zeros_like(u_true))
    v_oracle_backward = results_dict.get('v_oracle_backward', np.zeros_like(v_true))
    oracle_consistency = np.sqrt((u_oracle_forward + u_oracle_backward)**2 + 
                                 (v_oracle_forward + v_oracle_backward)**2)
    oracle_cons = oracle_consistency[valid_mask]
    data_consistency.append(oracle_cons[~np.isnan(oracle_cons)])
    
    # Add Ensemble consistency
    u_ensemble_forward = results_dict.get('u_ensemble_forward', np.zeros_like(u_true))
    v_ensemble_forward = results_dict.get('v_ensemble_forward', np.zeros_like(v_true))
    u_ensemble_backward = results_dict.get('u_ensemble_backward', np.zeros_like(u_true))
    v_ensemble_backward = results_dict.get('v_ensemble_backward', np.zeros_like(v_true))
    ensemble_consistency = np.sqrt((u_ensemble_forward + u_ensemble_backward)**2 + 
                                   (v_ensemble_forward + v_ensemble_backward)**2)
    ensemble_cons = ensemble_consistency[valid_mask]
    data_consistency.append(ensemble_cons[~np.isnan(ensemble_cons)])
    
    parts = axes[2, 1].violinplot(data_consistency, positions=positions, widths=0.7,
                                  showmeans=False, showmedians=False)
    hide_violin_whiskers(parts)
    
    for i, pc in enumerate(parts['bodies']):
        if i < n_configs:
            pc.set_facecolor('khaki')
        elif i == n_configs:
            pc.set_facecolor('green')
        else:
            pc.set_facecolor('red')
        pc.set_alpha(0.7)
    
    axes[2, 1].set_xticks(positions)
    axes[2, 1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[2, 1].set_ylabel('Consistency (pixels)', fontsize=10)
    axes[2, 1].set_title('Consistency (avg A+B)', fontweight='bold', fontsize=12)
    axes[2, 1].grid(axis='y', alpha=0.3)
    
    # === COL 2: Photometric ===
    data_photometric = []
    for result in results_full:
        photo_avg = (result['photometric_A'] + result['photometric_B']) / 2
        photo = photo_avg[valid_mask]
        photo = photo[~np.isnan(photo)]
        data_photometric.append(photo)
    
    # Add Oracle photometric
    photo_oracle = results_dict.get('photometric_oracle', np.zeros_like(u_true))
    oracle_photo = photo_oracle[valid_mask]
    data_photometric.append(oracle_photo[~np.isnan(oracle_photo)])
    
    # Add Ensemble photometric
    photo_ensemble = results_dict.get('photometric_ensemble', np.zeros_like(u_true))
    ensemble_photo = photo_ensemble[valid_mask]
    data_photometric.append(ensemble_photo[~np.isnan(ensemble_photo)])
    
    parts = axes[2, 2].violinplot(data_photometric, positions=positions, widths=0.7,
                                  showmeans=False, showmedians=False)
    hide_violin_whiskers(parts)
    
    for i, pc in enumerate(parts['bodies']):
        if i < n_configs:
            pc.set_facecolor('lightsalmon')
        elif i == n_configs:
            pc.set_facecolor('green')
        else:
            pc.set_facecolor('red')
        pc.set_alpha(0.7)
    
    axes[2, 2].set_xticks(positions)
    axes[2, 2].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[2, 2].set_ylabel('Photometric (intensity)', fontsize=10)
    axes[2, 2].set_title('Photometric (avg A+B)', fontweight='bold', fontsize=12)
    axes[2, 2].grid(axis='y', alpha=0.3)
    
    # Overall title
    plt.suptitle('Metric Distributions Across Configurations',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {output_path}")


if __name__ == "__main__":
    print("✨ Metric histogram visualization created!")
