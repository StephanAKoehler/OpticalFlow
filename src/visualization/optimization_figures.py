# File: src/visualization/optimization_figures.py
"""
Optimization-specific figure generation.

Creates plots for analyzing Optuna optimization results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import optuna


def generate_optimization_history(study: optuna.Study, 
                                  oracle_epe: float,
                                  output_path: Path):
    """
    Generate optimization history plot showing EPE over trials.
    
    Shows:
    - All trial EPEs (scatter)
    - Best-so-far trajectory (line)
    - Oracle EPE reference (horizontal line)
    - Best weights in legend
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Get complete trials
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if not complete_trials:
        ax.text(0.5, 0.5, 'No complete trials', ha='center', va='center')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    trial_numbers = [t.number for t in complete_trials]
    trial_values = [t.value for t in complete_trials]
    
    # Compute best-so-far
    best_so_far = []
    current_best = float('inf')
    for val in trial_values:
        current_best = min(current_best, val)
        best_so_far.append(current_best)
    
    # Plot all trials
    ax.scatter(trial_numbers, trial_values, alpha=0.5, s=30, 
              label='Trial EPE', color='lightblue', edgecolors='blue', linewidths=0.5)
    
    # Plot best-so-far trajectory
    ax.plot(trial_numbers, best_so_far, 'g-', linewidth=2, 
           label='Best so far', zorder=10)
    
    # Plot oracle reference
    ax.axhline(oracle_epe, color='red', linestyle='--', linewidth=2,
              label=f'Oracle EPE ({oracle_epe:.4f})', zorder=5)
    
    # Final best marker
    best_idx = trial_values.index(min(trial_values))
    ax.scatter([trial_numbers[best_idx]], [trial_values[best_idx]], 
              s=200, color='gold', marker='*', edgecolors='black', linewidths=2,
              label=f'Best trial (#{trial_numbers[best_idx]})', zorder=15)
    
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('EPE (pixels)', fontsize=12)
    ax.set_title('Optimization History', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Create legend with best weights
    best_trial = complete_trials[best_idx]
    best_weights = best_trial.params
    
    # Build legend text with weights
    weight_keys = ['traction_A', 'traction_B', 'consistency_A', 'consistency_B',
                   'photometric_A', 'photometric_B', 'displacements_N2S_A2B', 'displacements_N2S_B2A']
    
    legend_elements = ax.get_legend_handles_labels()
    
    # Add weight info as text box instead of in legend
    gap = min(trial_values) - oracle_epe
    gap_pct = 100 * (min(trial_values) / oracle_epe - 1)
    
    stats_text = f'Best Weights (Trial #{trial_numbers[best_idx]}):\n'
    stats_text += '-' * 35 + '\n'
    for key in weight_keys:
        val = best_weights.get(key, 0.0)
        if val > 0.001:  # Only show non-zero weights
            stats_text += f'{key:20s}: {val:.3f}\n'
    stats_text += '-' * 35 + '\n'
    stats_text += f'Best EPE:  {min(trial_values):.6f}\n'
    stats_text += f'Oracle:    {oracle_epe:.6f}\n'
    stats_text += f'Gap:       {gap:.6f} ({gap_pct:+.1f}%)\n'
    stats_text += f'Trials:    {len(complete_trials)}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
           fontsize=9, family='monospace')
    
    # Simple legend without weights
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_parameter_importance(study: optuna.Study,
                                  output_path: Path):
    """
    Generate parameter importance plot.
    
    Shows which weights have the biggest impact on EPE.
    Returns True if plot generated, False if skipped.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        # Calculate parameter importances
        importance = optuna.importance.get_param_importances(study)
        
        if not importance:
            ax.text(0.5, 0.5, 'Insufficient trials for importance analysis', 
                   ha='center', va='center')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return False
        
        # Skip if only 1-2 parameters (likely due to simplex constraint)
        if len(importance) < 3:
            print(f"      ⚠️  Only {len(importance)} parameters with measurable importance")
            print(f"         (Simplex constraint limits parameter variation)")
            ax.text(0.5, 0.5, 
                   f'Insufficient parameter variation\n'
                   f'Only {len(importance)} of 8 weights varied enough to measure importance\n'
                   f'(See weight_evolution.png for full search trajectory)',
                   ha='center', va='center', fontsize=11)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return False
        
        # Sort by importance
        params = list(importance.keys())
        values = [importance[p] for p in params]
        
        # Sort
        sorted_indices = np.argsort(values)
        params = [params[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(params))
        bars = ax.barh(y_pos, values, color='steelblue', edgecolor='navy', linewidth=1)
        
        # Color bars by importance
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Parameter Importance (Normalized)', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return True
        
    except Exception as e:
        # If importance calculation fails (not enough trials, etc.)
        ax.text(0.5, 0.5, f'Could not compute importances:\n{str(e)}', 
               ha='center', va='center')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return False


def generate_weight_evolution(study: optuna.Study,
                              output_path: Path):
    """
    Generate weight evolution plot showing how weights changed over trials.
    
    Shows:
    - All 8 weight trajectories
    - Best trial marker on each line
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get complete trials
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if not complete_trials:
        ax.text(0.5, 0.5, 'No complete trials', ha='center', va='center')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    # Extract weight trajectories
    weight_keys = ['traction_A', 'traction_B', 'consistency_A', 'consistency_B',
                   'photometric_A', 'photometric_B', 'displacements_N2S_A2B', 'displacements_N2S_B2A']
    
    trial_numbers = [t.number for t in complete_trials]
    weight_trajectories = {key: [] for key in weight_keys}
    
    for trial in complete_trials:
        for key in weight_keys:
            weight_trajectories[key].append(trial.params.get(key, 0.0))
    
    # Find best trial
    best_trial_idx = min(range(len(complete_trials)), 
                        key=lambda i: complete_trials[i].value)
    best_trial_number = complete_trials[best_trial_idx].number
    
    # Plot each weight trajectory
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    
    for idx, (key, color) in enumerate(zip(weight_keys, colors)):
        values = weight_trajectories[key]
        
        # Plot trajectory
        ax.plot(trial_numbers, values, label=key, color=color, 
               linewidth=1.5, alpha=0.7)
        
        # Mark best trial on this line
        best_value = values[best_trial_idx]
        ax.scatter([best_trial_number], [best_value], 
                  s=150, color=color, marker='*', 
                  edgecolors='black', linewidths=2, zorder=10)
    
    # Add vertical line at best trial
    ax.axvline(best_trial_number, color='red', linestyle='--', 
              linewidth=2, alpha=0.5, label=f'Best trial (#{best_trial_number})')
    
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Weight Value', fontsize=12)
    ax.set_title('Weight Evolution Over Trials', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add best weights text box
    best_weights = complete_trials[best_trial_idx].params
    weights_text = 'Best weights:\n'
    for key in weight_keys:
        val = best_weights.get(key, 0.0)
        weights_text += f'{key}: {val:.3f}\n'
    
    ax.text(0.02, 0.98, weights_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=9, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_all_optimization_figures(study: optuna.Study,
                                      oracle_epe: float,
                                      figures_dir: Path):
    """
    Generate all optimization-specific figures.
    
    Args:
        study: Optuna study object
        oracle_epe: Oracle EPE for reference
        figures_dir: Directory to save figures
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("   📊 Generating optimization figures...")
    
    # Optimization history (with best weights in text box)
    generate_optimization_history(study, oracle_epe, figures_dir / "optimization_history.png")
    print("      ✅ Optimization history")
    
    # Weight evolution
    generate_weight_evolution(study, figures_dir / "weight_evolution.png")
    print("      ✅ Weight evolution")


if __name__ == "__main__":
    print("✅ Optimization figures module loaded")
