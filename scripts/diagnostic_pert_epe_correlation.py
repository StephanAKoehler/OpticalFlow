# File: scripts/diagnostic_pert_epe_correlation.py
"""
Diagnostic: Where does perturbation ensemble beat photometric ensemble?

At each pixel, compare:
- EPE of photo-selected config
- EPE of pert-selected config

Find pixels where perturbation wins (lower EPE).
If many such pixels exist, combining could help.

Usage:
    python scripts/diagnostic_pert_epe_correlation.py data/.../results_full.pkl
"""

import numpy as np
import pickle
import sys
import re
from pathlib import Path
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt


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
        print(f"❌ Ground truth not found at {frames_dir}")
        sys.exit(1)
    
    u_data = np.load(u_path)
    v_data = np.load(v_path)
    u_truth = u_data[list(u_data.keys())[0]]
    v_truth = v_data[list(v_data.keys())[0]]
    
    valid_mask = (
        ~np.isnan(u_truth) & ~np.isnan(v_truth) &
        (np.abs(u_truth) < 1e8) & (np.abs(v_truth) < 1e8)
    )
    
    return u_truth, v_truth, valid_mask


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnostic_pert_epe_correlation.py <results_full.pkl>")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    
    print(f"📂 Loading {results_path}")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    n_configs = len(results)
    print(f"   {n_configs} configurations")
    
    # Load ground truth
    u_truth, v_truth, valid_mask = load_ground_truth(results_path)
    H, W = u_truth.shape
    n_valid = valid_mask.sum()
    print(f"   Shape: {H}×{W}, valid: {n_valid}")
    
    # Build stacks
    print("\n📊 Building data stacks...")
    
    u_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    v_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    epe_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    photo_log_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    pert_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    depths = []
    for i, r in enumerate(results):
        config_name = r['metadata'].get('config_name', '')
        match = re.search(r'win(\d+)', config_name)
        winsize = int(match.group(1)) if match else 15
        depth = winsize / 2 + 1.0
        depths.append(depth)
        
        u = r['flows']['u_AB']
        v = r['flows']['v_AB']
        
        u_stack[i] = u
        v_stack[i] = v
        epe_stack[i] = np.sqrt((u - u_truth)**2 + (v - v_truth)**2)
        photo_log_stack[i] = r['metrics']['photo_log_raw_A']
        pert_stack[i] = r['metrics']['perturbation_raw_A'] * depth  # scaled by depth
    
    # Per-pixel selection by each metric
    print("\n📊 Computing per-pixel selections...")
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    
    # Photometric selection
    photo_selected_idx = np.argmin(photo_log_stack, axis=0)
    epe_photo = epe_stack[photo_selected_idx, y_idx, x_idx]
    
    # Perturbation selection
    pert_selected_idx = np.argmin(pert_stack, axis=0)
    epe_pert = epe_stack[pert_selected_idx, y_idx, x_idx]
    
    # Oracle selection
    epe_oracle = np.min(epe_stack, axis=0)
    
    # Extract valid pixels
    epe_photo_valid = epe_photo[valid_mask]
    epe_pert_valid = epe_pert[valid_mask]
    epe_oracle_valid = epe_oracle[valid_mask]
    
    # Compare: where does pert beat photo?
    print("\n" + "=" * 70)
    print("PIXEL-WISE COMPARISON: PHOTOMETRIC vs PERTURBATION ENSEMBLE")
    print("=" * 70)
    
    pert_wins = epe_pert_valid < epe_photo_valid
    photo_wins = epe_photo_valid < epe_pert_valid
    tie = epe_pert_valid == epe_photo_valid
    
    n_pert_wins = pert_wins.sum()
    n_photo_wins = photo_wins.sum()
    n_tie = tie.sum()
    
    print(f"\n   Perturbation wins:  {n_pert_wins:>7} pixels ({n_pert_wins/n_valid*100:>5.1f}%)")
    print(f"   Photometric wins:   {n_photo_wins:>7} pixels ({n_photo_wins/n_valid*100:>5.1f}%)")
    print(f"   Tie:                {n_tie:>7} pixels ({n_tie/n_valid*100:>5.1f}%)")
    
    # Magnitude of wins
    print("\n   When perturbation wins:")
    if n_pert_wins > 0:
        pert_advantage = epe_photo_valid[pert_wins] - epe_pert_valid[pert_wins]
        print(f"      Mean advantage:   {pert_advantage.mean():.4f} px")
        print(f"      Median advantage: {np.median(pert_advantage):.4f} px")
        print(f"      Total saved:      {pert_advantage.sum():.1f} px")
    
    print("\n   When photometric wins:")
    if n_photo_wins > 0:
        photo_advantage = epe_pert_valid[photo_wins] - epe_photo_valid[photo_wins]
        print(f"      Mean advantage:   {photo_advantage.mean():.4f} px")
        print(f"      Median advantage: {np.median(photo_advantage):.4f} px")
        print(f"      Total saved:      {photo_advantage.sum():.1f} px")
    
    # Net effect
    total_epe_photo = epe_photo_valid.sum()
    total_epe_pert = epe_pert_valid.sum()
    print(f"\n   Net: photometric total EPE = {total_epe_photo:.1f}")
    print(f"        perturbation total EPE = {total_epe_pert:.1f}")
    print(f"        Δ = {total_epe_pert - total_epe_photo:+.1f} (positive = photo better)")
    
    # ======================================================================
    # LARGER DISPLACEMENT HEURISTIC
    # ======================================================================
    print("\n" + "=" * 70)
    print("LARGER DISPLACEMENT HEURISTIC")
    print("=" * 70)
    
    # Get flows from each selection method
    u_photo = u_stack[photo_selected_idx, y_idx, x_idx]
    v_photo = v_stack[photo_selected_idx, y_idx, x_idx]
    u_pert = u_stack[pert_selected_idx, y_idx, x_idx]
    v_pert = v_stack[pert_selected_idx, y_idx, x_idx]
    
    # Magnitude of each flow
    mag_photo = np.sqrt(u_photo**2 + v_photo**2)
    mag_pert = np.sqrt(u_pert**2 + v_pert**2)
    
    # Select larger displacement
    use_photo = mag_photo >= mag_pert
    u_larger = np.where(use_photo, u_photo, u_pert)
    v_larger = np.where(use_photo, v_photo, v_pert)
    
    epe_larger = np.sqrt((u_larger - u_truth)**2 + (v_larger - v_truth)**2)
    epe_larger_valid = epe_larger[valid_mask]
    
    # Select smaller displacement
    use_photo_smaller = mag_photo <= mag_pert
    u_smaller = np.where(use_photo_smaller, u_photo, u_pert)
    v_smaller = np.where(use_photo_smaller, v_photo, v_pert)
    
    epe_smaller = np.sqrt((u_smaller - u_truth)**2 + (v_smaller - v_truth)**2)
    epe_smaller_valid = epe_smaller[valid_mask]
    
    n_photo_larger = use_photo[valid_mask].sum()
    n_pert_larger = (~use_photo)[valid_mask].sum()
    
    print(f"\n   Photo has larger mag: {n_photo_larger} pixels ({n_photo_larger/n_valid*100:.1f}%)")
    print(f"   Pert has larger mag:  {n_pert_larger} pixels ({n_pert_larger/n_valid*100:.1f}%)")
    
    print(f"\n   EPE comparison:")
    print(f"      Photo ensemble:       {epe_photo_valid.mean():.4f}")
    print(f"      Pert ensemble:        {epe_pert_valid.mean():.4f}")
    print(f"      Larger displacement:  {epe_larger_valid.mean():.4f}")
    print(f"      Smaller displacement: {epe_smaller_valid.mean():.4f}")
    print(f"      Oracle:               {epe_oracle_valid.mean():.4f}")
    
    improvement_larger = (epe_photo_valid.mean() - epe_larger_valid.mean()) / epe_photo_valid.mean() * 100
    improvement_smaller = (epe_photo_valid.mean() - epe_smaller_valid.mean()) / epe_photo_valid.mean() * 100
    print(f"\n   vs photo:")
    print(f"      Larger:  {improvement_larger:+.1f}%")
    print(f"      Smaller: {improvement_smaller:+.1f}%")
    
    # ======================================================================
    # SMOOTHER FLOW HEURISTIC
    # ======================================================================
    print("\n" + "=" * 70)
    print("SMOOTHER FLOW HEURISTIC")
    print("=" * 70)
    
    # Compute local smoothness (Jacobian Frobenius norm) for each flow field
    def local_smoothness(u, v):
        """Compute per-pixel smoothness = sum of squared gradients."""
        du_dx = np.gradient(u, axis=1)
        du_dy = np.gradient(u, axis=0)
        dv_dx = np.gradient(v, axis=1)
        dv_dy = np.gradient(v, axis=0)
        return du_dx**2 + du_dy**2 + dv_dx**2 + dv_dy**2
    
    smooth_photo = local_smoothness(u_photo, v_photo)
    smooth_pert = local_smoothness(u_pert, v_pert)
    
    # Pick smoother flow (lower gradient = smoother)
    use_photo_smooth = smooth_photo <= smooth_pert
    u_smoother = np.where(use_photo_smooth, u_photo, u_pert)
    v_smoother = np.where(use_photo_smooth, v_photo, v_pert)
    
    epe_smoother = np.sqrt((u_smoother - u_truth)**2 + (v_smoother - v_truth)**2)
    epe_smoother_valid = epe_smoother[valid_mask]
    
    n_photo_smoother = use_photo_smooth[valid_mask].sum()
    n_pert_smoother = (~use_photo_smooth)[valid_mask].sum()
    
    print(f"\n   Photo is smoother: {n_photo_smoother} pixels ({n_photo_smoother/n_valid*100:.1f}%)")
    print(f"   Pert is smoother:  {n_pert_smoother} pixels ({n_pert_smoother/n_valid*100:.1f}%)")
    
    print(f"\n   EPE comparison:")
    print(f"      Photo ensemble:       {epe_photo_valid.mean():.4f}")
    print(f"      Smoother flow:        {epe_smoother_valid.mean():.4f}")
    print(f"      Oracle:               {epe_oracle_valid.mean():.4f}")
    
    improvement_smoother = (epe_photo_valid.mean() - epe_smoother_valid.mean()) / epe_photo_valid.mean() * 100
    print(f"\n   vs photo: {improvement_smoother:+.1f}%")
    
    # ======================================================================
    # CLOSEST TO SMOOTHED PHOTO HEURISTIC
    # ======================================================================
    print("\n" + "=" * 70)
    print("CLOSEST TO SMOOTHED PHOTO HEURISTIC")
    print("=" * 70)
    
    from scipy.ndimage import uniform_filter, gaussian_filter
    
    # Test multiple kernel sizes with uniform filter
    kernel_sizes = [3, 5, 9, 15, 21]
    
    print(f"\n   UNIFORM FILTER:")
    print(f"   {'Kernel':<8} | {'Photo closer':<15} | {'Pert closer':<15} | {'EPE':>10} | {'vs photo':>10}")
    print("-" * 70)
    
    for ksize in kernel_sizes:
        # Smooth the photo flow
        u_smooth = uniform_filter(u_photo, size=ksize, mode='nearest')
        v_smooth = uniform_filter(v_photo, size=ksize, mode='nearest')
        
        # Distance from each flow to smoothed reference
        dist_photo = np.sqrt((u_photo - u_smooth)**2 + (v_photo - v_smooth)**2)
        dist_pert = np.sqrt((u_pert - u_smooth)**2 + (v_pert - v_smooth)**2)
        
        # Pick closer to smooth
        use_photo_close = dist_photo <= dist_pert
        u_closest = np.where(use_photo_close, u_photo, u_pert)
        v_closest = np.where(use_photo_close, v_photo, v_pert)
        
        epe_closest = np.sqrt((u_closest - u_truth)**2 + (v_closest - v_truth)**2)
        epe_closest_valid = epe_closest[valid_mask]
        
        n_photo_closer = use_photo_close[valid_mask].sum()
        n_pert_closer = (~use_photo_close)[valid_mask].sum()
        
        improvement = (epe_photo_valid.mean() - epe_closest_valid.mean()) / epe_photo_valid.mean() * 100
        
        print(f"   {ksize:<8} | {n_photo_closer:>6} ({n_photo_closer/n_valid*100:>5.1f}%) | "
              f"{n_pert_closer:>6} ({n_pert_closer/n_valid*100:>5.1f}%) | "
              f"{epe_closest_valid.mean():>10.4f} | {improvement:>+9.1f}%")
    
    # Test Gaussian filter (sigma values roughly equivalent to uniform kernel sizes)
    sigma_values = [1.0, 1.5, 2.5, 4.0, 6.0]
    
    print(f"\n   GAUSSIAN FILTER:")
    print(f"   {'Sigma':<8} | {'Photo closer':<15} | {'Pert closer':<15} | {'EPE':>10} | {'vs photo':>10}")
    print("-" * 70)
    
    for sigma in sigma_values:
        # Smooth the photo flow
        u_smooth = gaussian_filter(u_photo, sigma=sigma, mode='nearest')
        v_smooth = gaussian_filter(v_photo, sigma=sigma, mode='nearest')
        
        # Distance from each flow to smoothed reference
        dist_photo = np.sqrt((u_photo - u_smooth)**2 + (v_photo - v_smooth)**2)
        dist_pert = np.sqrt((u_pert - u_smooth)**2 + (v_pert - v_smooth)**2)
        
        # Pick closer to smooth
        use_photo_close = dist_photo <= dist_pert
        u_closest = np.where(use_photo_close, u_photo, u_pert)
        v_closest = np.where(use_photo_close, v_photo, v_pert)
        
        epe_closest = np.sqrt((u_closest - u_truth)**2 + (v_closest - v_truth)**2)
        epe_closest_valid = epe_closest[valid_mask]
        
        n_photo_closer = use_photo_close[valid_mask].sum()
        n_pert_closer = (~use_photo_close)[valid_mask].sum()
        
        improvement = (epe_photo_valid.mean() - epe_closest_valid.mean()) / epe_photo_valid.mean() * 100
        
        print(f"   {sigma:<8} | {n_photo_closer:>6} ({n_photo_closer/n_valid*100:>5.1f}%) | "
              f"{n_pert_closer:>6} ({n_pert_closer/n_valid*100:>5.1f}%) | "
              f"{epe_closest_valid.mean():>10.4f} | {improvement:>+9.1f}%")
    
    # How close is each to oracle?
    print("\n   Oracle comparison:")
    gap_photo = (epe_photo_valid - epe_oracle_valid).sum()
    gap_pert = (epe_pert_valid - epe_oracle_valid).sum()
    print(f"      Photo gap to oracle:  {gap_photo:.1f}")
    print(f"      Pert gap to oracle:   {gap_pert:.1f}")
    
    # Can we do better by combining?
    print("\n   Potential of combination (per-pixel min):")
    epe_combined = np.minimum(epe_photo_valid, epe_pert_valid)
    total_combined = epe_combined.sum()
    print(f"      Combined total EPE: {total_combined:.1f}")
    print(f"      vs photo: {total_combined - total_epe_photo:+.1f} ({(total_combined - total_epe_photo)/total_epe_photo*100:+.1f}%)")
    
    # ======================================================================
    # FLOW DISAGREEMENT ANALYSIS
    # ======================================================================
    print("\n" + "=" * 70)
    print("FLOW DISAGREEMENT ANALYSIS")
    print("=" * 70)
    
    # Get flows from each selection method
    u_photo = u_stack[photo_selected_idx, y_idx, x_idx]
    v_photo = v_stack[photo_selected_idx, y_idx, x_idx]
    u_pert = u_stack[pert_selected_idx, y_idx, x_idx]
    v_pert = v_stack[pert_selected_idx, y_idx, x_idx]
    
    # Disagreement = Euclidean distance between flow vectors
    disagreement = np.sqrt((u_photo - u_pert)**2 + (v_photo - v_pert)**2)
    
    disagreement_valid = disagreement[valid_mask]
    
    print(f"\n   Disagreement stats:")
    print(f"      Mean:   {disagreement_valid.mean():.4f} px")
    print(f"      Median: {np.median(disagreement_valid):.4f} px")
    print(f"      Std:    {disagreement_valid.std():.4f} px")
    print(f"      p90:    {np.percentile(disagreement_valid, 90):.4f} px")
    print(f"      p99:    {np.percentile(disagreement_valid, 99):.4f} px")
    
    # Correlation: disagreement vs EPE (of photo ensemble)
    rho_photo, p_photo = scipy_stats.spearmanr(disagreement_valid, epe_photo_valid)
    rho_oracle, p_oracle = scipy_stats.spearmanr(disagreement_valid, epe_oracle_valid)
    
    print(f"\n   Correlation (disagreement vs EPE):")
    print(f"      vs photo EPE:  ρ = {rho_photo:+.4f} (p={p_photo:.2e})")
    print(f"      vs oracle EPE: ρ = {rho_oracle:+.4f} (p={p_oracle:.2e})")
    
    # Binned analysis: EPE by disagreement quintiles
    print(f"\n   EPE by disagreement quintile:")
    percentiles = [0, 20, 40, 60, 80, 100]
    disagree_bins = np.percentile(disagreement_valid, percentiles)
    
    print(f"   {'Quintile':<10} | {'Disagree range':<18} | {'Mean EPE':>10} | {'Pert wins':>10}")
    print("-" * 60)
    
    for i in range(len(percentiles) - 1):
        low, high = disagree_bins[i], disagree_bins[i + 1]
        if i < len(percentiles) - 2:
            mask = (disagreement_valid >= low) & (disagreement_valid < high)
        else:
            mask = (disagreement_valid >= low) & (disagreement_valid <= high)
        
        epe_bin = epe_photo_valid[mask]
        pert_wins_bin = (epe_pert_valid[mask] < epe_photo_valid[mask]).mean() * 100
        
        bin_name = f"Q{i+1}"
        print(f"   {bin_name:<10} | {low:>7.4f} - {high:<7.4f} | {epe_bin.mean():>10.4f} | {pert_wins_bin:>9.1f}%")
    
    # Key question: when disagreement is HIGH, does pert win more often?
    high_disagree_mask = disagreement_valid > np.percentile(disagreement_valid, 80)
    low_disagree_mask = disagreement_valid < np.percentile(disagreement_valid, 20)
    
    pert_wins_high = (epe_pert_valid[high_disagree_mask] < epe_photo_valid[high_disagree_mask]).mean() * 100
    pert_wins_low = (epe_pert_valid[low_disagree_mask] < epe_photo_valid[low_disagree_mask]).mean() * 100
    
    print(f"\n   Pert win rate by disagreement level:")
    print(f"      Low disagreement (Q1):   {pert_wins_low:.1f}%")
    print(f"      High disagreement (Q5):  {pert_wins_high:.1f}%")
    
    # Zero disagreement = same config selected
    same_config = (photo_selected_idx == pert_selected_idx)
    n_same = same_config[valid_mask].sum()
    print(f"\n   Same config selected: {n_same} pixels ({n_same/n_valid*100:.1f}%)")
    if n_same > 0:
        epe_same = epe_photo_valid[same_config[valid_mask]]
        epe_diff = epe_photo_valid[~same_config[valid_mask]]
        print(f"      EPE when same config:  mean={epe_same.mean():.4f}")
        print(f"      EPE when diff config:  mean={epe_diff.mean():.4f}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("📈 DISAGREEMENT INTERPRETATION:")
    print("=" * 70)
    
    if rho_photo > 0.1:
        print(f"   ✓ Positive correlation (ρ={rho_photo:+.3f}): disagreement flags high-EPE regions")
        print("   → Disagreement is a useful GT-free confidence signal!")
    else:
        print(f"   ~ Weak/no correlation (ρ={rho_photo:+.3f}): disagreement not predictive of EPE")
    
    if pert_wins_high > pert_wins_low + 5:
        print(f"   ✓ Pert wins more when disagreement is high ({pert_wins_high:.1f}% vs {pert_wins_low:.1f}%)")
        print("   → In conflict regions, perturbation may have useful info")
    elif pert_wins_low > pert_wins_high + 5:
        print(f"   ✗ Pert wins LESS when disagreement is high ({pert_wins_high:.1f}% vs {pert_wins_low:.1f}%)")
        print("   → Photo is especially reliable in conflict regions")
    else:
        print(f"   ~ Pert win rate similar regardless of disagreement")

    # Update visualization to include disagreement
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: EPE maps
    ax = axes[0, 0]
    im = ax.imshow(epe_photo, cmap='hot', vmin=0, vmax=np.percentile(epe_photo[valid_mask], 95))
    ax.set_title(f'EPE: Photometric ensemble\nmean={epe_photo_valid.mean():.3f}')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 1]
    im = ax.imshow(epe_pert, cmap='hot', vmin=0, vmax=np.percentile(epe_photo[valid_mask], 95))
    ax.set_title(f'EPE: Perturbation ensemble\nmean={epe_pert_valid.mean():.3f}')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 2]
    im = ax.imshow(epe_oracle, cmap='hot', vmin=0, vmax=np.percentile(epe_photo[valid_mask], 95))
    ax.set_title(f'EPE: Oracle\nmean={epe_oracle_valid.mean():.3f}')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    # Disagreement map
    ax = axes[0, 3]
    disagree_display = disagreement.copy()
    disagree_display[~valid_mask] = np.nan
    im = ax.imshow(disagree_display, cmap='viridis', vmin=0, vmax=np.percentile(disagreement_valid, 95))
    ax.set_title(f'Flow disagreement\nmean={disagreement_valid.mean():.3f} px')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    # Row 2: Winner map, diff map, and jointplot-style heatmap
    ax = axes[1, 0]
    winner_map = np.zeros((H, W), dtype=np.float32)
    winner_map[~valid_mask] = np.nan
    pert_wins_2d = epe_pert < epe_photo
    photo_wins_2d = epe_photo < epe_pert
    winner_map[pert_wins_2d & valid_mask] = 1  # green
    winner_map[photo_wins_2d & valid_mask] = 2  # blue
    winner_map[(~pert_wins_2d) & (~photo_wins_2d) & valid_mask] = 0  # tie
    
    colors = plt.cm.colors.ListedColormap(['gray', 'green', 'blue'])
    im = ax.imshow(winner_map, cmap=colors, vmin=0, vmax=2)
    ax.set_title(f'Winner map\nGreen=pert ({n_pert_wins/n_valid*100:.1f}%), Blue=photo ({n_photo_wins/n_valid*100:.1f}%)')
    ax.axis('off')
    
    ax = axes[1, 1]
    diff_map = epe_photo - epe_pert  # positive = pert is better
    diff_map[~valid_mask] = np.nan
    vmax = np.percentile(np.abs(diff_map[valid_mask]), 95)
    im = ax.imshow(diff_map, cmap='RdBu', vmin=-vmax, vmax=vmax)
    ax.set_title('EPE(photo) - EPE(pert)\nRed=pert better, Blue=photo better')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    # Jointplot-style: 2D heatmap with marginal histograms
    # Merge the two rightmost subplots into one larger area
    axes[1, 2].remove()
    axes[1, 3].remove()
    
    # Create new axes manually for jointplot
    # Position: right half of bottom row
    left = 0.56
    bottom = 0.08
    width = 0.38
    height = 0.38
    
    # Main heatmap
    ax_main = fig.add_axes([left, bottom, width - 0.06, height - 0.06])
    # Top marginal
    ax_top = fig.add_axes([left, bottom + height - 0.04, width - 0.06, 0.06])
    # Right marginal  
    ax_right = fig.add_axes([left + width - 0.04, bottom, 0.04, height - 0.06])
    
    # Clip data for visualization
    disagree_clip = np.clip(disagreement_valid, 0, np.percentile(disagreement_valid, 99))
    epe_clip = np.clip(epe_photo_valid, 0, np.percentile(epe_photo_valid, 99))
    
    # 2D histogram (log scale for visibility, fewer bins for clarity)
    h = ax_main.hist2d(disagree_clip, epe_clip, bins=25, cmap='viridis', 
                       norm=plt.matplotlib.colors.LogNorm(vmin=1))
    ax_main.set_xlabel('Flow disagreement (px)')
    ax_main.set_ylabel('EPE of photo ensemble')
    
    # Top marginal histogram (log scale)
    ax_top.hist(disagree_clip, bins=25, color='steelblue', alpha=0.7)
    ax_top.set_xlim(ax_main.get_xlim())
    ax_top.set_xticks([])
    ax_top.set_yscale('log')
    ax_top.set_ylabel('Count')
    ax_top.set_title(f'Disagreement vs EPE\nρ={rho_photo:.3f}')
    
    # Right marginal histogram (log scale)
    ax_right.hist(epe_clip, bins=25, orientation='horizontal', color='steelblue', alpha=0.7)
    ax_right.set_ylim(ax_main.get_ylim())
    ax_right.set_yticks([])
    ax_right.set_xscale('log')
    ax_right.set_xlabel('Count')
    
    plt.tight_layout()
    
    output_path = results_path.parent / 'diagnostic_photo_vs_pert.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved figure to {output_path}")


if __name__ == "__main__":
    main()
