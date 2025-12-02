# File: scripts/visualize_diagnostic.py
"""
Visualize diagnostic information for optical flow analysis.

Layout (4x3):
┌─────────────┬─────────┬─────────┐
│ FrameA+Flow │ Mag+Dir │Traction │
├─────────────┼─────────┼─────────┤
│  Perturb    │ Consist │Photomet │
├─────────────┼─────────┼─────────┤
│  Spread     │Dominant │ Config  │
├─────────────┼─────────┼─────────┤
│ Salience    │  Cost   │   EPE   │
└─────────────┴─────────┴─────────┘
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ensemble.selection import select_ensemble, validate_selection_config, compute_spread_salience
from src.core.data_loader import load_movie_sequence


def gather_metric(results_full: list, selection: np.ndarray, metric_key: str) -> np.ndarray:
    """
    Gather metric values from winning config at each pixel.
    
    Args:
        results_full: list of config results with metrics
        selection: (H, W) config index per pixel
        metric_key: key into results['metrics']
        
    Returns:
        gathered: (H, W) metric values from selected configs
    """
    n_configs = len(results_full)
    H, W = selection.shape
    
    # Stack metrics from all configs
    metric_stack = np.stack([r['metrics'][metric_key] for r in results_full], axis=0)
    
    # Gather using advanced indexing
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    gathered = metric_stack[selection, ii, jj]
    
    return gathered


def flow_to_middlebury(u: np.ndarray, v: np.ndarray, 
                        max_mag: float = None) -> np.ndarray:
    """
    Convert optical flow to Middlebury colorwheel visualization.
    
    Middlebury convention:
    - Hue = direction (0°=red=right, 90°=green=down, etc.)
    - Saturation = magnitude (normalized)
    - Value = 1 (full brightness)
    
    Args:
        u, v: (H, W) flow components
        max_mag: normalization factor (if None, uses max in field)
        
    Returns:
        rgb: (H, W, 3) RGB image in [0, 1]
    """
    import colorsys
    
    mag = np.sqrt(u**2 + v**2)
    if max_mag is None:
        max_mag = np.nanmax(mag)
    if max_mag < 1e-6:
        max_mag = 1.0
    
    # Direction: atan2 gives angle in radians, convert to [0, 1] for hue
    angle = np.arctan2(-v, -u)  # negative to match Middlebury convention
    hue = (angle + np.pi) / (2 * np.pi)  # [0, 1]
    
    # Saturation = normalized magnitude
    sat = np.clip(mag / max_mag, 0, 1)
    
    # Value = 1
    val = np.ones_like(mag)
    
    # Handle NaN
    nan_mask = np.isnan(u) | np.isnan(v)
    hue[nan_mask] = 0
    sat[nan_mask] = 0
    val[nan_mask] = 0.5  # gray for invalid
    
    # Convert HSV to RGB
    H, W = u.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    
    for i in range(H):
        for j in range(W):
            rgb[i, j] = colorsys.hsv_to_rgb(hue[i, j], sat[i, j], val[i, j])
    
    return rgb


def flow_to_middlebury_vectorized(u: np.ndarray, v: np.ndarray,
                                   max_mag: float = None) -> np.ndarray:
    """
    Vectorized Middlebury colorwheel visualization.
    """
    # Handle NaN first
    nan_mask = np.isnan(u) | np.isnan(v)
    u_clean = np.where(nan_mask, 0, u)
    v_clean = np.where(nan_mask, 0, v)
    
    mag = np.sqrt(u_clean**2 + v_clean**2)
    if max_mag is None:
        max_mag = np.max(mag)
    if max_mag < 1e-6:
        max_mag = 1.0
    
    # Direction
    angle = np.arctan2(-v_clean, -u_clean)
    hue = (angle + np.pi) / (2 * np.pi)  # [0, 1]
    
    # Saturation = normalized magnitude
    sat = np.clip(mag / max_mag, 0, 1)
    
    # HSV to RGB (vectorized)
    # Based on standard HSV conversion formula
    h6 = hue * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    
    p = 1.0 - sat  # v * (1 - s) where v=1
    q = 1.0 - sat * f  # v * (1 - s*f)
    t = 1.0 - sat * (1.0 - f)  # v * (1 - s*(1-f))
    v_val = np.ones_like(sat)  # value = 1
    
    # Build RGB based on hue sector
    rgb = np.zeros((*u.shape, 3), dtype=np.float32)
    
    mask0 = (i == 0)
    mask1 = (i == 1)
    mask2 = (i == 2)
    mask3 = (i == 3)
    mask4 = (i == 4)
    mask5 = (i == 5)
    
    rgb[mask0] = np.stack([v_val[mask0], t[mask0], p[mask0]], axis=-1)
    rgb[mask1] = np.stack([q[mask1], v_val[mask1], p[mask1]], axis=-1)
    rgb[mask2] = np.stack([p[mask2], v_val[mask2], t[mask2]], axis=-1)
    rgb[mask3] = np.stack([p[mask3], q[mask3], v_val[mask3]], axis=-1)
    rgb[mask4] = np.stack([t[mask4], p[mask4], v_val[mask4]], axis=-1)
    rgb[mask5] = np.stack([v_val[mask5], p[mask5], q[mask5]], axis=-1)
    
    # Gray for invalid
    rgb[nan_mask] = [0.5, 0.5, 0.5]
    
    return rgb


def draw_colorwheel(ax, max_mag: float = 1.0, n_segments: int = 64):
    """
    Draw Middlebury colorwheel legend.
    
    Args:
        ax: matplotlib axes (should be square)
        max_mag: maximum magnitude for label
        n_segments: number of segments in wheel
    """
    import colorsys
    
    # Create wedges for colorwheel
    patches = []
    colors = []
    
    for i in range(n_segments):
        theta1 = i * 360 / n_segments
        theta2 = (i + 1) * 360 / n_segments
        
        wedge = Wedge((0, 0), 1, theta1, theta2, width=0.4)
        patches.append(wedge)
        
        # Hue from angle (Middlebury: 0° = right = red)
        angle_rad = np.radians((theta1 + theta2) / 2)
        hue = (angle_rad + np.pi) / (2 * np.pi)
        hue = hue % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append(rgb)
    
    collection = PatchCollection(patches, facecolors=colors, edgecolors='none')
    ax.add_collection(collection)
    
    # Set limits and remove axes
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add magnitude label
    ax.text(0, 0, f'{max_mag:.1f}', ha='center', va='center', fontsize=8)
    
    # Add direction labels
    ax.text(1.15, 0, '→', ha='center', va='center', fontsize=10)
    ax.text(-1.15, 0, '←', ha='center', va='center', fontsize=10)
    ax.text(0, 1.15, '↑', ha='center', va='center', fontsize=10)
    ax.text(0, -1.15, '↓', ha='center', va='center', fontsize=10)


def visualize_diagnostic(frame_a: np.ndarray,
                         u: np.ndarray, v: np.ndarray,
                         traction: np.ndarray,
                         perturbation: np.ndarray,
                         consistency: np.ndarray,
                         photometric: np.ndarray,
                         spread: np.ndarray,
                         salience: np.ndarray,
                         cost: np.ndarray,
                         winner_scale: np.ndarray,
                         u_gt: np.ndarray = None,
                         v_gt: np.ndarray = None,
                         output_path: Path = None,
                         title: str = None):
    """
    Create 4x3 diagnostic visualization of flow and quality metrics.
    
    Layout:
    ┌─────────────┬─────────┬─────────┐
    │ FrameA+Flow │ Mag+Dir │Traction │
    ├─────────────┼─────────┼─────────┤
    │  Perturb    │ Consist │Photomet │
    ├─────────────┼─────────┼─────────┤
    │  Spread     │Dominant │ Config  │
    ├─────────────┼─────────┼─────────┤
    │ Salience    │  Cost   │   EPE   │
    └─────────────┴─────────┴─────────┘
    
    Args:
        frame_a: (H, W) or (H, W, 3) image
        u, v: (H, W) flow components
        traction: (H, W) traction metric
        perturbation: (H, W) perturbation sensitivity
        consistency: (H, W) forward/backward consistency
        photometric: (H, W) photometric error
        spread: (H, W) spread magnitude
        salience: (H, W) spread salience (chi-squared)
        cost: (H, W) selection cost (winning config's cost)
        winner_scale: (H, W) scale value of winning config
        u_gt, v_gt: (H, W) ground truth flow (optional, for EPE)
        output_path: where to save figure
        title: optional title
    """
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import Patch
    
    # Blue to red colormap for metrics
    blue_red = LinearSegmentedColormap.from_list('blue_red', ['blue', 'red'])
    
    # Categorical colormap for dominant metric
    # 0=perturbation (blue), 1=consistency (green), 2=photometric (red)
    dominant_cmap = ListedColormap(['blue', 'green', 'red'])
    
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    
    H, W = u.shape
    
    # =========================================================================
    # Row 0: Frame+Flow, Flow, Traction
    # =========================================================================
    
    # Frame A with flow vectors overlay
    ax = axes[0, 0]
    if frame_a.ndim == 3:
        ax.imshow(frame_a)
    else:
        ax.imshow(frame_a, cmap='gray')
    
    # Subsample for quiver (every 24 pixels for cleaner look)
    step = 24
    y_q, x_q = np.mgrid[step//2:H:step, step//2:W:step]
    u_q = u[step//2::step, step//2::step]
    v_q = v[step//2::step, step//2::step]
    
    # Quiver with proper scaling
    # angles='xy' ensures arrows point in data coordinates
    # scale_units='xy' makes scale consistent with data units
    # Negate v because image y-axis is inverted (down is positive)
    mag = np.sqrt(u_q**2 + v_q**2)
    max_mag_q = np.nanmax(mag)
    if max_mag_q < 0.01:
        max_mag_q = 1.0  # avoid division issues for zero flow
    ax.quiver(x_q, y_q, u_q, -v_q, 
              color='yellow', 
              angles='xy', 
              scale_units='xy',
              scale=max_mag_q/12,  # arrow length ~12 pixels at max magnitude
              width=0.004,
              headwidth=4,
              headlength=5)
    ax.set_title('Frame A + Flow')
    ax.axis('off')
    
    # Magnitude with normalized direction arrows
    ax = axes[0, 1]
    flow_mag = np.sqrt(u**2 + v**2)
    max_mag = np.nanmax(flow_mag)
    im = ax.imshow(flow_mag, cmap='viridis', vmin=0, vmax=max_mag)
    ax.set_title(f'Mag + Dir ({max_mag:.2f})')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Normalized direction arrows
    step = 24
    y_q, x_q = np.mgrid[step//2:H:step, step//2:W:step]
    u_q = u[step//2::step, step//2::step]
    v_q = v[step//2::step, step//2::step]
    mag_q = np.sqrt(u_q**2 + v_q**2)
    # Normalize to unit vectors (avoid div by zero)
    mag_q_safe = np.where(mag_q > 0.01, mag_q, 1.0)
    u_norm = u_q / mag_q_safe
    v_norm = v_q / mag_q_safe
    # Mask out near-zero flow
    u_norm[mag_q < 0.01] = 0
    v_norm[mag_q < 0.01] = 0
    ax.quiver(x_q, y_q, u_norm, -v_norm,
              color='white',
              angles='xy',
              scale_units='xy',
              scale=0.15,  # fixed scale for unit vectors
              width=0.004,
              headwidth=4,
              headlength=5)
    
    # Traction
    ax = axes[0, 2]
    vmax = np.nanmax(traction)
    im = ax.imshow(traction, cmap=blue_red, vmin=0, vmax=vmax)
    ax.set_title(f'Traction ({vmax:.2f})')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # =========================================================================
    # Row 1: Perturbation, Consistency, Photometric
    # =========================================================================
    
    ax = axes[1, 0]
    vmax = np.nanmax(perturbation)
    im = ax.imshow(perturbation, cmap=blue_red, vmin=0, vmax=vmax)
    ax.set_title(f'Perturbation ({vmax:.2f})')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax = axes[1, 1]
    vmax = np.nanmax(consistency)
    im = ax.imshow(consistency, cmap=blue_red, vmin=0, vmax=vmax)
    ax.set_title(f'Consistency ({vmax:.2f})')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax = axes[1, 2]
    vmax = np.nanmax(photometric)
    im = ax.imshow(photometric, cmap=blue_red, vmin=0, vmax=vmax)
    ax.set_title(f'Photometric ({vmax:.2f})')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # =========================================================================
    # Row 2: Spread, Dominant, Config
    # =========================================================================
    
    # Spread
    ax = axes[2, 0]
    vmax = np.nanmax(spread)
    im = ax.imshow(spread, cmap=blue_red, vmin=0, vmax=vmax)
    ax.set_title(f'Spread ({vmax:.2f})')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Dominant metric (categorical)
    ax = axes[2, 1]
    stacked = np.stack([perturbation, consistency, photometric], axis=0)
    dominant = np.argmax(stacked, axis=0).astype(float)
    dominant[np.isnan(perturbation)] = np.nan  # preserve NaN mask
    im = ax.imshow(dominant, cmap=dominant_cmap, vmin=-0.5, vmax=2.5)
    ax.set_title('Dominant Metric')
    ax.axis('off')
    
    # Legend for dominant
    legend_elements = [
        Patch(facecolor='blue', label='Perturb'),
        Patch(facecolor='green', label='Consist'),
        Patch(facecolor='red', label='Photom')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    # Config (winner scale - categorical)
    ax = axes[2, 2]
    unique_scales = np.unique(winner_scale[~np.isnan(winner_scale)])
    n_scales = len(unique_scales)
    
    # Map scales to indices for colormap
    scale_to_idx = {s: i for i, s in enumerate(sorted(unique_scales))}
    config_idx = np.zeros_like(winner_scale)
    for scale, idx in scale_to_idx.items():
        config_idx[winner_scale == scale] = idx
    config_idx[np.isnan(winner_scale)] = np.nan
    
    # Use tab10 colormap for categorical
    config_cmap = plt.colormaps.get_cmap('tab10').resampled(n_scales)
    im = ax.imshow(config_idx, cmap=config_cmap, vmin=-0.5, vmax=n_scales-0.5)
    ax.set_title('Config (scale)')
    ax.axis('off')
    
    # Legend for config
    legend_elements = [
        Patch(facecolor=config_cmap(i), label=f'{int(s)}')
        for i, s in enumerate(sorted(unique_scales))
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    # =========================================================================
    # Row 3: Salience, Cost, EPE
    # =========================================================================
    
    # Salience (chi-squared style)
    ax = axes[3, 0]
    vmax = np.nanmax(salience)
    im = ax.imshow(salience, cmap='hot', vmin=0, vmax=vmax)
    ax.set_title(f'Salience ({vmax:.2f})')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Cost (selection criterion)
    ax = axes[3, 1]
    vmax = np.nanmax(cost)
    im = ax.imshow(cost, cmap=blue_red, vmin=0, vmax=vmax)
    ax.set_title(f'Cost ({vmax:.2f})')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # EPE (if ground truth available)
    ax = axes[3, 2]
    if u_gt is not None and v_gt is not None:
        epe = np.sqrt((u - u_gt)**2 + (v - v_gt)**2)
        # Log scale with min clip at 0.1
        epe_clipped = np.clip(epe, 0.1, None)
        from matplotlib.colors import LogNorm
        vmax = np.nanmax(epe_clipped)
        im = ax.imshow(epe_clipped, cmap='hot', norm=LogNorm(vmin=0.1, vmax=vmax))
        median_epe = np.nanmedian(epe)
        if median_epe < 0.001:
            ax.set_title(f'EPE (med={median_epe:.1e})')
        else:
            ax.set_title(f'EPE (med={median_epe:.3f})')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, 'No ground truth', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')
        ax.set_title('EPE (N/A)')
    ax.axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 {output_path}")
    
    plt.close(fig)
    return fig


def main():
    import tomllib
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_diagnostic.py <config.toml>")
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)
    
    # Load config to get data directory
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    
    data_dir = Path('data')
    
    # Compute movie hash from config (same logic as run_experiment.py)
    if 'sprites' in config:
        from src_sprites.generate_sprites import extract_generation_config, compute_generation_hash
        gen_config = extract_generation_config(config)
        movie_hash = compute_generation_hash(gen_config)
    elif 'source' in config:
        import hashlib
        import json
        source = config['source']
        source_type = source['type']
        
        # Match extract_source_config from run_experiment.py
        source_config = {
            'type': source_type,
            'sequence': source.get('sequence'),
            'frames': source.get('frames'),
        }
        
        if source_type == 'sintel':
            source_config['pass'] = source.get('pass')
            source_config['root'] = source.get('sintel_root', '')
        elif source_type == 'middlebury':
            source_config['root'] = source.get('middlebury_root', '')
        else:
            source_config['root'] = source.get('root', '')
        
        config_str = json.dumps(source_config, sort_keys=True)
        movie_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]
    else:
        print("❌ Config must have [sprites.*] or [source] section")
        sys.exit(1)
    
    movie_dir = data_dir / movie_hash
    if not movie_dir.exists():
        print(f"❌ Movie directory not found: {movie_dir}")
        print(f"   Run 'python scripts/run_experiment.py {config_path}' first")
        sys.exit(1)
    
    print(f"📁 Movie: {movie_dir}")
    
    # Load movie sequence for ground truth access
    movie = load_movie_sequence(movie_dir, boundary_margin=10)
    
    # Apply preprocessing if configured
    from scripts.optical_flow_track import get_preprocess_config, preprocess_movie
    preprocess_cfg = get_preprocess_config(config)
    movie = preprocess_movie(movie, preprocess_cfg)
    
    # Compute OF hash from config
    from scripts.optical_flow_track import extract_of_config, compute_of_hash
    of_config = extract_of_config(config)
    of_hash = compute_of_hash(of_config)
    
    analysis_dir = movie_dir / 'analysis' / of_hash
    if not analysis_dir.exists():
        print(f"❌ Analysis directory not found: {analysis_dir}")
        print(f"   Run 'python scripts/run_experiment.py {config_path}' first")
        sys.exit(1)
    
    print(f"📁 Analysis: {analysis_dir}")
    
    # Find all pair directories
    sweep_dir = analysis_dir / 'sweep'
    if not sweep_dir.exists():
        print(f"❌ No sweep directory in {analysis_dir}")
        sys.exit(1)
    
    pair_dirs = sorted([d for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith('pair_')])
    if not pair_dirs:
        print(f"❌ No pair directories in {sweep_dir}")
        sys.exit(1)
    
    print(f"✅ Found {len(pair_dirs)} pairs")
    
    # Selection config
    selection_config = {
        'normalize': 'none',
        'aggregation': 'max',
        'power': 2.0,
        'traction': 0.0,
        'perturbation_rms': 1.0,
        'consistency': 1.0,
        'photometric': 1.0,
    }
    validate_selection_config(selection_config, 'test')
    
    # Process each pair
    for pair_dir in pair_dirs:
        # Extract pair index from directory name (pair_000 -> 0)
        pair_idx = int(pair_dir.name.split('_')[1])
        
        # Load results
        results_path = pair_dir / 'results_full.pkl'
        if not results_path.exists():
            print(f"⚠️  Skipping {pair_dir.name}: no results_full.pkl")
            continue
        
        with open(results_path, 'rb') as f:
            results_full = pickle.load(f)
        
        # Get frame from preprocessed movie
        if pair_idx >= len(movie.pairs):
            print(f"⚠️  Skipping {pair_dir.name}: pair index out of range")
            continue
        
        frame_a = movie.pairs[pair_idx].frame1_original
        
        # Create valid mask
        H, W = results_full[0]['flows']['u_AB'].shape
        valid_mask = np.ones((H, W), dtype=bool)
        valid_mask[:10, :] = False
        valid_mask[-10:, :] = False
        valid_mask[:, :10] = False
        valid_mask[:, -10:] = False
        
        # Run selection
        ensemble = select_ensemble(results_full, selection_config, valid_mask)
        selection = ensemble['ensemble_selection']
        
        # Gather flows
        u = ensemble['u_ensemble_forward']
        v = ensemble['v_ensemble_forward']
        
        # Gather metrics from winning config per pixel
        traction = gather_metric(results_full, selection, 'traction_A')
        perturbation = gather_metric(results_full, selection, 'displacements_sensitivity_A2B')
        consistency = gather_metric(results_full, selection, 'consistency_A')
        photometric = gather_metric(results_full, selection, 'photometric_A')
        
        # Spread, salience, and winner_scale from ensemble
        spread_u = ensemble['spread_u']
        spread_v = ensemble['spread_v']
        spread = np.sqrt(spread_u**2 + spread_v**2)
        salience = ensemble['spread_salience']
        winner_scale = ensemble['winner_scale']
        
        # Gather cost from winning config per pixel
        cost_stack = ensemble['cost_stack']
        ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        cost = cost_stack[selection, ii, jj]
        
        # Mask invalid
        u[~valid_mask] = np.nan
        v[~valid_mask] = np.nan
        traction[~valid_mask] = np.nan
        perturbation[~valid_mask] = np.nan
        consistency[~valid_mask] = np.nan
        photometric[~valid_mask] = np.nan
        spread[~valid_mask] = np.nan
        salience[~valid_mask] = np.nan
        cost[~valid_mask] = np.nan
        winner_scale[~valid_mask] = np.nan
        
        # Get ground truth if available
        u_gt = None
        v_gt = None
        if pair_idx < len(movie.pairs) and movie.pairs[pair_idx].has_gt:
            u_gt = movie.pairs[pair_idx].u_truth.copy()
            v_gt = movie.pairs[pair_idx].v_truth.copy()
            u_gt[~valid_mask] = np.nan
            v_gt[~valid_mask] = np.nan
        
        # Visualize
        output_path = pair_dir / 'diagnostic.png'
        visualize_diagnostic(
            frame_a, u, v,
            traction, perturbation, consistency,
            photometric, spread, salience, cost, winner_scale,
            u_gt=u_gt, v_gt=v_gt,
            output_path=output_path,
            title=f'Pair {pair_idx}: Frame {pair_idx} → {pair_idx + 1}'
        )
    
    print(f"\n✨ Generated figures for {len(pair_dirs)} pairs")


if __name__ == "__main__":
    main()
