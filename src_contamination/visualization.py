# File: src_contamination/visualization.py
"""
Visualization for boundary contamination measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Optional

from .test_pattern import create_template
from .contamination import measure_contamination
from .of_hash import config_hash, config_name


def plot_single_config(
    config: dict,
    size: int,
    shift: float,
    output_dir: Path,
    threshold: float = 0.01
) -> Path:
    """
    Create visualization for a single configuration at one size.
    
    Two-panel figure:
    - Left: Template pattern (center tile)
    - Right: Error heatmap with clean interior marked
    
    Args:
        config: Algorithm config dict (must include 'algorithm' key)
        size: Center tile size
        shift: Applied motion
        output_dir: Where to save
        threshold: Error threshold
        
    Returns:
        Path to saved figure
    """
    # Measure contamination
    depth, error = measure_contamination(config, size, shift, threshold)
    
    # Create template for display
    template = create_template(size)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # Left: Template
    ax = axes[0]
    ax.imshow(template, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Template Pattern ({size}×{size})')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Right: Error heatmap
    ax = axes[1]
    
    # Use log scale for better visibility
    error_display = np.maximum(error, 1e-6)  # Avoid log(0)
    im = ax.imshow(error_display, cmap='hot', norm=plt.matplotlib.colors.LogNorm(
        vmin=max(threshold/10, error_display.min()),
        vmax=max(error_display.max(), threshold*10)
    ))
    
    # Mark clean interior with cyan rectangle
    if depth < size // 2:
        rect = patches.Rectangle(
            (depth, depth),
            size - 2*depth,
            size - 2*depth,
            linewidth=2,
            edgecolor='cyan',
            facecolor='none',
            linestyle='--'
        )
        ax.add_patch(rect)
    
    ax.set_title(f'Error (margin={depth}px)')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Error (px)')
    
    # Add threshold line to colorbar
    cbar.ax.axhline(y=threshold, color='cyan', linewidth=2, linestyle='--')
    
    # Overall title
    name = config_name(config)
    fig.suptitle(f'{name} / shift={shift}px', fontsize=12)
    
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"template_{size}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def plot_config_comparison(
    configs: list[dict],
    shift: float,
    output_path: Path,
    sizes: list[int] = [100, 200, 400]
) -> Path:
    """
    Create horizontal bar chart comparing contamination across configs.
    
    Args:
        configs: List of config dicts to compare (each must include 'algorithm')
        shift: Applied motion
        output_path: Where to save figure
        sizes: Tile sizes to show
        
    Returns:
        Path to saved figure
    """
    from .contamination import measure_contamination
    
    # Collect data
    data = []
    for config in configs:
        name = config_name(config)
        depths = {}
        for size in sizes:
            depth, _ = measure_contamination(config, size, shift)
            depths[size] = depth
        data.append({
            'name': name,
            'depths': depths,
            'max': max(depths.values())
        })
    
    # Sort by max depth
    data.sort(key=lambda x: x['max'])
    
    # Get algorithm from first config for title
    algorithm = configs[0].get('algorithm', 'unknown') if configs else 'unknown'
    
    # Create figure
    n_configs = len(configs)
    n_sizes = len(sizes)
    
    fig, ax = plt.subplots(figsize=(10, max(4, n_configs * 0.8)))
    
    # Bar positions
    bar_height = 0.25
    y_positions = np.arange(n_configs)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_sizes))
    
    # Plot bars for each size
    for i, size in enumerate(sizes):
        depths = [d['depths'][size] for d in data]
        offset = (i - n_sizes/2 + 0.5) * bar_height
        bars = ax.barh(
            y_positions + offset,
            depths,
            height=bar_height,
            label=f'{size}px',
            color=colors[i],
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add value labels
        for bar, depth in zip(bars, depths):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height()/2,
                f'{depth}',
                va='center',
                fontsize=8
            )
    
    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels([d['name'] for d in data])
    ax.set_xlabel('Contamination Depth (pixels)')
    ax.set_title(f'{algorithm.upper()} Boundary Contamination (shift={shift}px)')
    ax.legend(title='Tile Size', loc='lower right')
    ax.set_xlim(0, max(d['max'] for d in data) * 1.3)
    
    # Grid
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_error_profile(
    config: dict,
    size: int,
    shift: float,
    output_path: Path,
    threshold: float = 0.01
) -> Path:
    """
    Plot error profile from edge to center.
    
    Shows how error decreases as we move inward from tile boundary.
    
    Args:
        config: Config dict (must include 'algorithm')
        size: Tile size
        shift: Applied motion
        output_path: Where to save
        threshold: Error threshold
        
    Returns:
        Path to saved figure
    """
    # Measure
    depth, error = measure_contamination(config, size, shift, threshold)
    
    # Compute profile: max error at each depth
    max_error_at_depth = []
    for d in range(1, size // 2):
        interior = error[d:-d, d:-d]
        if interior.size > 0:
            max_error_at_depth.append(np.max(interior))
        else:
            break
    
    depths = np.arange(1, len(max_error_at_depth) + 1)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.semilogy(depths, max_error_at_depth, 'b.-', linewidth=2, markersize=8)
    ax.axhline(threshold, color='r', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold})')
    ax.axvline(depth, color='g', linestyle='--', linewidth=1.5, label=f'Margin ({depth}px)')
    
    ax.set_xlabel('Depth from Edge (pixels)')
    ax.set_ylabel('Max Error (pixels)')
    ax.set_title(f'{config_name(config)} / size={size}px')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(50, len(depths)))
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


if __name__ == "__main__":
    from pathlib import Path
    
    print("📊 Visualization demo")
    print("=" * 40)
    
    output_dir = Path("results_contamination/demo")
    
    config = {
        'algorithm': 'farneback',
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 15,
        'iterations': 3,
        'poly_n': 5,
        'poly_sigma': 1.1,
        'flags': 0
    }
    
    # Single config plot
    path = plot_single_config(
        config, size=100, shift=1.0,
        output_dir=output_dir / "win15"
    )
    print(f"✅ Saved: {path}")
    
    # Error profile
    path = plot_error_profile(
        config, size=200, shift=1.0,
        output_path=output_dir / "error_profile.png"
    )
    print(f"✅ Saved: {path}")
