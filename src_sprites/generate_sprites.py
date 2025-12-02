#!/usr/bin/env python3
# File: scripts/generate_sprites.py
"""
Sprite-based movie generation with ground truth optical flow.

Extracts generation config sections and creates frame sequences.
Outputs are content-hashed for reproducibility and caching.

Usage:
    python scripts/generate_sprites.py config.toml
    python scripts/generate_sprites.py config.toml --output data/
    python scripts/generate_sprites.py config.toml --force
"""

import sys
import hashlib
import shutil
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import tomli
import tomli_w
from PIL import Image

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src_sprites.texture_generators import generate_texture
from src_sprites.motion import compute_flow_field
from src_sprites.compositor import composite_frame


# =============================================================================
# Configuration Extraction
# =============================================================================

GENERATION_SECTIONS = ['image', 'temporal', 'background', 'sprites']


def extract_generation_config(config: dict) -> dict:
    """
    Extract generation-relevant sections from full config.
    
    Args:
        config: Full TOML config dict
        
    Returns:
        Dict with only generation sections
    """
    gen_config = {}
    
    for section in GENERATION_SECTIONS:
        if section in config:
            gen_config[section] = config[section]
    
    return gen_config


def compute_generation_hash(gen_config: dict) -> str:
    """
    Compute deterministic hash from generation config.
    
    Args:
        gen_config: Generation sections only
        
    Returns:
        12-character hash string
    """
    config_str = tomli_w.dumps(gen_config)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def validate_generation_config(gen_config: dict):
    """
    Validate generation config has required sections.
    
    Exits with error if validation fails.
    """
    if 'image' not in gen_config:
        print("❌ ERROR: Missing [image] section in config")
        sys.exit(1)
    
    if 'size' not in gen_config['image']:
        print("❌ ERROR: [image] section must have 'size' field")
        sys.exit(1)
    
    if 'sprites' not in gen_config or len(gen_config['sprites']) == 0:
        print("❌ ERROR: No [sprites.*] sections in config")
        sys.exit(1)
    
    # Validate each sprite
    for name, sprite in gen_config['sprites'].items():
        required = ['midpoint', 'motion', 'z_order', 'shape', 'texture_type']
        missing = [k for k in required if k not in sprite]
        if missing:
            print(f"❌ ERROR: [sprites.{name}] missing required fields: {missing}")
            sys.exit(1)


# =============================================================================
# Sprite Parsing
# =============================================================================

@dataclass
class Sprite:
    """Parsed sprite configuration."""
    name: str
    id: int
    midpoint: tuple[float, float]
    motion: tuple[float, float, float]  # dx, dy, rot_deg
    z_order: int
    shape: str
    size: tuple[int, int]  # height, width
    texture_type: str
    texture_params: dict


def parse_sprite(name: str, sprite_config: dict, image_size: tuple[int, int], sprite_id: int) -> Sprite:
    """
    Parse sprite configuration and compute derived values.
    
    Args:
        name: Sprite name from config key
        sprite_config: Sprite section from TOML
        image_size: (height, width) of output image
        sprite_id: Unique sprite ID (1-based)
    
    Returns:
        Parsed Sprite object
    """
    H, W = image_size
    shape = sprite_config['shape']
    
    # Compute sprite size based on shape
    if shape == 'quadrant':
        sprite_w, sprite_h = W // 2, H // 2
    elif shape == 'square':
        size = sprite_config.get('dimensions', 50)
        sprite_w, sprite_h = size, size
    elif shape == 'circle':
        radius = sprite_config.get('dimensions', 30)
        sprite_w, sprite_h = 2 * radius, 2 * radius
    else:
        print(f"❌ ERROR: Unknown sprite shape: {shape}")
        sys.exit(1)
    
    # Extract texture params (everything except sprite-level fields)
    sprite_fields = ['midpoint', 'motion', 'z_order', 'shape', 'dimensions', 'texture_type']
    texture_params = {k: v for k, v in sprite_config.items() if k not in sprite_fields}
    
    return Sprite(
        name=name,
        id=sprite_id,
        midpoint=tuple(sprite_config['midpoint']),
        motion=tuple(sprite_config['motion']),
        z_order=sprite_config['z_order'],
        shape=shape,
        size=(sprite_h, sprite_w),
        texture_type=sprite_config['texture_type'],
        texture_params=texture_params
    )


# =============================================================================
# Frame Generation
# =============================================================================

def generate_frame(
    background: np.ndarray,
    sprites: list[Sprite],
    frame_idx: int,
    image_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a single frame with composited sprites.
    
    Args:
        background: Background image [H, W, 3] uint8
        sprites: List of Sprite objects
        frame_idx: Current frame index
        image_size: (height, width)
    
    Returns:
        frame_image: Composited frame [H, W, 3] uint8
        sprite_mask: Sprite ID per pixel [H, W] int32
        u_field: Horizontal flow [H, W] float32
        v_field: Vertical flow [H, W] float32
    """
    H, W = image_size
    frame_sprites = []
    
    for sprite in sprites:
        # Generate sprite texture
        sprite_texture = generate_texture(
            sprite.texture_type,
            sprite.size,
            **sprite.texture_params
        )
        
        # Compute accumulated motion for this frame
        dx, dy, rot_deg = sprite.motion
        total_dx = dx * frame_idx
        total_dy = dy * frame_idx
        
        # Current midpoint (accumulated from original)
        orig_mx, orig_my = sprite.midpoint
        current_midpoint = (orig_mx + total_dx, orig_my + total_dy)
        
        # Create full-size sprite image (centered at midpoint)
        sprite_full = np.zeros((H, W, 3), dtype=np.uint8)
        sh, sw = sprite.size
        
        # Compute top-left corner to center sprite at midpoint
        top = int(current_midpoint[1] - sh // 2)
        left = int(current_midpoint[0] - sw // 2)
        
        # Compute valid regions for both source sprite and destination canvas
        src_top = max(0, -top)
        src_left = max(0, -left)
        dst_top = max(0, top)
        dst_left = max(0, left)
        dst_bottom = min(H, top + sh)
        dst_right = min(W, left + sw)
        
        # Paste valid region (clip at boundaries)
        if dst_bottom > dst_top and dst_right > dst_left:
            src_h = dst_bottom - dst_top
            src_w = dst_right - dst_left
            sprite_full[dst_top:dst_bottom, dst_left:dst_right] = \
                sprite_texture[src_top:src_top+src_h, src_left:src_left+src_w]
        
        # Create shape mask (independent of texture color)
        # This ensures black pixels in checkerboard textures are properly marked as sprite
        shape_mask = np.zeros((H, W), dtype=bool)
        if dst_bottom > dst_top and dst_right > dst_left:
            shape_mask[dst_top:dst_bottom, dst_left:dst_right] = True
        
        # Compute flow field (per-frame motion, not accumulated)
        u, v = compute_flow_field(dx, dy, rot_deg, current_midpoint, image_size)
        
        frame_sprites.append({
            'image': sprite_full,
            'mask': shape_mask,  # Explicit shape mask
            'z_order': sprite.z_order,
            'id': sprite.id,
            'u': u,
            'v': v
        })
    
    # Composite frame
    frame_image, sprite_mask, u_field, v_field = composite_frame(
        background, frame_sprites, image_size
    )
    
    return frame_image, sprite_mask, u_field, v_field


# =============================================================================
# Preview Figure Generation
# =============================================================================

def generate_preview_figure(output_dir: Path, frames_dir: Path, num_frames: int):
    """
    Generate preview figure showing first, middle, and last frames.
    
    Args:
        output_dir: Directory to save figure
        frames_dir: Directory containing frame images
        num_frames: Total number of frames
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    frame_indices = [0, num_frames // 2, num_frames - 1]
    titles = ['First Frame', 'Middle Frame', 'Last Frame']
    
    for ax, idx, title in zip(axes, frame_indices, titles):
        img_path = frames_dir / f'image_{idx:03d}.png'
        if img_path.exists():
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(f'{title} ({idx})')
        else:
            ax.text(0.5, 0.5, f'Frame {idx}\nnot found', ha='center', va='center')
        ax.axis('off')
    
    plt.tight_layout()
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    plt.savefig(figures_dir / 'preview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Saved preview figure")


# =============================================================================
# Main Generation Function
# =============================================================================

def generate_movie(config: dict, output_base: Path = Path('data'), force: bool = False) -> str:
    """
    Generate sprite-based movie with ground truth optical flow.
    
    Args:
        config: Full TOML config dict
        output_base: Base directory for output (default: data/)
        force: Force regeneration even if exists
    
    Returns:
        movie_hash: Hash identifying this movie
    """
    # Extract and validate generation config
    gen_config = extract_generation_config(config)
    validate_generation_config(gen_config)
    
    # Compute hash
    movie_hash = compute_generation_hash(gen_config)
    
    # Setup paths
    output_dir = output_base / movie_hash
    frames_dir = output_dir / 'frames'
    
    print("=" * 80)
    print("🎬 SPRITE MOVIE GENERATION")
    print("=" * 80)
    print(f"Hash: {movie_hash}")
    print(f"Output: {output_dir}")
    print()
    
    # Check if already exists
    if output_dir.exists() and not force:
        existing_frames = list(frames_dir.glob('image_*.png')) if frames_dir.exists() else []
        if len(existing_frames) > 0:
            print(f"✅ Already exists ({len(existing_frames)} frames)")
            print("   Use --force to regenerate")
            print("=" * 80)
            return movie_hash
    
    # Create directories
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(exist_ok=True)
    
    # Save generation config
    with open(output_dir / 'config.toml', 'wb') as f:
        tomli_w.dump(gen_config, f)
    
    # Extract parameters
    image_config = gen_config.get('image', {})
    H, W = image_config.get('size', [288, 288])
    image_size = (H, W)
    
    temporal_config = gen_config.get('temporal', {})
    num_frames = temporal_config.get('num_frames', 10)
    
    background_config = gen_config.get('background', {})
    sprite_configs = gen_config.get('sprites', {})
    
    # Parse sprites
    sprites = []
    for sprite_id, (sprite_name, sprite_cfg) in enumerate(sprite_configs.items(), start=1):
        sprite = parse_sprite(sprite_name, sprite_cfg, image_size, sprite_id)
        sprites.append(sprite)
    
    print(f"📋 Configuration:")
    print(f"   Image size: {W}×{H}")
    print(f"   Frames: {num_frames}")
    print(f"   Sprites: {len(sprites)}")
    for sprite in sprites:
        print(f"      - {sprite.name}: {sprite.shape}, z={sprite.z_order}, motion={sprite.motion}")
    print()
    
    # Generate background
    print("🎨 Generating background...")
    bg_type = background_config.get('type', 'solid')
    bg_params = {k: v for k, v in background_config.items() if k != 'type'}
    background = generate_texture(bg_type, image_size, **bg_params)
    print(f"   ✅ Background: {bg_type}")
    print()
    
    # Generate frames
    print(f"🎬 Generating {num_frames} frames...")
    
    for frame_idx in range(num_frames):
        # Generate frame
        frame_image, sprite_mask, u_field, v_field = generate_frame(
            background, sprites, frame_idx, image_size
        )
        
        # Save frame image
        frame_path = frames_dir / f'image_{frame_idx:03d}.png'
        Image.fromarray(frame_image).save(frame_path)
        
        # Save flow GT (from this frame to next)
        if frame_idx < num_frames - 1:
            valid = np.ones((H, W), dtype=bool)
            
            np.savez_compressed(
                frames_dir / f'u_{frame_idx:03d}.npz',
                u=u_field,
                valid=valid
            )
            np.savez_compressed(
                frames_dir / f'v_{frame_idx:03d}.npz',
                v=v_field,
                valid=valid
            )
            np.savez_compressed(
                frames_dir / f'sprite_{frame_idx:03d}.npz',
                sprite_mask=sprite_mask
            )
        
        # Progress
        if (frame_idx + 1) % 10 == 0 or frame_idx == num_frames - 1:
            print(f"   Frame {frame_idx + 1}/{num_frames}")
    
    print()
    
    # Generate preview figure
    print("📊 Generating figures...")
    generate_preview_figure(output_dir, frames_dir, num_frames)
    print()
    
    print("=" * 80)
    print("✅ GENERATION COMPLETE")
    print("=" * 80)
    print(f"Output: {output_dir}")
    print(f"Frames: {num_frames} images + {num_frames - 1} flow pairs")
    print(f"Hash: {movie_hash}")
    print("=" * 80)
    
    return movie_hash


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate sprite-based movie with ground truth optical flow'
    )
    parser.add_argument('config', type=Path, help='TOML configuration file')
    parser.add_argument('--output', type=Path, default=Path('data'),
                       help='Output base directory (default: data/)')
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration even if exists')
    
    args = parser.parse_args()
    
    # Validate config exists
    if not args.config.exists():
        print(f"❌ ERROR: Config file not found: {args.config}")
        sys.exit(1)
    
    # Load config
    with open(args.config, 'rb') as f:
        config = tomli.load(f)
    
    # Generate
    movie_hash = generate_movie(config, args.output, args.force)
    
    # Print hash for scripting
    print()
    print(f"MOVIE_HASH={movie_hash}")


if __name__ == "__main__":
    main()
