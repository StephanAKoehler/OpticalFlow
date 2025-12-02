#!/usr/bin/env python3
# File: src_sprites/sprite_generator.py
"""
Main sprite movie generation orchestrator.

Generates frame sequences with ground truth optical flow.
"""

import sys
import hashlib
import shutil
from pathlib import Path
import numpy as np
import tomli
import tomli_w
from PIL import Image

# Handle both module and script execution
try:
    from .texture_generators import generate_texture
    from .motion import apply_motion, compute_flow_field
    from .compositor import composite_frame
except ImportError:
    from texture_generators import generate_texture
    from motion import apply_motion, compute_flow_field
    from compositor import composite_frame


def compute_config_hash(config: dict) -> str:
    """Compute deterministic hash from config."""
    config_str = tomli_w.dumps(config)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def parse_sprite_config(sprite_config: dict, image_size: tuple[int, int]) -> dict:
    """
    Parse sprite configuration and compute derived values.
    
    Args:
        sprite_config: Sprite section from TOML
        image_size: (height, width) of output image
    
    Returns:
        Parsed sprite dict with computed values
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
    
    return {
        'name': sprite_config.get('name', 'sprite'),
        'midpoint': tuple(sprite_config['midpoint']),
        'motion': tuple(sprite_config['motion']),  # [dx, dy, rot_deg]
        'z_order': sprite_config['z_order'],
        'shape': shape,
        'size': (sprite_h, sprite_w),
        'texture_type': sprite_config['texture_type'],
        'texture_params': {k: v for k, v in sprite_config.items() 
                          if k not in ['name', 'midpoint', 'motion', 'z_order', 'shape', 'dimensions', 'texture_type']}
    }


def generate_sprite_movie(config_path: Path, output_base: Path = Path('results_movies')):
    """
    Generate sprite-based movie with ground truth optical flow.
    
    Args:
        config_path: Path to TOML configuration file
        output_base: Base directory for output (default: results_movies/)
    """
    # Load config
    with open(config_path, 'rb') as f:
        config = tomli.load(f)
    
    # Compute hash
    config_hash = compute_config_hash(config)
    output_dir = output_base / config_hash
    frames_dir = output_dir / 'frames'
    
    print("=" * 80)
    print("SPRITE MOVIE GENERATION")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Hash: {config_hash}")
    print(f"Output: {output_dir}")
    print()
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(exist_ok=True)
    
    # Copy config
    shutil.copy(config_path, output_dir / 'config.toml')
    
    # Extract parameters
    image_config = config.get('image', {})
    H, W = image_config.get('size', [288, 288])
    image_size = (H, W)
    
    temporal_config = config.get('temporal', {})
    num_frames = temporal_config.get('num_frames', 10)
    
    background_config = config.get('background', {})
    sprite_configs = config.get('sprites', {})
    
    # Parse sprites
    sprites = []
    for sprite_id, (sprite_name, sprite_cfg) in enumerate(sprite_configs.items(), start=1):
        sprite_cfg['name'] = sprite_name
        parsed = parse_sprite_config(sprite_cfg, image_size)
        parsed['id'] = sprite_id
        sprites.append(parsed)
    
    print(f"📊 Configuration:")
    print(f"   Image size: {W}×{H}")
    print(f"   Frames: {num_frames}")
    print(f"   Sprites: {len(sprites)}")
    for sprite in sprites:
        print(f"      - {sprite['name']}: {sprite['shape']}, z={sprite['z_order']}, motion={sprite['motion']}")
    print()
    
    # Generate background texture once
    print("🎨 Generating background...")
    bg_type = background_config.get('type', 'solid')
    bg_params = {k: v for k, v in background_config.items() if k != 'type'}
    background = generate_texture(bg_type, image_size, **bg_params)
    print(f"   ✅ Background: {bg_type}")
    print()
    
    # Generate frames
    print(f"🎬 Generating {num_frames} frames...")
    
    for frame_idx in range(num_frames):
        # Generate sprite textures and apply motion
        frame_sprites = []
        
        for sprite in sprites:
            # Generate sprite texture
            sprite_texture = generate_texture(
                sprite['texture_type'],
                sprite['size'],
                **sprite['texture_params']
            )
            
            # Compute accumulated motion for this frame
            dx, dy, rot_deg = sprite['motion']
            total_dx = dx * frame_idx
            total_dy = dy * frame_idx
            total_rot = rot_deg * frame_idx
            
            # Current midpoint (accumulated from original)
            orig_mx, orig_my = sprite['midpoint']
            current_midpoint = (orig_mx + total_dx, orig_my + total_dy)
            
            # Create full-size sprite image (centered at midpoint)
            sprite_full = np.zeros((H, W, 3), dtype=np.uint8)
            sh, sw = sprite['size']
            
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
            
            # Apply rotation around current midpoint
            if total_rot != 0:
                sprite_full, _ = apply_motion(sprite_full, 0, 0, total_rot, current_midpoint)
            
            # Compute flow field (per-frame motion, not accumulated)
            u, v = compute_flow_field(dx, dy, rot_deg, current_midpoint, image_size)
            
            frame_sprites.append({
                'image': sprite_full,
                'z_order': sprite['z_order'],
                'id': sprite['id'],
                'u': u,
                'v': v
            })
        
        # Composite frame
        frame_image, sprite_mask, u_field, v_field = composite_frame(
            background, frame_sprites, image_size
        )
        
        # Save frame image
        frame_path = frames_dir / f'image_{frame_idx:03d}.png'
        Image.fromarray(frame_image).save(frame_path)
        
        # Save flow GT (from this frame to next)
        if frame_idx < num_frames - 1:
            # Valid mask: all pixels that have a sprite or background
            valid = np.ones((H, W), dtype=bool)
            
            # Save flow components
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
        
        if (frame_idx + 1) % 10 == 0 or frame_idx == num_frames - 1:
            print(f"   Frame {frame_idx + 1}/{num_frames}")
    
    print()
    print("=" * 80)
    print("✅ GENERATION COMPLETE")
    print("=" * 80)
    print(f"Output: {output_dir}")
    print(f"Frames: {num_frames} images + {num_frames - 1} flow pairs")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sprite-based movie with GT flow')
    parser.add_argument('config', type=Path, help='Path to TOML config file')
    parser.add_argument('--output', type=Path, default=Path('results_movies'),
                       help='Output base directory (default: results_movies/)')
    
    args = parser.parse_args()
    
    if not args.config.exists():
        print(f"❌ ERROR: Config file not found: {args.config}")
        sys.exit(1)
    
    generate_sprite_movie(args.config, args.output)
