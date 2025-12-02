# File: src/synthesis/__init__.py
"""
Synthetic data generation for optical flow testing.
"""

from .patterns import (
    generate_checkerboard,
    generate_perlin_noise,
    generate_random_noise,
    generate_gradient,
    generate_rainbow_grid,
    generate_from_config as generate_pattern_from_config
)

from .flows import (
    generate_uniform_translation,
    generate_rotation,
    generate_radial,
    generate_shear,
    generate_sinusoidal,
    generate_sinusoidal_power,
    generate_vortex,
    generate_from_config as generate_flow_from_config
)

from .warping import (
    warp_image,
    create_frame_pair,
    shift_image
)

__all__ = [
    # Patterns
    'generate_checkerboard',
    'generate_perlin_noise',
    'generate_random_noise',
    'generate_gradient',
    'generate_rainbow_grid',
    'generate_pattern_from_config',
    # Flows
    'generate_uniform_translation',
    'generate_rotation',
    'generate_radial',
    'generate_shear',
    'generate_sinusoidal',
    'generate_sinusoidal_power',
    'generate_vortex',
    'generate_flow_from_config',
    # Warping
    'warp_image',
    'create_frame_pair',
    'shift_image',
]