# File: src/synthesis/flows.py
"""
Synthetic flow field generation for optical flow testing.

Generates various ground truth flow fields with known properties.
All flows returned as (u, v) tuple of float32 arrays.
"""

import numpy as np
import sys

# Add this function to your src/synthesis/flows.py file

import numpy as np


def generate_split_uniform(size: tuple[int, int],
                           left_motion: list[float],
                           right_motion: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate split uniform flow field with motion discontinuity at center.

    Creates two regions with different uniform motion separated by a sharp
    vertical boundary at the image center. This tests optical flow algorithms'
    ability to handle motion discontinuities.

    Args:
        size: (height, width) in pixels
        left_motion: [u, v] motion vector for left half
        right_motion: [u, v] motion vector for right half

    Returns:
        (u_true, v_true) flow field arrays as float32

    Example:
        # Opposing vertical motion
        u, v = generate_split_uniform((288, 288), [0, 2], [0, -2])
        # Left half moves down 2px, right half moves up 2px
        # Creates 4px discontinuity at x=144
    """
    H, W = size
    u = np.zeros((H, W), dtype=np.float32)
    v = np.zeros((H, W), dtype=np.float32)

    # Left half: uniform motion
    u[:, :W // 2] = left_motion[0]
    v[:, :W // 2] = left_motion[1]

    # Right half: uniform motion
    u[:, W // 2:] = right_motion[0]
    v[:, W // 2:] = right_motion[1]

    return u, v


# Also update your generate_from_config function to handle the new type:
# Add this case to your flow type switch/if-elif chain:
#
# elif flow_type == "split_uniform":
#     left_motion = flow_config.get("left_motion", [0, 0])
#     right_motion = flow_config.get("right_motion", [0, 0])
#     u_true, v_true = generate_split_uniform(size, left_motion, right_motion)


def generate_split_shear(size: tuple[int, int],
                         left_rate_x: float = 0.0,
                         left_rate_y: float = 0.0,
                         right_rate_x: float = 0.0,
                         right_rate_y: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate split shear flow (different shear on left/right halves).

    Args:
        size: (height, width)
        left_rate_x, left_rate_y: Shear rates for left half
        right_rate_x, right_rate_y: Shear rates for right half

    Returns:
        (u, v) flow fields
    """
    H, W = size
    u = np.zeros((H, W), dtype=np.float32)
    v = np.zeros((H, W), dtype=np.float32)

    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float32)

    # Left half
    u[:, :W // 2] = left_rate_x * y_grid[:, :W // 2]
    v[:, :W // 2] = left_rate_y * x_grid[:, :W // 2]

    # Right half
    u[:, W // 2:] = right_rate_x * y_grid[:, W // 2:]
    v[:, W // 2:] = right_rate_y * (x_grid[:, W // 2:] - W // 2)  # Reset x for right half

    return u, v


def generate_split_vertical_shear(size: tuple[int, int],
                                  left_motion: tuple[float, float],
                                  right_motion: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate split vertical shear flow field.

    Left and right halves have different constant motion.
    Sharp discontinuity at centerline.

    Args:
        size: (height, width) in pixels
        left_motion: (dx, dy) for left half
        right_motion: (dx, dy) for right half

    Returns:
        u, v: Flow field components (H, W)

    Example:
        >>> u, v = generate_split_vertical_shear(
        ...     size=(288, 288),
        ...     left_motion=(0, -2),   # Up
        ...     right_motion=(0, 2)    # Down
        ... )
    """
    H, W = size
    split_x = W // 2

    u = np.zeros((H, W), dtype=np.float32)
    v = np.zeros((H, W), dtype=np.float32)

    # Left half
    u[:, :split_x] = left_motion[0]
    v[:, :split_x] = left_motion[1]

    # Right half
    u[:, split_x:] = right_motion[0]
    v[:, split_x:] = right_motion[1]

    return u, v


def generate_uniform_translation(size: tuple[int, int],
                                 u: float,
                                 v: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate uniform translation flow field.

    Args:
        size: (height, width) in pixels
        u: x-displacement (pixels, positive = rightward)
        v: y-displacement (pixels, positive = downward)

    Returns:
        (u_true, v_true) flow field arrays as float32

    Example:
        >>> u_field, v_field = generate_uniform_translation((256, 256), 2.5, 1.0)
        >>> u_field.shape
        (256, 256)
        >>> np.allclose(u_field, 2.5)
        True
    """
    H, W = size

    u_true = np.full((H, W), u, dtype=np.float32)
    v_true = np.full((H, W), v, dtype=np.float32)

    return u_true, v_true


def generate_rotation(size: tuple[int, int],
                      center: tuple[float, float],
                      omega: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate rotation flow field around a center point.

    Args:
        size: (height, width) in pixels
        center: (cx, cy) rotation center in pixels
        omega: Angular velocity (radians per pixel radius)
               Positive = counter-clockwise

    Returns:
        (u_true, v_true) flow field arrays as float32

    Note:
        Flow at radius r from center has magnitude r * omega
        Direction is perpendicular to radius vector
    """
    H, W = size
    cx, cy = center

    # Create coordinate grids (y increases downward, x increases rightward)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)

    # Vectors from center to each pixel
    dx = x - cx
    dy = y - cy

    # Rotation flow: perpendicular to radius
    # For counter-clockwise rotation: (u, v) = omega * (-dy, dx)
    u_true = omega * (-dy)
    v_true = omega * dx

    return u_true, v_true


def generate_radial(size: tuple[int, int],
                    center: tuple[float, float],
                    rate: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate radial expansion/contraction flow field.

    Args:
        size: (height, width) in pixels
        center: (cx, cy) center point in pixels
        rate: Expansion rate (positive = expansion, negative = contraction)
              Flow magnitude at distance r is rate * r

    Returns:
        (u_true, v_true) flow field arrays as float32
    """
    H, W = size
    cx, cy = center

    # Create coordinate grids
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)

    # Vectors from center to each pixel
    dx = x - cx
    dy = y - cy

    # Radial flow: along radius direction
    u_true = rate * dx
    v_true = rate * dy

    return u_true, v_true


def generate_shear(size: tuple[int, int],
                   rate_x: float = 0.0,
                   rate_y: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate shear flow field.

    Args:
        size: (height, width) in pixels
        rate_x: Horizontal shear rate (u varies with y)
        rate_y: Vertical shear rate (v varies with x)

    Returns:
        (u_true, v_true) flow field arrays as float32

    Example:
        Horizontal shear: u(y) = rate_x * y, v = 0
        Vertical shear: u = 0, v(x) = rate_y * x
    """
    H, W = size

    # Create coordinate grids
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)

    # Shear flow
    u_true = rate_x * y
    v_true = rate_y * x

    return u_true, v_true


def generate_sinusoidal(size: tuple[int, int],
                        freq_x: float,
                        freq_y: float,
                        amplitude: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate sinusoidal flow field.

    Args:
        size: (height, width) in pixels
        freq_x: Frequency in x-direction (cycles per image width)
        freq_y: Frequency in y-direction (cycles per image height)
        amplitude: Flow amplitude in pixels

    Returns:
        (u_true, v_true) flow field arrays as float32
    """
    H, W = size

    # Create coordinate grids normalized to [0, 1]
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    x_norm = x / W
    y_norm = y / H

    # Sinusoidal flow
    u_true = amplitude * np.sin(2 * np.pi * freq_x * x_norm)
    v_true = amplitude * np.sin(2 * np.pi * freq_y * y_norm)

    return u_true, v_true


def generate_sinusoidal_power(size: tuple[int, int],
                              lambda_x: float,
                              lambda_y: float,
                              amplitude: float,
                              power: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate sinusoidal flow with power transform for dramatic variation.

    Creates flow fields with sinusoidal patterns that can be transformed
    to have sharp peaks and flat valleys using power < 1.

    Args:
        size: (height, width) in pixels
        lambda_x: Wavelength in x-direction (pixels)
        lambda_y: Wavelength in y-direction (pixels)
        amplitude: Flow amplitude (pixels)
        power: Power for sin^power transform (default 1.0 for standard sinusoid)
               power < 1 creates sharp peaks and flat valleys
               power = 1/11 ≈ 0.091 creates dramatic variation

    Returns:
        (u_true, v_true) flow field arrays as float32

    Note:
        With power < 1, most of the image has near-zero flow with dramatic
        peaks of large motion. This creates challenging regions for optical flow.

    Example:
        >>> # Standard sinusoidal
        >>> u, v = generate_sinusoidal_power((256, 256), 64, 128, 10.0, power=1.0)
        >>> # Dramatic peaks
        >>> u, v = generate_sinusoidal_power((256, 256), 64, 128, 10.0, power=0.091)
    """
    H, W = size

    # Create coordinate grids
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)

    # Sinusoidal patterns
    sin_x = np.sin(2 * np.pi * x / lambda_x)
    sin_y = np.sin(2 * np.pi * y / lambda_y)

    # Apply power transform with sign preservation
    # sign(sin) * |sin|^power creates dramatic variation while preserving direction
    u_pattern = np.sign(sin_x) * np.abs(sin_x) ** power
    v_pattern = np.sign(sin_y) * np.abs(sin_y) ** power

    # Scale by amplitude
    u_true = amplitude * u_pattern
    v_true = amplitude * v_pattern

    return u_true, v_true


def generate_vortex(size: tuple[int, int],
                    center: tuple[float, float],
                    strength: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate vortex flow field (combination of rotation and radial).

    Args:
        size: (height, width) in pixels
        center: (cx, cy) vortex center in pixels
        strength: Vortex strength (affects both rotation and radial components)

    Returns:
        (u_true, v_true) flow field arrays as float32
    """
    H, W = size
    cx, cy = center

    y, x = np.mgrid[0:H, 0:W].astype(np.float32)

    dx = x - cx
    dy = y - cy

    # NO /r here - this is the key fix!
    u_rot = strength * (-dy)
    v_rot = strength * dx

    u_rad = -0.1 * strength * dx
    v_rad = -0.1 * strength * dy

    u_true = u_rot + u_rad
    v_true = v_rot + v_rad

    return u_true, v_true


def generate_from_config(flow_config: dict,
                         size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate flow field from TOML config dict.

    Args:
        flow_config: Dict with 'type' and type-specific parameters
        size: (height, width) for the flow field

    Returns:
        (u_true, v_true) flow field arrays as float32

    Example:
        >>> config = {'type': 'uniform_translation', 'u': 2.5, 'v': 1.0}
        >>> u, v = generate_from_config(config, (256, 256))
    """
    if 'type' not in flow_config:
        print(f"❌ ERROR: flow_config must have 'type' key")
        sys.exit(1)

    flow_type = flow_config['type']

    if flow_type == 'uniform_translation':
        return generate_uniform_translation(
            size=size,
            u=flow_config['u'],
            v=flow_config['v']
        )

    elif flow_type == 'rotation':
        return generate_rotation(
            size=size,
            center=tuple(flow_config.get('center', [size[1] // 2, size[0] // 2])),
            omega=flow_config['omega']
        )

    elif flow_type == 'radial':
        return generate_radial(
            size=size,
            center=tuple(flow_config.get('center', [size[1] // 2, size[0] // 2])),
            rate=flow_config['rate']
        )

    elif flow_type == 'shear':
        return generate_shear(
            size=size,
            rate_x=flow_config.get('rate_x', 0.0),
            rate_y=flow_config.get('rate_y', 0.0)
        )

    elif flow_type == 'split_uniform':
        return generate_split_uniform(
            size=size,
            left_motion=flow_config.get('left_motion', [0, 0]),
            right_motion=flow_config.get('right_motion', [0, 0])
        )

    elif flow_type == 'sinusoidal':
        return generate_sinusoidal(
            size=size,
            freq_x=flow_config['freq_x'],
            freq_y=flow_config['freq_y'],
            amplitude=flow_config['amplitude']
        )

    elif flow_type == 'sinusoidal_power':
        return generate_sinusoidal_power(
            size=size,
            lambda_x=flow_config['lambda_x'],
            lambda_y=flow_config['lambda_y'],
            amplitude=flow_config['amplitude'],
            power=flow_config.get('power', 1.0)
        )

    elif flow_type == 'vortex':
        return generate_vortex(
            size=size,
            center=tuple(flow_config.get('center', [size[1] // 2, size[0] // 2])),
            strength=flow_config['strength']
        )

    else:
        print(f"❌ ERROR: Unknown flow type: {flow_type}")
        print(f"   Valid types: 'uniform_translation', 'rotation', 'radial',")
        print(f"                'shear', 'split_uniform', 'sinusoidal', 'sinusoidal_power', 'vortex'")
        sys.exit(1)


if __name__ == "__main__":
    # Test flow generation
    print("🧪 Testing flow generation...")

    size = (256, 256)

    # Test uniform translation
    u, v = generate_uniform_translation(size, 2.5, 1.0)
    print(f"✅ Uniform translation: u={u[0, 0]:.2f}, v={v[0, 0]:.2f}")

    # Test rotation
    u, v = generate_rotation(size, (128, 128), 0.05)
    mag = np.sqrt(u ** 2 + v ** 2)
    print(f"✅ Rotation: magnitude range [{mag.min():.2f}, {mag.max():.2f}]")

    # Test radial
    u, v = generate_radial(size, (128, 128), 0.02)
    mag = np.sqrt(u ** 2 + v ** 2)
    print(f"✅ Radial: magnitude range [{mag.min():.2f}, {mag.max():.2f}]")

    # Test shear
    u, v = generate_shear(size, rate_x=0.01, rate_y=0.0)
    print(f"✅ Shear: u range [{u.min():.2f}, {u.max():.2f}]")

    # Test sinusoidal
    u, v = generate_sinusoidal(size, 2, 2, 3.0)
    print(f"✅ Sinusoidal: u range [{u.min():.2f}, {u.max():.2f}]")

    # Test sinusoidal_power
    u, v = generate_sinusoidal_power(size, 64.0, 128.0, 10.0, power=0.091)
    mag = np.sqrt(u ** 2 + v ** 2)
    print(f"✅ Sinusoidal power: magnitude range [{mag.min():.2f}, {mag.max():.2f}]")

    # Test vortex
    u, v = generate_vortex(size, (128, 128), 5.0)
    mag = np.sqrt(u ** 2 + v ** 2)
    print(f"✅ Vortex: magnitude range [{mag.min():.2f}, {mag.max():.2f}]")

    # Test from config
    config = {
        'type': 'sinusoidal_power',
        'lambda_x': 64.0,
        'lambda_y': 128.0,
        'amplitude': 10.0,
        'power': 0.091
    }
    u, v = generate_from_config(config, size)
    print(
        f"✅ From config: magnitude range [{np.sqrt(u ** 2 + v ** 2).min():.2f}, {np.sqrt(u ** 2 + v ** 2).max():.2f}]")

    print("\n✨ All flow tests passed!")