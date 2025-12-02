# File: src/synthesis/patterns.py
"""
Synthetic pattern generation for optical flow testing.

Generates various test patterns including checkerboard, Perlin noise, etc.
All images are returned as float32 in [0, 1] range.
"""

import numpy as np
import sys

try:
    from noise import pnoise2

    NOISE_AVAILABLE = True
except ImportError:
    NOISE_AVAILABLE = False


def generate_sinusoidal_grid(size: tuple[int, int],
                             wavelength_x: float = 20.0,
                             wavelength_y: float = 20.0,
                             amplitude: float = 100.0,
                             offset: float = 128.0,
                             seed: int = 42) -> np.ndarray:
    H, W = size

    y, x = np.ogrid[0:H, 0:W]

    pattern = (np.sin(2 * np.pi * x / wavelength_x) *
               np.sin(2 * np.pi * y / wavelength_y))

    print(f"DEBUG: pattern range: [{pattern.min():.3f}, {pattern.max():.3f}]")

    image = offset + amplitude * pattern

    print(f"DEBUG: before clip: [{image.min():.1f}, {image.max():.1f}]")

    image = np.clip(image, 0, 255).astype(np.float32)

    print(f"DEBUG: after clip: [{image.min():.1f}, {image.max():.1f}]")

    return image

def generate_gaussian_noise_texture(size: tuple[int, int],
                                    base_level: float = 128.0,
                                    noise_std: float = 50.0,
                                    blur_sigma: float = 3.0,
                                    seed: int = 42) -> np.ndarray:
    """
    Generate smooth Gaussian noise texture.

    Creates dense, trackable texture by blurring random noise.
    Strong gradients everywhere, continuous features.

    Args:
        size: (height, width) in pixels
        base_level: Mean intensity (0-255)
        noise_std: Standard deviation of noise
        blur_sigma: Gaussian blur for smoothness (larger = smoother features)
        seed: Random seed

    Returns:
        image: (H, W) grayscale float32 in [0, 255]

    Example:
        >>> img = generate_gaussian_noise_texture((288, 288),
        ...                                       noise_std=50,
        ...                                       blur_sigma=3.0)
    """
    from scipy.ndimage import gaussian_filter

    np.random.seed(seed)
    H, W = size

    # Generate random noise
    image = np.random.randn(H, W).astype(np.float32) * noise_std + base_level

    # Smooth to create continuous trackable features
    image = gaussian_filter(image, sigma=blur_sigma)

    # Clip to valid intensity range
    image = np.clip(image, 0, 255)

    return image

# File: src/synthesis/patterns.py (add this function)

def generate_split_meshgrid(size: tuple[int, int],
                            square_size: int,
                            left_color: tuple[int, int, int],
                            right_color: tuple[int, int, int],
                            line_width: int = 1,
                            line_color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Generate split meshgrid with different colors on left and right.

    Args:
        size: (height, width) in pixels
        square_size: Size of each square in meshgrid
        left_color: RGB color for left half
        right_color: RGB color for right half
        line_width: Width of grid lines in pixels
        line_color: RGB color for grid lines

    Returns:
        image: (H, W, 3) RGB uint8 image with split meshgrid
    """
    H, W = size
    image = np.zeros((H, W, 3), dtype=np.uint8)  # ← Already uint8

    # Fill left half
    split_x = W // 2
    image[:, :split_x] = left_color

    # Fill right half
    image[:, split_x:] = right_color

    # Draw horizontal grid lines
    for y in range(0, H, square_size):
        y_start = max(0, y - line_width // 2)
        y_end = min(H, y + line_width // 2 + 1)
        image[y_start:y_end, :] = line_color

    # Draw vertical grid lines
    for x in range(0, W, square_size):
        x_start = max(0, x - line_width // 2)
        x_end = min(W, x + line_width // 2 + 1)
        image[:, x_start:x_end] = line_color

    return image

def generate_checkerboard(size: tuple[int, int],
                          square_size: int,
                          seed: int = None) -> np.ndarray:
    """
    Generate checkerboard pattern.

    Args:
        size: (height, width) in pixels
        square_size: Size of each square in pixels
        seed: Random seed (currently unused, for future noise variation)

    Returns:
        Grayscale image as float32 in [0, 1]

    Example:
        >>> img = generate_checkerboard((512, 512), 32)
        >>> img.shape
        (512, 512)
        >>> img.dtype
        dtype('float32')
    """
    H, W = size

    if square_size <= 0:
        print(f"❌ ERROR: square_size must be positive, got {square_size}")
        sys.exit(1)

    # Create coordinate grids
    y, x = np.mgrid[0:H, 0:W]

    # Compute checkerboard pattern
    # Each square alternates between 0 and 1
    checkerboard = ((x // square_size) + (y // square_size)) % 2

    return checkerboard.astype(np.float32)


def generate_perlin_noise(size: tuple[int, int],
                          scale: float = 100.0,
                          octaves: int = 6,
                          persistence: float = 0.5,
                          lacunarity: float = 2.0,
                          seed: int = 0) -> np.ndarray:
    """
    Generate Perlin noise pattern.

    Args:
        size: (height, width) in pixels
        scale: Scale of the noise (larger = smoother)
        octaves: Number of octaves for detail
        persistence: Amplitude decay per octave
        lacunarity: Frequency increase per octave
        seed: Random seed for reproducibility

    Returns:
        Grayscale image as float32 in [0, 1]

    Note:
        Requires 'noise' package: pip install noise
    """
    if not NOISE_AVAILABLE:
        print(f"❌ ERROR: Perlin noise requires 'noise' package")
        print(f"   Install with: pip install noise")
        sys.exit(1)

    H, W = size

    # Generate Perlin noise
    image = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            # Normalize coordinates
            x = j / scale
            y = i / scale

            # Generate noise value
            value = pnoise2(x, y,
                            octaves=octaves,
                            persistence=persistence,
                            lacunarity=lacunarity,
                            base=seed)

            image[i, j] = value

    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    return image


def generate_random_noise(size: tuple[int, int],
                          seed: int = None) -> np.ndarray:
    """
    Generate random Gaussian noise pattern.

    Args:
        size: (height, width) in pixels
        seed: Random seed for reproducibility

    Returns:
        Grayscale image as float32 in [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)

    H, W = size

    # Generate Gaussian noise
    noise = np.random.randn(H, W).astype(np.float32)

    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

    return noise


def generate_gradient(size: tuple[int, int],
                      direction: str = 'horizontal') -> np.ndarray:
    """
    Generate linear gradient pattern.

    Args:
        size: (height, width) in pixels
        direction: 'horizontal', 'vertical', or 'diagonal'

    Returns:
        Grayscale image as float32 in [0, 1]
    """
    H, W = size

    if direction == 'horizontal':
        gradient = np.linspace(0, 1, W, dtype=np.float32)
        image = np.tile(gradient, (H, 1))

    elif direction == 'vertical':
        gradient = np.linspace(0, 1, H, dtype=np.float32)
        image = np.tile(gradient[:, np.newaxis], (1, W))

    elif direction == 'diagonal':
        y, x = np.mgrid[0:H, 0:W]
        image = (x + y) / (W + H - 2)
        image = image.astype(np.float32)

    else:
        print(f"❌ ERROR: Unknown gradient direction: {direction}")
        print(f"   Valid options: 'horizontal', 'vertical', 'diagonal'")
        sys.exit(1)

    return image


def generate_rainbow_grid(size: tuple[int, int],
                          grid_sizes: list[int] = None,
                          seed: int = 42) -> np.ndarray:
    """
    Generate rainbow gradient background with overlapping grids.

    Creates rich, multi-scale texture ideal for optical flow tracking:
    - Rainbow gradient background (sinusoidal RGB variation)
    - Perlin noise overlay for texture
    - Multiple black grid lines at different scales

    Args:
        size: (height, width) in pixels
        grid_sizes: List of grid sizes in pixels (default: [16, 32, 64])
        seed: Random seed for Perlin noise

    Returns:
        RGB image as float32 in [0, 1], shape (H, W, 3)

    Example:
        >>> img = generate_rainbow_grid((256, 256), grid_sizes=[16, 32, 64])
        >>> img.shape
        (256, 256, 3)
    """
    if grid_sizes is None:
        grid_sizes = [16, 32, 64]

    H, W = size

    # Create coordinate grids
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)

    # Rainbow gradient background - sinusoidal variation in RGB
    # Phase shifts by 1/3 create rainbow effect
    r = 0.5 + 0.5 * np.sin(2 * np.pi * x / W)
    g = 0.5 + 0.5 * np.sin(2 * np.pi * (x / W + 1 / 3))
    b = 0.5 + 0.5 * np.sin(2 * np.pi * (x / W + 2 / 3))

    image = np.stack([r, g, b], axis=-1)

    # Add Perlin noise for texture (if available)
    if NOISE_AVAILABLE:
        perlin = generate_perlin_noise(size, scale=30.0, octaves=4, seed=seed)
        for c in range(3):
            # Modulate RGB channels with Perlin noise
            # 0.7 + 0.3*perlin keeps values in [0.7, 1.0] range
            image[:, :, c] = image[:, :, c] * (0.7 + 0.3 * perlin)

    # Add black grid lines at multiple scales
    for grid_size in grid_sizes:
        thickness = max(2, grid_size // 16)

        # Vertical lines
        v_lines = (x % grid_size) < thickness
        # Horizontal lines
        h_lines = (y % grid_size) < thickness

        # Combine
        grid_mask = v_lines | h_lines

        # Darken grid lines (multiply by 0.3)
        image[grid_mask] *= 0.3

    return image.astype(np.float32)


def apply_split_coloring(grayscale_image: np.ndarray,
                        left_color: tuple[int, int, int],
                        right_color: tuple[int, int, int]) -> np.ndarray:
    """
    Apply left/right RGB coloring to grayscale pattern.
    
    Args:
        grayscale_image: (H, W) float32 in [0, 1]
        left_color: RGB tuple (0-255) for left half
        right_color: RGB tuple (0-255) for right half
    
    Returns:
        RGB image as float32 in [0, 1], shape (H, W, 3)
    """
    H, W = grayscale_image.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    
    split_x = W // 2
    
    # Left half: scale grayscale by left_color
    for c in range(3):
        rgb[:, :split_x, c] = grayscale_image[:, :split_x] * (left_color[c] / 255.0)
    
    # Right half: scale grayscale by right_color
    for c in range(3):
        rgb[:, split_x:, c] = grayscale_image[:, split_x:] * (right_color[c] / 255.0)
    
    return rgb


def apply_uniform_coloring(grayscale_image: np.ndarray,
                           color: tuple[int, int, int]) -> np.ndarray:
    """
    Apply uniform RGB coloring to grayscale pattern.
    
    Args:
        grayscale_image: (H, W) float32 in [0, 1]
        color: RGB tuple (0-255)
    
    Returns:
        RGB image as float32 in [0, 1], shape (H, W, 3)
    """
    H, W = grayscale_image.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    
    for c in range(3):
        rgb[:, :, c] = grayscale_image * (color[c] / 255.0)
    
    return rgb


def generate_from_config(pattern_config: dict) -> np.ndarray:
    """
    Generate pattern from TOML config dict with optional RGB coloring.

    Args:
        pattern_config: Dict with 'type' and type-specific parameters
                       Optional: 'left' and 'right' subsections for split coloring
                       Optional: 'color' for uniform coloring

    Returns:
        Generated image as float32 in [0, 1]
        Shape: (H, W) for grayscale or (H, W, 3) for RGB

    Example:
        >>> # Grayscale checkerboard
        >>> config = {'type': 'checkerboard', 'size': [256, 256], 'square_size': 16}
        >>> img = generate_from_config(config)
        >>> 
        >>> # RGB split checkerboard
        >>> config = {
        ...     'type': 'checkerboard',
        ...     'size': [256, 256],
        ...     'square_size': 16,
        ...     'left': {'color_rgb': [255, 0, 0]},
        ...     'right': {'color_rgb': [0, 0, 255]}
        ... }
        >>> img = generate_from_config(config)
    """
    if 'type' not in pattern_config:
        print(f"❌ ERROR: pattern_config must have 'type' key")
        sys.exit(1)

    pattern_type = pattern_config['type']

    # Generate base grayscale pattern
    if pattern_type == 'checkerboard':
        image = generate_checkerboard(
            size=tuple(pattern_config['size']),
            square_size=pattern_config['square_size'],
            seed=pattern_config.get('seed')
        )

    elif pattern_type == 'perlin_noise':
        image = generate_perlin_noise(
            size=tuple(pattern_config['size']),
            scale=pattern_config.get('scale', 100.0),
            octaves=pattern_config.get('octaves', 6),
            persistence=pattern_config.get('persistence', 0.5),
            lacunarity=pattern_config.get('lacunarity', 2.0),
            seed=pattern_config.get('seed', 0)
        )

    elif pattern_type == 'random_noise':
        image = generate_random_noise(
            size=tuple(pattern_config['size']),
            seed=pattern_config.get('seed')
        )

    elif pattern_type == 'gradient':
        image = generate_gradient(
            size=tuple(pattern_config['size']),
            direction=pattern_config.get('direction', 'horizontal')
        )

    elif pattern_type == 'rainbow_grid':
        # Rainbow grid is already RGB, return as-is
        return generate_rainbow_grid(
            size=tuple(pattern_config['size']),
            grid_sizes=pattern_config.get('grid_sizes', [16, 32, 64]),
            seed=pattern_config.get('seed', 42)
        )

    else:
        print(f"❌ ERROR: Unknown pattern type: {pattern_type}")
        print(f"   Valid types: 'checkerboard', 'perlin_noise', 'random_noise', 'gradient', 'rainbow_grid'")
        sys.exit(1)

    # Apply RGB coloring if specified
    if 'left' in pattern_config and 'right' in pattern_config:
        # Split coloring (left/right halves)
        left_color = tuple(pattern_config['left']['color_rgb'])
        right_color = tuple(pattern_config['right']['color_rgb'])
        image = apply_split_coloring(image, left_color, right_color)
        print(f"   Applied split RGB coloring: left={left_color}, right={right_color}")
    
    elif 'color' in pattern_config:
        # Uniform coloring
        color = tuple(pattern_config['color'])
        image = apply_uniform_coloring(image, color)
        print(f"   Applied uniform RGB coloring: {color}")
    
    return image


if __name__ == "__main__":
    # Test pattern generation
    print("🧪 Testing pattern generation...")

    # Test checkerboard
    checkerboard = generate_checkerboard((256, 256), 32)
    print(f"✅ Checkerboard: shape={checkerboard.shape}, dtype={checkerboard.dtype}")
    print(f"   Range: [{checkerboard.min():.2f}, {checkerboard.max():.2f}]")

    # Test random noise
    noise = generate_random_noise((256, 256), seed=42)
    print(f"✅ Random noise: shape={noise.shape}, dtype={noise.dtype}")
    print(f"   Range: [{noise.min():.2f}, {noise.max():.2f}]")

    # Test gradient
    gradient = generate_gradient((256, 256), 'horizontal')
    print(f"✅ Gradient: shape={gradient.shape}, dtype={gradient.dtype}")
    print(f"   Range: [{gradient.min():.2f}, {gradient.max():.2f}]")

    # Test rainbow grid
    rainbow = generate_rainbow_grid((256, 256), grid_sizes=[16, 32, 64])
    print(f"✅ Rainbow grid: shape={rainbow.shape}, dtype={rainbow.dtype}")
    print(f"   Range: [{rainbow.min():.2f}, {rainbow.max():.2f}]")

    # Test Perlin noise if available
    if NOISE_AVAILABLE:
        perlin = generate_perlin_noise((256, 256), scale=50.0, seed=42)
        print(f"✅ Perlin noise: shape={perlin.shape}, dtype={perlin.dtype}")
        print(f"   Range: [{perlin.min():.2f}, {perlin.max():.2f}]")
    else:
        print(f"⚠️  Perlin noise skipped (install 'noise' package to enable)")

    # Test from config
    config = {
        'type': 'rainbow_grid',
        'size': [128, 128],
        'grid_sizes': [16, 32],
        'seed': 42
    }
    img = generate_from_config(config)
    print(f"✅ From config: shape={img.shape}")

    print("\n✨ All pattern tests passed!")