# File: src/core/data_loader.py
"""
Data loading for optical flow tracking.

Handles both single frame pairs (experiments) and movie sequences.
Auto-detects input format and ground truth availability.
"""

import sys
import numpy as np
import cv2
import tomli
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Union
from PIL import Image


@dataclass
class FramePair:
    """
    Single frame pair with optional ground truth.
    
    Attributes:
        frame1, frame2: Grayscale frames (H, W) float32 in [0, 1]
        frame1_original, frame2_original: Original frames (may be RGB)
        u_truth, v_truth: Ground truth flow (None if unavailable)
        valid_mask: Valid pixel mask (H, W) bool
        has_gt: Whether ground truth is available
        metadata: Additional info (H, W, boundary_margin, etc.)
    """
    frame1: np.ndarray
    frame2: np.ndarray
    frame1_original: np.ndarray
    frame2_original: np.ndarray
    u_truth: Optional[np.ndarray]
    v_truth: Optional[np.ndarray]
    valid_mask: np.ndarray
    has_gt: bool
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate frame pair data."""
        H, W = self.frame1.shape[:2]
        
        # Validate frame shapes
        if self.frame2.shape[:2] != (H, W):
            print(f"❌ ERROR: Frame shape mismatch: {self.frame1.shape} vs {self.frame2.shape}")
            sys.exit(1)
        
        # Validate GT if present
        if self.has_gt:
            if self.u_truth is None or self.v_truth is None:
                print(f"❌ ERROR: has_gt=True but ground truth is None")
                sys.exit(1)
            if self.u_truth.shape != (H, W) or self.v_truth.shape != (H, W):
                print(f"❌ ERROR: GT shape mismatch: expected ({H}, {W})")
                sys.exit(1)
        
        # Validate valid_mask
        if self.valid_mask.shape != (H, W):
            print(f"❌ ERROR: valid_mask shape mismatch: expected ({H}, {W})")
            sys.exit(1)
        
        # Ensure metadata has basic info
        if 'H' not in self.metadata:
            self.metadata['H'] = H
        if 'W' not in self.metadata:
            self.metadata['W'] = W


@dataclass
class MovieSequence:
    """
    Sequence of frame pairs from a movie.
    
    Attributes:
        pairs: List of consecutive frame pairs
        config: Configuration dict from TOML
        hash: Directory hash identifier
        metadata: Additional movie-level info
    """
    pairs: List[FramePair]
    config: dict
    hash: str
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate movie sequence."""
        if len(self.pairs) == 0:
            print(f"❌ ERROR: MovieSequence has no frame pairs")
            sys.exit(1)
        
        # Ensure metadata has basic info
        if 'n_pairs' not in self.metadata:
            self.metadata['n_pairs'] = len(self.pairs)
        if 'n_frames' not in self.metadata:
            self.metadata['n_frames'] = len(self.pairs) + 1


def compute_valid_mask(H: int, W: int, boundary_margin: int) -> np.ndarray:
    """
    Compute valid pixel mask based on boundary margin.
    
    Args:
        H, W: Image dimensions
        boundary_margin: Margin in pixels to exclude from edges
        
    Returns:
        Boolean mask (True = valid pixel)
    """
    valid_mask = np.ones((H, W), dtype=bool)
    if boundary_margin > 0:
        valid_mask[:boundary_margin, :] = False
        valid_mask[-boundary_margin:, :] = False
        valid_mask[:, :boundary_margin] = False
        valid_mask[:, -boundary_margin:] = False
    return valid_mask


def load_experiment_pair(exp_dir: Path, boundary_margin: int = None) -> FramePair:
    """
    Load single frame pair from experiment directory.
    
    Expected structure:
        exp_dir/
            frame1.npy, frame2.npy      # Input frames
            u_truth.npy, v_truth.npy    # Ground truth (optional)
            config.toml                 # Config (optional)
    
    Args:
        exp_dir: Path to experiment directory
        boundary_margin: Override boundary margin (if None, read from config or use 0)
    
    Returns:
        FramePair with loaded data
    """
    
    if not exp_dir.exists():
        print(f"❌ ERROR: Experiment directory not found: {exp_dir}")
        sys.exit(1)
    
    print(f"📂 Loading experiment pair from {exp_dir.name}/")
    
    # ========================================================================
    # Load config if available
    # ========================================================================
    
    config_path = exp_dir / 'config.toml'
    config = {}
    if config_path.exists():
        with open(config_path, 'rb') as f:
            config = tomli.load(f)
        print(f"   ✅ Loaded config.toml")
    
    # Get boundary margin from config or argument
    if boundary_margin is None:
        boundary_margin = config.get('evaluation', {}).get('boundary_margin', 0)
    
    # ========================================================================
    # Load frames
    # ========================================================================
    
    frame1_path = exp_dir / 'frame1.npy'
    frame2_path = exp_dir / 'frame2.npy'
    
    if not frame1_path.exists() or not frame2_path.exists():
        print(f"❌ ERROR: Missing frames: {frame1_path} or {frame2_path}")
        sys.exit(1)
    
    frame1 = np.load(frame1_path)
    frame2 = np.load(frame2_path)
    
    # Store originals
    frame1_original = frame1.copy()
    frame2_original = frame2.copy()
    
    # Convert to grayscale if needed
    if len(frame1.shape) == 3:
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    else:
        frame1_gray = frame1.copy()
        frame2_gray = frame2.copy()
    
    # Ensure float32 in [0, 1]
    if frame1_gray.dtype != np.float32:
        frame1_gray = frame1_gray.astype(np.float32)
        if frame1_gray.max() > 1.0:
            frame1_gray = frame1_gray / 255.0
    
    if frame2_gray.dtype != np.float32:
        frame2_gray = frame2_gray.astype(np.float32)
        if frame2_gray.max() > 1.0:
            frame2_gray = frame2_gray / 255.0
    
    H, W = frame1_gray.shape
    
    print(f"   ✅ Loaded frames: {H}×{W}")
    
    # ========================================================================
    # Load ground truth if available
    # ========================================================================
    
    u_truth_path = exp_dir / 'u_truth.npy'
    v_truth_path = exp_dir / 'v_truth.npy'
    
    has_gt = u_truth_path.exists() and v_truth_path.exists()
    
    if has_gt:
        u_truth = np.load(u_truth_path)
        v_truth = np.load(v_truth_path)
        print(f"   ✅ Ground truth: Available")
    else:
        u_truth = None
        v_truth = None
        print(f"   ⚠️  Ground truth: Not available")
    
    # ========================================================================
    # Compute valid mask
    # ========================================================================
    
    valid_mask = compute_valid_mask(H, W, boundary_margin)
    
    # Also exclude pixels where GT is NaN (e.g., Middlebury unknown regions)
    if u_truth is not None:
        gt_valid = ~(np.isnan(u_truth) | np.isnan(v_truth))
        valid_mask = valid_mask & gt_valid
    
    valid_pct = 100 * valid_mask.sum() / (H * W)
    
    print(f"   ✅ Valid mask: {boundary_margin}px margin ({valid_pct:.1f}% valid)")
    print()
    
    # ========================================================================
    # Create FramePair
    # ========================================================================
    
    metadata = {
        'H': H,
        'W': W,
        'boundary_margin': boundary_margin,
        'source': 'experiment',
        'path': str(exp_dir)
    }
    
    return FramePair(
        frame1=frame1_gray,
        frame2=frame2_gray,
        frame1_original=frame1_original,
        frame2_original=frame2_original,
        u_truth=u_truth,
        v_truth=v_truth,
        valid_mask=valid_mask,
        has_gt=has_gt,
        metadata=metadata
    )


def load_movie_sequence(movie_dir: Path, boundary_margin: int = 15) -> MovieSequence:
    """
    Load movie sequence from sprite generator output.
    
    Expected structure:
        movie_dir/
            config.toml
            frames/
                image_000.png ... image_009.png
                u_000.npz ... u_008.npz  (contains 'u', 'valid')
                v_000.npz ... v_008.npz  (contains 'v', 'valid')
    
    Args:
        movie_dir: Path to movie directory
        boundary_margin: Boundary margin for valid mask
    
    Returns:
        MovieSequence with N-1 frame pairs for N frames
    """
    
    if not movie_dir.exists():
        print(f"❌ ERROR: Movie directory not found: {movie_dir}")
        sys.exit(1)
    
    print(f"🎬 Loading movie sequence from {movie_dir.name}/")
    
    # ========================================================================
    # Load config
    # ========================================================================
    
    config_path = movie_dir / 'config.toml'
    if not config_path.exists():
        print(f"❌ ERROR: config.toml not found in {movie_dir}")
        sys.exit(1)
    
    with open(config_path, 'rb') as f:
        config = tomli.load(f)
    
    print(f"   ✅ Loaded config.toml")
    
    # ========================================================================
    # Find all frames
    # ========================================================================
    
    frames_dir = movie_dir / 'frames'
    if not frames_dir.exists():
        print(f"❌ ERROR: frames/ directory not found in {movie_dir}")
        sys.exit(1)
    
    # Find all image files
    image_files = sorted(frames_dir.glob('image_*.png'))
    n_frames = len(image_files)
    
    if n_frames == 0:
        print(f"❌ ERROR: No image files found in {frames_dir}")
        sys.exit(1)
    
    print(f"   ✅ Found {n_frames} frames")
    
    # ========================================================================
    # Load all frames and create pairs
    # ========================================================================
    
    pairs = []
    
    for i in range(n_frames - 1):
        # Load consecutive frame images
        img1_path = frames_dir / f'image_{i:03d}.png'
        img2_path = frames_dir / f'image_{i+1:03d}.png'
        
        if not img1_path.exists() or not img2_path.exists():
            print(f"❌ ERROR: Missing frame images: {img1_path} or {img2_path}")
            sys.exit(1)
        
        img1 = np.array(Image.open(img1_path))
        img2 = np.array(Image.open(img2_path))
        
        # Store originals
        frame1_original = img1.copy()
        frame2_original = img2.copy()
        
        # Convert to grayscale
        if len(img1.shape) == 3:
            frame1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            frame2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            frame1 = img1.copy()
            frame2 = img2.copy()
        
        # Convert to float32 in [0, 1]
        frame1 = frame1.astype(np.float32) / 255.0
        frame2 = frame2.astype(np.float32) / 255.0
        
        H, W = frame1.shape
        
        # Load ground truth flow
        u_path = frames_dir / f'u_{i:03d}.npz'
        v_path = frames_dir / f'v_{i:03d}.npz'
        
        has_gt = u_path.exists() and v_path.exists()
        
        if has_gt:
            u_data = np.load(u_path)
            v_data = np.load(v_path)
            u_truth = u_data['u']
            v_truth = v_data['v']
            # Note: 'valid' key exists in npz but we compute our own valid_mask
        else:
            u_truth = None
            v_truth = None
        
        # Compute valid mask
        valid_mask = compute_valid_mask(H, W, boundary_margin)
        
        # Also exclude pixels where GT is NaN (e.g., Middlebury unknown regions)
        if u_truth is not None:
            gt_valid = ~(np.isnan(u_truth) | np.isnan(v_truth))
            valid_mask = valid_mask & gt_valid
        
        # Create metadata
        metadata = {
            'H': H,
            'W': W,
            'boundary_margin': boundary_margin,
            'source': 'movie',
            'frame_index': i,
            'path': str(movie_dir)
        }
        
        # Create FramePair
        pair = FramePair(
            frame1=frame1,
            frame2=frame2,
            frame1_original=frame1_original,
            frame2_original=frame2_original,
            u_truth=u_truth,
            v_truth=v_truth,
            valid_mask=valid_mask,
            has_gt=has_gt,
            metadata=metadata
        )
        
        pairs.append(pair)
    
    print(f"   ✅ Created {len(pairs)} frame pairs")
    print()
    
    # ========================================================================
    # Create MovieSequence
    # ========================================================================
    
    movie_hash = movie_dir.name
    
    movie_metadata = {
        'n_frames': n_frames,
        'n_pairs': len(pairs),
        'has_gt': all(p.has_gt for p in pairs),
        'path': str(movie_dir)
    }
    
    return MovieSequence(
        pairs=pairs,
        config=config,
        hash=movie_hash,
        metadata=movie_metadata
    )


def load_data(input_path: Path, boundary_margin: int = None) -> Union[FramePair, MovieSequence]:
    """
    Auto-detect and load data from path.
    
    Detects whether input is:
    - Experiment directory (single frame pair)
    - Movie directory (frame sequence)
    
    Args:
        input_path: Path to input directory
        boundary_margin: Boundary margin for valid mask (if None, auto-detect)
    
    Returns:
        FramePair or MovieSequence depending on input type
    """
    
    if not input_path.exists():
        print(f"❌ ERROR: Input path not found: {input_path}")
        sys.exit(1)
    
    if not input_path.is_dir():
        print(f"❌ ERROR: Input path is not a directory: {input_path}")
        sys.exit(1)
    
    # ========================================================================
    # Detect input type
    # ========================================================================
    
    # Check for movie structure (frames/ subdirectory)
    if (input_path / 'frames').exists():
        print(f"🎬 Detected: Movie sequence")
        if boundary_margin is None:
            boundary_margin = 15  # Default for movies
        return load_movie_sequence(input_path, boundary_margin)
    
    # Check for experiment structure (frame1.npy exists)
    elif (input_path / 'frame1.npy').exists():
        print(f"📂 Detected: Experiment pair")
        return load_experiment_pair(input_path, boundary_margin)
    
    else:
        print(f"❌ ERROR: Unrecognized directory structure: {input_path}")
        print(f"   Expected either:")
        print(f"   - Movie: {input_path}/frames/image_*.png")
        print(f"   - Experiment: {input_path}/frame1.npy")
        sys.exit(1)


if __name__ == "__main__":
    print("✅ Data loader module loaded")
    
    # Test dataclass validation
    print("\n🧪 Testing FramePair validation...")
    
    H, W = 100, 100
    frame1 = np.zeros((H, W), dtype=np.float32)
    frame2 = np.zeros((H, W), dtype=np.float32)
    u_truth = np.zeros((H, W), dtype=np.float32)
    v_truth = np.zeros((H, W), dtype=np.float32)
    valid_mask = np.ones((H, W), dtype=bool)
    
    pair = FramePair(
        frame1=frame1,
        frame2=frame2,
        frame1_original=frame1,
        frame2_original=frame2,
        u_truth=u_truth,
        v_truth=v_truth,
        valid_mask=valid_mask,
        has_gt=True
    )
    
    print(f"   ✅ FramePair created: {pair.metadata['H']}×{pair.metadata['W']}, GT={pair.has_gt}")
    
    # Test without GT
    pair_no_gt = FramePair(
        frame1=frame1,
        frame2=frame2,
        frame1_original=frame1,
        frame2_original=frame2,
        u_truth=None,
        v_truth=None,
        valid_mask=valid_mask,
        has_gt=False
    )
    
    print(f"   ✅ FramePair without GT: {pair_no_gt.metadata['H']}×{pair_no_gt.metadata['W']}, GT={pair_no_gt.has_gt}")
    
    print("\n✨ All data loader tests passed!")
