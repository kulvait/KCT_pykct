"""
PyKCT - Python bindings for KCT C++ Parallel Beam 2D Projector

This package provides a clean interface to the KCT C++ projector,
including phantom generation and convenience functions for creating
projections without manually constructing projection matrices.

Main functionality:
- pb2d_projector: Wrapper for kct-pb2d-projector binary
- Geometry utilities: Automatic projection matrix creation
- Phantom generation: Ball, Shepp-Logan phantoms
- Sinogram generation: Direct projection from images
- Centroid calculation: For CoR detection
"""

from .core import pb2d_projector
from .core import create_projection_matrices, project
from .core import project_image
from .phantoms import (
    create_ball_phantom,
    create_shepp_logan_phantom,
    create_ball_sinogram,
    create_shepp_logan_sinogram,
)
from .phantoms import calculate_centroid

__version__ = "0.1.0"

__all__ = [
    # Core projection
    "pb2d_projector",
    "create_projection_matrices",
    "project",
    "project_image",
    
    # Phantom generation
    "create_ball_phantom",
    "create_shepp_logan_phantom",
    "create_ball_sinogram",
    "create_shepp_logan_sinogram",
    
    # Utilities
    "calculate_centroid",
]
