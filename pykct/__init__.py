"""
PyKCT - Python bindings for KCT C++ Parallel Beam 2D Projector

This package provides a clean interface to the KCT C++ projector,
including phantom generation and convenience functions for creating
projections without manually constructing projection matrices.

Main functionality:
- pb2d_projector: Wrapper for kct-pb2d-projector binary
- PB2D: projection matrices of the KCT parallel ray geometry, 2x4 for
  kct-pb-* (orthogonal, skewed, tilted or general detector) and 1x4 for
  kct-pb2d-*; used by the automatic matrix creation of the projector
- Phantoms on the pixel grid: objects that draw themselves (shapes Circle,
  HollowCircle, Ellipse, Rectangle, Triangle, GaussianDot; image objects
  DenpyPhantom2D, ArrayImage), combined by create_phantom2d; Shepp-Logan
  variants (create_shepp_logan_phantom2d)
- Sinograms of drawn phantoms: create_sinogram2d (KCT projection)
- Centroid calculation: For CoR detection
"""

from . import PB2D
from .core import pb2d_projector
from .core import create_projection_matrices, project
from .core import project_image
from .phantoms import (
    PhantomObject,
    PhantomShape,
    PhantomImageObject,
    DenpyPhantom2D,
    ArrayImage,
    Circle,
    HollowCircle,
    Ellipse,
    Rectangle,
    Triangle,
    GaussianDot,
    SHEPP_LOGAN_VARIANTS,
    create_phantom2d,
    shepp_logan_shapes2d,
    create_shepp_logan_phantom2d,
    create_legacy_shepp_logan_phantom2d,
    create_sinogram2d,
    calculate_centroid,
)

__version__ = "0.2.0"

__all__ = [
    # Projection matrices of the parallel ray geometry
    "PB2D",

    # Core projection
    "pb2d_projector",
    "create_projection_matrices",
    "project",
    "project_image",

    # Phantom objects and phantoms
    "PhantomObject",
    "PhantomShape",
    "PhantomImageObject",
    "DenpyPhantom2D",
    "ArrayImage",
    "Circle",
    "HollowCircle",
    "Ellipse",
    "Rectangle",
    "Triangle",
    "GaussianDot",
    "SHEPP_LOGAN_VARIANTS",
    "create_phantom2d",
    "shepp_logan_shapes2d",
    "create_shepp_logan_phantom2d",
    "create_legacy_shepp_logan_phantom2d",

    # Sinograms
    "create_sinogram2d",

    # Utilities
    "calculate_centroid",
]
