"""
PyKCT - Python bindings for KCT C++ Parallel Beam 2D Projector

This package provides a clean interface to the KCT C++ projector,
including phantom generation and convenience functions for creating
projections without manually constructing projection matrices.

Main functionality:
- PB2D: wrappers of the 2D parallel beam tools, pb2d_projector (projection),
  pb2d_backprojector (backprojection, FBP) and pb2d_krylov (CGLS, PDHG)
- PM: projection matrices of the KCT parallel ray geometry, 2x4 for
  kct-pb-* (orthogonal, skewed, tilted or general detector) and 1x4 for
  kct-pb2d-*; used by the automatic matrix creation of the PB2D tools
- Phantoms on the pixel grid: objects that draw themselves (shapes Circle,
  HollowCircle, Ellipse, Rectangle, Triangle, GaussianDot; image objects
  DenpyPhantom2D, ArrayImage), combined by create_phantom2d; Shepp-Logan
  variants (create_shepp_logan_phantom2d)
- Sinograms of drawn phantoms: create_sinogram2d (KCT projection)
- Centroid calculation: For CoR detection
"""

from . import PM
from . import PB2D
from .PB2D import pb2d_projector, pb2d_backprojector, pb2d_krylov
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
    "PM",

    # 2D parallel beam tools
    "PB2D",
    "pb2d_projector",
    "pb2d_backprojector",
    "pb2d_krylov",

    # Core projection
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
