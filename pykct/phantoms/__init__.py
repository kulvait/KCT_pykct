#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phantoms on the pixel grid and their sinograms.

Phantom objects (:mod:`pykct.phantoms.objects2d`) draw themselves on a
``size`` x ``size`` pixel grid; :func:`create_phantom2d` adds a list of them
on one canvas and :func:`create_sinogram2d` projects the result with the KCT
projector. Every sinogram therefore comes from a discretized image.

Objects
-------
* shapes given by a formula (:class:`PhantomShape`): :class:`Circle`,
  :class:`HollowCircle`, :class:`Ellipse`, :class:`Rectangle`,
  :class:`Triangle`, :class:`GaussianDot`
* objects given as an image (:class:`PhantomImageObject`):
  :class:`DenpyPhantom2D`, :class:`ArrayImage`

All have ``x_center``, ``y_center`` (None: image centre (size-1)/2),
``x_offset``, ``y_offset`` and ``value``, lengths in grid pixels, and the
methods ``draw(size, antialias_factor=1, supersampling_factor=1)`` and
``add_to(canvas, size, ...)``. See :mod:`pykct.phantoms.objects2d` for the
coordinate conventions.

Discretization
--------------
``supersampling_factor`` k: the image has size*k pixels per side, every
pixel is 1/k of a ``size`` pixel and all objects are scaled by k, so the
phantom covers the same region for every k.
``antialias_factor`` a: every output pixel is the mean of a x a samples on
a regular sub-grid (otherwise the objects are sampled at the pixel centre).

Main functions
--------------
* :func:`create_phantom2d`             objects -> 2D image (size*k, size*k)
* :func:`shepp_logan_shapes2d`         Shepp-Logan variants as Ellipse shapes
* :func:`create_shepp_logan_phantom2d` Shepp-Logan image (= denpy.PHANTOM.phantom2d)
* :func:`create_legacy_shepp_logan_phantom2d` deprecated pykct <= 0.1.0 phantom
* :func:`create_sinogram2d`            objects -> image -> KCT sinogram
* :func:`calculate_centroid`           intensity-weighted centroid of a frame
"""

import os
import warnings
from typing import Optional, Tuple, Union

import numpy as np
from denpy import DEN

from .objects2d import (ArrayImage, Circle, DenpyPhantom2D, Ellipse, GaussianDot, HollowCircle,
                        PhantomImageObject, PhantomObject, PhantomShape, Rectangle, Triangle)

__all__ = [
    "PhantomObject", "PhantomShape", "PhantomImageObject",
    "Circle", "HollowCircle", "Ellipse", "Rectangle", "Triangle", "GaussianDot",
    "DenpyPhantom2D", "ArrayImage", "SHEPP_LOGAN_VARIANTS",
    "create_phantom2d", "shepp_logan_shapes2d", "create_shepp_logan_phantom2d",
    "create_legacy_shepp_logan_phantom2d", "create_sinogram2d", "calculate_centroid",
]


def _get_factor_size(enabled, factor, name):
    """Factor of an (enable switch, factor) pair; refuse silent no-ops.

    Returns ``factor`` if the switch is on (it must then be >= 2), 1 if it is
    off (the factor must then be left at 1).
    """
    if isinstance(factor, bool) or float(factor) != int(factor) or int(factor) < 1:
        raise ValueError(f"{name}_factor must be an integer >= 1, got {factor!r}")
    factor = int(factor)
    if enabled:
        if factor < 2:
            raise ValueError(f"{name}=True needs {name}_factor >= 2, got {factor}")
        return factor
    if factor != 1:
        raise ValueError(f"{name}_factor={factor} given but {name}=False; "
                         f"set {name}=True to use it")
    return 1


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def create_phantom2d(
    size: int,
    objects,
    antialias: bool = False,
    antialias_factor: int = 1,
    supersampling: bool = False,
    supersampling_factor: int = 1,
    dtype=np.float32,
) -> np.ndarray:
    """
    Draw phantom objects on a 2D pixel grid.

    Parameters
    ----------
    size : int
        Grid size in pixels; all shape parameters are in these pixels.
    objects : PhantomObject or iterable of PhantomObject
        Shapes (:class:`Circle`, :class:`HollowCircle`, :class:`Ellipse`,
        :class:`Rectangle`, :class:`Triangle`, :class:`GaussianDot`) and image
        objects (:class:`DenpyPhantom2D`, :class:`ArrayImage`); each is added
        to the canvas by its ``add_to`` method, values of overlapping objects
        add up.
    antialias, antialias_factor : bool, int
        If enabled, each output pixel is the mean of antialias_factor^2
        samples on a regular sub-grid (factor >= 2); otherwise each shape is
        sampled at the pixel centre.
    supersampling, supersampling_factor : bool, int
        If enabled, the output has size * supersampling_factor pixels per
        side and every shape parameter is scaled by the factor: the same
        phantom on a finer grid (factor >= 2).
    dtype : numpy dtype
        Output dtype (default float32).

    Returns
    -------
    np.ndarray
        2D image of shape (size*k, size*k), k = supersampling factor.

    Example
    -------
    >>> from pykct.phantoms import create_phantom2d, Circle, HollowCircle, Rectangle
    >>> image = create_phantom2d(256, [
    ...     HollowCircle(radius=100, inner_radius=90),
    ...     Circle(radius=10, x_offset=40, value=0.5),
    ...     Rectangle(width=30, height=10, angle_deg=30, y_offset=-50),
    ... ], antialias=True, antialias_factor=4)
    """
    size = int(size)
    if size < 1:
        raise ValueError(f"size must be positive, got {size}")
    antialias_factor = _get_factor_size(antialias, antialias_factor, "antialias")
    supersampling_factor = _get_factor_size(supersampling, supersampling_factor, "supersampling")
    if isinstance(objects, PhantomObject):
        objects = [objects]
    objects = list(objects)
    for obj in objects:
        if not isinstance(obj, PhantomObject):
            raise TypeError(f"Expected PhantomObject instances (Circle, Ellipse, ...), got {obj!r}")
    n = size * supersampling_factor
    image = np.zeros((n, n), dtype=np.float64)
    for obj in objects:
        obj.add_to(image, size, antialias_factor=antialias_factor,
                   supersampling_factor=supersampling_factor)
    return image.astype(dtype, copy=False)


# ---------------------------------------------------------------------------
# Shepp-Logan
# ---------------------------------------------------------------------------

#: Variants accepted by :func:`shepp_logan_shapes2d` (denpy.PHANTOM definitions).
SHEPP_LOGAN_VARIANTS = ("SheppLogan2D", "Octave2D", "Toft2D")


def shepp_logan_shapes2d(size: int, phantom: str = "SheppLogan2D"):
    """
    Shepp-Logan type phantom as :class:`Ellipse` shapes on a size x size grid.

    The denpy.PHANTOM specification on [-1, 1]^2 is mapped exactly as
    :func:`denpy.PHANTOM.phantom2d` samples it: x = -1 at column 0, x = +1 at
    column size-1, y = +1 at row 0 (the head is upright on screen).

    Parameters
    ----------
    size : int
        Grid size in pixels.
    phantom : str
        One of :data:`SHEPP_LOGAN_VARIANTS`: "SheppLogan2D" (original 1974
        specification, values in [0, 2]), "Octave2D" (skull amplitude 1),
        "Toft2D" (Toft's modified, higher contrast).
    """
    if phantom not in SHEPP_LOGAN_VARIANTS:
        raise ValueError(f"Unknown phantom '{phantom}', expected one of {SHEPP_LOGAN_VARIANTS}")
    from denpy import PHANTOM
    half = 0.5 * (size - 1)
    shapes = []
    for e in getattr(PHANTOM, phantom)():
        # rows point down: y_pixel = centre - y * half, so the tilt changes sign
        shapes.append(Ellipse(radius_x=e.a * half, radius_y=e.b * half, angle_deg=-e.phi,
                              x_offset=e.x0 * half, y_offset=-e.y0 * half, value=e.A))
    return shapes


def create_shepp_logan_phantom2d(
    size: int = 256,
    phantom: str = "SheppLogan2D",
    antialias: bool = False,
    antialias_factor: int = 1,
    supersampling: bool = False,
    supersampling_factor: int = 1,
    dtype=np.float32,
) -> np.ndarray:
    """
    Shepp-Logan type 2D phantom, see :func:`shepp_logan_shapes2d`.

    Without antialiasing and supersampling the image equals
    ``denpy.PHANTOM.phantom2d(phantom, size)``. With supersampling the
    ellipses are scaled with the grid, so the phantom covers the same region
    for every factor.
    """
    return create_phantom2d(size, shepp_logan_shapes2d(size, phantom), antialias=antialias,
                            antialias_factor=antialias_factor, supersampling=supersampling,
                            supersampling_factor=supersampling_factor, dtype=dtype)


def create_legacy_shepp_logan_phantom2d(size: int = 256, _stacklevel: int = 2) -> np.ndarray:
    """
    Non-standard "Shepp-Logan" phantom of pykct <= 0.1.0.

    .. deprecated:: 0.1.0
        Subject to removal. Kept only for reproducing earlier experiments
        (e.g. ``sinogram_fft_analysis.ipynb``). Use
        :func:`create_shepp_logan_phantom2d` for new work. Calling it emits a
        ``FutureWarning``.

    It deviates from every standard Shepp-Logan definition:

    - semi-axes a/b are swapped, so the head is wider than tall,
    - no ellipse rotations (the -18/+18 degree ventricles are axis aligned),
    - ellipse positions and sizes do not match the published table,
    - Toft-like amplitudes with wrong signs (max value 1.2 instead of 1.0),
    - centred at size/2 instead of the geometric centre (size-1)/2,
      i.e. shifted by +0.5 px in x and y.

    Returns
    -------
    np.ndarray
        2D float32 image of shape (size, size), bit-identical to the
        output of ``pykct.phantoms.create_shepp_logan_phantom`` in 0.1.0.
    """
    warnings.warn(
        "The legacy pykct Shepp-Logan phantom is non-standard and subject to "
        "removal in a later version; use create_shepp_logan_phantom2d(size) for new work.",
        FutureWarning,
        stacklevel=_stacklevel,
    )
    ny, nx = size, size
    phantom = np.zeros((ny, nx), dtype=np.float32)

    y, x = np.ogrid[:ny, :nx]
    cx, cy = nx / 2.0, ny / 2.0

    # Ellipses: (x_offset, y_offset, rx, ry, amplitude)
    ellipses = [
        (0, 0, 0.92 * ny / 2, 0.69 * nx / 2, 1.0),
        (0, -0.01 * ny / 2, 0.31 * ny / 2, 0.35 * nx / 2, -0.8),
        (-0.22 * nx / 2, 0, 0.33 * ny / 2, 0.21 * nx / 2, 0.2),
        (0.22 * nx / 2, 0, 0.21 * ny / 2, 0.25 * nx / 2, 0.1),
        (0, 0.35 * ny / 2, 0.21 * ny / 2, 0.04 * nx / 2, 0.1),
        (0, -0.35 * ny / 2, 0.21 * ny / 2, 0.04 * nx / 2, 0.1),
        (-0.08 * nx / 2, 0.65 * ny / 2, 0.04 * ny / 2, 0.01 * nx / 2, 0.1),
        (0, -0.65 * ny / 2, 0.06 * ny / 2, 0.02 * nx / 2, 0.1),
        (-0.12 * nx / 2, -0.6 * ny / 2, 0.08 * ny / 2, 0.016 * nx / 2, 0.1),
        (0.12 * nx / 2, 0.6 * ny / 2, 0.06 * ny / 2, 0.022 * nx / 2, 0.1),
    ]

    for ex, ey, rx, ry, amp in ellipses:
        mask = ((x - cx - ex) / rx)**2 + ((y - cy - ey) / ry)**2 <= 1
        phantom[mask] += amp

    return phantom


# ---------------------------------------------------------------------------
# sinograms
# ---------------------------------------------------------------------------

def create_sinogram2d(
    size: int,
    objects,
    n_angles: int = 512,
    start_angle: float = 0.0,
    angle_range: float = 360.0,
    detector_size: Optional[int] = None,
    cor_offset: float = 0.0,
    antialias: bool = False,
    antialias_factor: int = 1,
    supersampling: bool = False,
    supersampling_factor: int = 1,
    output_file: Optional[str] = None,
    platform_id: Optional[str] = None,
    verbose: bool = False,
    return_phantom: bool = False,
):
    """
    Draw shapes with :func:`create_phantom2d` and project them with KCT.

    Geometry is in pixels of the ``size`` grid, whatever the supersampling:
    a detector pixel is one grid pixel, the rotation axis is the image
    centre and is imaged at detector pixel (detector_size-1)/2 + cor_offset.
    With supersampling factor k the image has k x k voxels per grid pixel,
    it is projected with detector pixel size k voxels (projection matrices
    of :func:`pykct.project_image` with ``pixel_size=k``), and the line
    integrals are divided by k, so they are in grid pixel units for every k.

    Parameters
    ----------
    size, objects, antialias, antialias_factor, supersampling, supersampling_factor
        See :func:`create_phantom2d`.
    n_angles, start_angle, angle_range : int, float, float
        Angles start_angle + i * angle_range / n_angles in degrees, endpoint
        excluded.
    detector_size : int, optional
        Number of detector pixels, default ``size``.
    cor_offset : float
        Rotation axis offset from the detector centre in pixels.
    output_file : str, optional
        Store the sinogram as a DEN file of shape (1, n_angles, detector_size).
    platform_id : str, optional
        OpenCL "platform:device"; leave None to let the C++ runtime choose
        (recommended), see :class:`pykct.pb2d_projector`.
    verbose : bool
        Print progress information.
    return_phantom : bool
        Also return the drawn image.

    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
        Sinogram (n_angles, detector_size) float32; with ``return_phantom``
        also the image (size*k, size*k).
    """
    from pykct.core import project_image

    supersampling_factor = _get_factor_size(supersampling, supersampling_factor, "supersampling")
    image = create_phantom2d(size, objects, antialias=antialias,
                             antialias_factor=antialias_factor, supersampling=supersampling,
                             supersampling_factor=supersampling_factor)
    if detector_size is None:
        detector_size = int(size)
    sinogram = project_image(image, n_angles=n_angles, start_angle=start_angle,
                             angle_range=angle_range, detector_sizex=int(detector_size),
                             detector_sizey=1, cor_offset=cor_offset,
                             pixel_size=float(supersampling_factor),   # detector pixel = k voxels
                             platform_id=platform_id, verbose=verbose)[0]
    sinogram = (np.asarray(sinogram, dtype=np.float32)
                / np.float32(supersampling_factor)).astype(np.float32)
    if output_file is not None:
        DEN.storeNdarrayAsDEN(output_file, sinogram[np.newaxis], force=True)
    return (sinogram, image) if return_phantom else sinogram


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def calculate_centroid(
    data: Union[str, np.ndarray],
    threshold: Optional[float] = None
) -> Optional[Tuple[float, float]]:
    """
    Calculate intensity-weighted centroid of a 2D frame.

    Parameters
    ----------
    data : np.ndarray or str
        2D frame, or path to a DEN file (its first frame is used).
        A single-frame 3D array of shape (1, ny, nx) is accepted too.
    threshold : float, optional
        Ignore pixels with intensity below this threshold.

    Returns
    -------
    tuple or None
        (x_centroid, y_centroid), or None if the frame has zero mass.

    Notes
    -----
    Uses 0-indexed pixel coordinates (consistent with NumPy), so the
    geometric centre of an N-pixel axis is (N-1)/2.

    Applied to a full 360 degree parallel-beam sinogram (angles x detector),
    the x centroid is the detector position of the rotation axis, i.e.
    x_centroid - (n_det-1)/2 == cor_offset in the pykct convention.
    """
    if isinstance(data, (str, os.PathLike)):
        frame = DEN.getFrame(str(data), 0)
    else:
        frame = np.asarray(data)
        if frame.ndim == 3 and frame.shape[0] == 1:
            frame = frame[0]
    if frame.ndim != 2:
        raise ValueError(f"Expected a 2D frame, got shape {frame.shape}")

    frame = frame.astype(np.float64)

    if threshold is not None:
        frame = np.where(frame >= threshold, frame, 0.0)

    total_mass = np.sum(frame)

    if np.isclose(total_mass, 0):
        return None

    yy, xx = np.indices(frame.shape)

    x_centroid = np.sum(xx * frame) / total_mass
    y_centroid = np.sum(yy * frame) / total_mass

    return float(x_centroid), float(y_centroid)
