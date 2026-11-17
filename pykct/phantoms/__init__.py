#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phantom generation utilities for pykct.

This module provides functions for creating test phantoms (ball, Shepp-Logan)
and generating sinograms from them.

Main functions:
- create_ball_phantom: Create circular phantom with antialiasing
- create_multi_ball_phantom: Several (possibly overlapping) discs
- create_shepp_logan_phantom: Create Shepp-Logan phantom (denpy variants or legacy)
- create_legacy_shepp_logan_phantom: Non-standard phantom of pykct <= 0.1.0
- create_ball_sinogram: Project ball phantom with specified parameters
- create_shepp_logan_sinogram: Project Shepp-Logan phantom
- calculate_centroid: Calculate intensity-weighted centroid
"""

import numpy as np
from denpy import DEN
import os
import warnings
from typing import Optional, Tuple, Union


def create_circle(
    size: int,
    radius: float,
    x_center: Optional[float] = None,
    y_center: Optional[float] = None,
    antialias: int = 1
) -> np.ndarray:
    """
    Create a 2D circular phantom.
    
    Parameters
    ----------
    size : int
        Image width and height.
    radius : float
        Circle radius in pixels.
    x_center : float, optional
        X coordinate of circle center. Default: (size-1)/2
    y_center : float, optional
        Y coordinate of circle center. Default: (size-1)/2
    antialias : int
        Antialiasing factor. 1 = binary, >1 = supersampling.
    
    Returns
    -------
    np.ndarray
        2D image of shape (size, size) with dtype float32.
    
    Notes
    -----
    Uses geometric center at (size-1)/2, consistent with KCT convention.
    """
    if antialias < 1:
        raise ValueError("Antialias factor must be >= 1")
    
    # Geometric center
    if x_center is None:
        x_center = (size - 1) / 2.0
    if y_center is None:
        y_center = (size - 1) / 2.0
    
    # Pixel coordinates
    yy, xx = np.ogrid[:size, :size]
    
    if antialias == 1:
        # Binary circle
        distance_squared = (xx - x_center)**2 + (yy - y_center)**2
        return (distance_squared <= radius**2).astype(np.float32)
    
    # Supersampling antialiasing
    offsets = (np.arange(antialias) + 0.5) / antialias - 0.5
    
    image = np.zeros((size, size), dtype=np.float32)
    
    for dy in offsets:
        for dx in offsets:
            distance_squared = (xx + dx - x_center)**2 + (yy + dy - y_center)**2
            image += (distance_squared <= radius**2)
    
    image /= antialias**2
    
    return image


def create_ball_phantom(
    size: int = 256,
    radius: float = 10.0,
    x_center: Optional[float] = None,
    y_center: Optional[float] = None,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    antialias: bool = True,
    antialias_factor: int = 4
) -> np.ndarray:
    """
    Create a 2D circular phantom as a single-slice 3D volume.
    
    Parameters
    ----------
    size : int
        Image width and height (default: 256).
    radius : float
        Circle radius in pixels (default: 10.0).
    x_center : float, optional
        Absolute X coordinate. Default: image center.
    y_center : float, optional
        Absolute Y coordinate. Default: image center.
    x_offset : float
        X offset from the selected/default X coordinate (default: 0.0).
    y_offset : float
        Y offset from the selected/default Y coordinate (default: 0.0).
    antialias : bool
        Enable supersampling antialiasing (default: True).
    antialias_factor : int
        Number of subpixel samples per dimension (default: 4).
    
    Returns
    -------
    np.ndarray
        3D volume of shape (1, size, size) with dtype float32.
    
    Example
    -------
    >>> from pykct.phantoms import create_ball_phantom
    >>> 
    >>> # Centered ball
    >>> phantom = create_ball_phantom(size=128, radius=10)
    >>> 
    >>> # Off-center ball
    >>> phantom = create_ball_phantom(
    ...     size=128, radius=10, x_offset=20.0
    ... )
    """
    # Geometric center
    center = (size - 1) / 2.0
    
    # Absolute coordinates
    if x_center is None:
        x_center = center + x_offset
    else:
        x_center = x_center + x_offset
    
    if y_center is None:
        y_center = center + y_offset
    else:
        y_center = y_center + y_offset
    
    # Select antialiasing factor
    aa_factor = antialias_factor if antialias else 1
    
    # Create 2D circle
    image_2d = create_circle(
        size=size,
        radius=radius,
        x_center=x_center,
        y_center=y_center,
        antialias=aa_factor
    )
    
    # Convert to 3D volume (z, y, x)
    volume = image_2d[np.newaxis, :, :].astype(np.float32)
    
    return volume


def create_multi_ball_phantom(
    size: int = 256,
    balls=((0.0, 0.0, 10.0),),
    antialias: bool = True,
    antialias_factor: int = 4
) -> np.ndarray:
    """
    Create a 2D phantom composed of several discs as a single-slice 3D volume.

    Parameters
    ----------
    size : int
        Image width and height (default: 256).
    balls : iterable
        One entry per disc, either a tuple ``(x_offset, y_offset, radius)``
        or ``(x_offset, y_offset, radius, value)``, or a dict with keys
        ``x_offset``, ``y_offset``, ``radius`` and optional ``value``
        (default 1.0). Offsets are in pixels from the geometric centre
        (size-1)/2, as in :func:`create_ball_phantom`. Overlapping discs
        add up.
    antialias : bool
        Enable supersampling antialiasing (default: True).
    antialias_factor : int
        Number of subpixel samples per dimension (default: 4).

    Returns
    -------
    np.ndarray
        3D volume of shape (1, size, size) with dtype float32.

    Example
    -------
    >>> phantom = create_multi_ball_phantom(
    ...     size=256,
    ...     balls=[(0, 0, 20), (60, -30, 8, 0.5), {"x_offset": -70, "y_offset": 40, "radius": 5}],
    ... )
    """
    center = (size - 1) / 2.0
    aa_factor = antialias_factor if antialias else 1
    image = np.zeros((size, size), dtype=np.float32)
    for i, ball in enumerate(balls):
        if isinstance(ball, dict):
            x_offset = ball["x_offset"]
            y_offset = ball["y_offset"]
            radius = ball["radius"]
            value = ball.get("value", 1.0)
        else:
            if len(ball) not in (3, 4):
                raise ValueError(
                    f"Ball {i}: expected (x_offset, y_offset, radius[, value]), got {ball!r}"
                )
            x_offset, y_offset, radius = ball[:3]
            value = ball[3] if len(ball) == 4 else 1.0
        if radius <= 0:
            raise ValueError(f"Ball {i}: radius must be positive")
        image += value * create_circle(
            size=size,
            radius=radius,
            x_center=center + x_offset,
            y_center=center + y_offset,
            antialias=aa_factor
        )
    return image[np.newaxis, :, :]


#: Phantom names accepted by :func:`create_shepp_logan_phantom`.
#: All except ``"legacy"`` are taken from :func:`denpy.PHANTOM.phantom2d`.
SHEPP_LOGAN_VARIANTS = ("SheppLogan2D", "Octave2D", "Toft2D", "legacy")


def create_shepp_logan_phantom(
    size: int = 256,
    phantom: str = "SheppLogan2D",
    dtype=np.float32
) -> np.ndarray:
    """
    Create a 2D Shepp-Logan type phantom.

    Parameters
    ----------
    size : int
        Image width and height (default: 256).
    phantom : str
        Phantom variant, one of :data:`SHEPP_LOGAN_VARIANTS`:

        - ``"SheppLogan2D"`` (default): original Shepp & Logan (1974)
          specification, values in [0, 2].
        - ``"Octave2D"``: Octave/Matlab ``shepp_logan``, skull amplitude 1.0.
        - ``"Toft2D"``: Toft's modified Shepp-Logan (higher contrast).
        - ``"legacy"``: non-standard phantom produced by pykct <= 0.1.0,
          see :func:`create_legacy_shepp_logan_phantom`. Deprecated,
          subject to removal in a later version.
    dtype : numpy dtype
        Output data type (default: float32).

    Returns
    -------
    np.ndarray
        2D image of shape (size, size).

    Notes
    -----
    Standard variants are generated by :func:`denpy.PHANTOM.phantom2d`
    on the continuous domain [-1, 1] x [-1, 1] sampled at ``size`` points,
    so the phantom is centred at the geometric centre (size-1)/2.
    """
    if phantom == "legacy":
        return create_legacy_shepp_logan_phantom(size=size, _stacklevel=3).astype(dtype)
    if phantom not in SHEPP_LOGAN_VARIANTS:
        raise ValueError(
            f"Unknown phantom '{phantom}', expected one of {SHEPP_LOGAN_VARIANTS}"
        )
    from denpy import PHANTOM
    return PHANTOM.phantom2d(phantom=phantom, n=size).astype(dtype)


def create_legacy_shepp_logan_phantom(size: int = 256, _stacklevel: int = 2) -> np.ndarray:
    """
    Non-standard "Shepp-Logan" phantom of pykct <= 0.1.0.

    .. deprecated:: 0.1.0
        Subject to removal in a later version. Kept only for reproducing
        earlier experiments (e.g. ``sinogram_fft_analysis.ipynb`` in the
        sinal package). Use :func:`create_shepp_logan_phantom` for new work.
        Calling it emits a ``FutureWarning``.

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
        "removal in a later version; use create_shepp_logan_phantom(size, "
        "phantom='SheppLogan2D') for new work.",
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


def create_ball_sinogram(
    size: int = 256,
    radius: float = 10.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    n_angles: int = 512,
    start_angle: float = 0.0,
    angle_range: float = 360.0,
    detector_size: int = 256,
    cor_offset: float = 0.0,
    pixel_size: float = 1.0,
    output_file: Optional[str] = None,
    antialias: bool = True,
    antialias_factor: int = 4,
    platform_id: Optional[str] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Create sinogram from ball phantom.
    
    Convenience function that combines phantom creation and projection.
    
    Parameters
    ----------
    size : int
        Phantom size (default: 256).
    radius : float
        Ball radius in pixels (default: 10.0).
    x_offset : float
        X offset from center (default: 0.0).
    y_offset : float
        Y offset from center (default: 0.0).
    n_angles : int
        Number of projection angles (default: 512).
    start_angle : float
        Starting angle in degrees (default: 0.0).
    angle_range : float
        Angular range in degrees (default: 360.0).
    detector_size : int
        Detector size (default: 256).
    cor_offset : float
        Center of rotation offset in pixels (default: 0.0).
    pixel_size : float
        Pixel size (default: 1.0).
    output_file : str, optional
        Save sinogram to DEN file if specified.
    antialias : bool
        Enable antialiasing (default: True).
    antialias_factor : int
        Antialiasing factor (default: 4).
    platform_id : str, optional
        OpenCL "platform:device". Leave None to let the C++ runtime choose
        (recommended); see :class:`pykct.pb2d_projector`.
    verbose : bool
        Print progress information.
    
    Returns
    -------
    np.ndarray
        Sinogram of shape (1, n_angles, detector_size).
    
    Example
    -------
    >>> from pykct.phantoms import create_ball_sinogram
    >>> 
    >>> # Simple projection
    >>> sino = create_ball_sinogram(
    ...     size=128, radius=10, n_angles=360, detector_size=256
    ... )
    >>> 
    >>> # With CoR offset
    >>> sino = create_ball_sinogram(
    ...     size=128, radius=10, n_angles=360, detector_size=256,
    ...     cor_offset=5.0
    ... )
    """
    # Create phantom
    phantom = create_ball_phantom(
        size=size,
        radius=radius,
        x_offset=x_offset,
        y_offset=y_offset,
        antialias=antialias,
        antialias_factor=antialias_factor
    )
    
    # Project
    from pykct.core import project_image
    
    sinogram = project_image(
        image=phantom[0, :, :],  # Extract 2D image
        n_angles=n_angles,
        start_angle=start_angle,
        angle_range=angle_range,
        detector_sizex=detector_size,
        detector_sizey=1,
        cor_offset=cor_offset,
        pixel_size=pixel_size,
        platform_id=platform_id,
        verbose=verbose
    )
    
    # Save if requested
    if output_file is not None:
        DEN.storeNdarrayAsDEN(output_file, sinogram, force=True)
    
    return sinogram


def create_shepp_logan_sinogram(
    size: int = 256,
    n_angles: int = 512,
    start_angle: float = 0.0,
    angle_range: float = 360.0,
    detector_size: int = 256,
    cor_offset: float = 0.0,
    pixel_size: float = 1.0,
    output_file: Optional[str] = None,
    phantom: str = "SheppLogan2D",
    platform_id: Optional[str] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Create sinogram from Shepp-Logan phantom.
    
    Convenience function that combines phantom creation and projection.
    
    Parameters
    ----------
    size : int
        Phantom size (default: 256).
    n_angles : int
        Number of projection angles (default: 512).
    start_angle : float
        Starting angle in degrees (default: 0.0).
    angle_range : float
        Angular range in degrees (default: 360.0).
    detector_size : int
        Detector size (default: 256).
    cor_offset : float
        Center of rotation offset in pixels (default: 0.0).
    pixel_size : float
        Pixel size (default: 1.0).
    output_file : str, optional
        Save sinogram to DEN file if specified.
    phantom : str
        Phantom variant, see :func:`create_shepp_logan_phantom`
        (default: "SheppLogan2D").
    platform_id : str, optional
        OpenCL "platform:device". Leave None to let the C++ runtime choose
        (recommended); see :class:`pykct.pb2d_projector`.
    verbose : bool
        Print progress information.
    
    Returns
    -------
    np.ndarray
        Sinogram of shape (1, n_angles, detector_size).
    """
    image = create_shepp_logan_phantom(size=size, phantom=phantom)
    
    from pykct.core import project_image
    
    sinogram = project_image(
        image=image,
        n_angles=n_angles,
        start_angle=start_angle,
        angle_range=angle_range,
        detector_sizex=detector_size,
        detector_sizey=1,
        cor_offset=cor_offset,
        pixel_size=pixel_size,
        platform_id=platform_id,
        verbose=verbose
    )
    
    # Save if requested
    if output_file is not None:
        DEN.storeNdarrayAsDEN(output_file, sinogram, force=True)
    
    return sinogram


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


# Convenience alias for single ball phantom projection
def create_dot_sinogram(
    size: int = 256,
    n_angles: int = 512,
    detector_size: int = 256,
    cor_offset: float = 0.0,
    output_file: Optional[str] = None,
    platform_id: Optional[str] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Create sinogram from single dot (point source) phantom.
    
    Parameters
    ----------
    size : int
        Phantom size (default: 256).
    n_angles : int
        Number of projection angles (default: 512).
    detector_size : int
        Detector size (default: 256).
    cor_offset : float
        CoR offset in pixels (default: 0.0).
    output_file : str, optional
        Save sinogram to DEN file.
    platform_id : str, optional
        OpenCL "platform:device". Leave None to let the C++ runtime choose
        (recommended); see :class:`pykct.pb2d_projector`.
    verbose : bool
        Print progress information.
    
    Returns
    -------
    np.ndarray
        Sinogram of shape (1, n_angles, detector_size).
    
    Notes
    -----
    Creates a single point at the center of the volume.
    """
    return create_ball_sinogram(
        size=size,
        radius=0.5,  # Point-like
        n_angles=n_angles,
        detector_size=detector_size,
        cor_offset=cor_offset,
        output_file=output_file,
        antialias=False,
        platform_id=platform_id,
        verbose=verbose
    )
