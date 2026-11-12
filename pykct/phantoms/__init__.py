#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phantom generation utilities for pykct.

This module provides functions for creating test phantoms (ball, Shepp-Logan)
and generating sinograms from them.

Main functions:
- create_ball_phantom: Create circular phantom with antialiasing
- create_shepp_logan_phantom: Create Shepp-Logan phantom
- create_ball_sinogram: Project ball phantom with specified parameters
- create_shepp_logan_sinogram: Project Shepp-Logan phantom
- calculate_centroid: Calculate intensity-weighted centroid
"""

import numpy as np
from denpy import DEN
import os
from typing import Optional, Tuple, Union
from scipy import special


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


def create_shepp_logan_phantom(
    size: int = 256,
    antialias: bool = False
) -> np.ndarray:
    """
    Create a Shepp-Logan phantom.
    
    Parameters
    ----------
    size : int
        Image width and height (default: 256).
    antialias : bool
        Enable antialiasing (default: False).
    
    Returns
    -------
    np.ndarray
        2D image of shape (size, size) with dtype float32.
    
    Notes
    -----
    Standard Shepp-Logan phantom with 10 ellipses.
    """
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
    
    if antialias:
        # Apply simple antialiasing (can be improved)
        pass
    
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
        OpenCL platform ID.
    verbose : bool
        Print progress information.
    
    Returns
    -------
    np.ndarray
        Sinogram of shape (n_angles, 1, detector_size).
    
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
    antialias: bool = False,
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
    antialias : bool
        Enable antialiasing (default: False).
    platform_id : str, optional
        OpenCL platform ID.
    verbose : bool
        Print progress information.
    
    Returns
    -------
    np.ndarray
        Sinogram of shape (n_angles, 1, detector_size).
    """
    # Create phantom
    phantom = create_shepp_logan_phantom(size=size, antialias=antialias)
    
    # Project
    from pykct.core import project_image
    
    sinogram = project_image(
        image=phantom,
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
    input_file: str,
    threshold: Optional[float] = None
) -> Tuple[float, float]:
    """
    Calculate intensity-weighted centroid of a 2D frame.
    
    Parameters
    ----------
    input_file : str
        Path to DEN file.
    threshold : float, optional
        Ignore pixels with intensity below this threshold.
    
    Returns
    -------
    tuple
        (x_centroid, y_centroid) or None if frame is empty.
    
    Notes
    -----
    Uses 0-indexed pixel coordinates (consistent with NumPy).
    """
    # Read DEN header and data
    info = DEN.readHeader(input_file)
    data = DEN.getNumpyArray(input_file)
    
    # Interpret first dimension as frames
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    
    if data.ndim != 3:
        raise ValueError("Expected a 2D or 3D DEN dataset")
    
    # Use first frame
    frame = data[0, :, :].copy().astype(np.float64)
    
    if threshold is not None:
        frame = np.where(frame >= threshold, frame, 0.0)
    
    total_mass = np.sum(frame)
    
    if np.isclose(total_mass, 0):
        return None
    
    yy, xx = np.indices(frame.shape)
    
    x_centroid = np.sum(xx * frame) / total_mass
    y_centroid = np.sum(yy * frame) / total_mass
    
    return x_centroid, y_centroid


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
        OpenCL platform ID.
    verbose : bool
        Print progress information.
    
    Returns
    -------
    np.ndarray
        Sinogram of shape (n_angles, 1, detector_size).
    
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
