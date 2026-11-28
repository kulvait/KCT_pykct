#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convenience functions of the 2D parallel beam projector, kept for compatibility; the tools are
the classes of pykct.PB2D.

This module provides:
- pb2d_projector: pykct.PB2D.pb2d_projector
- create_projection_matrices: 1x4 matrices of a circular scan stored in a DEN file
- project: projection of DEN files
- project_image: projection of an image or volume array
"""

from typing import Optional

import numpy as np
from denpy import DEN

from pykct.PB2D import circular_trajectory_projection_matrices, pb2d_projector


def create_projection_matrices(
    output_file: str,
    n_angles: int = 512,
    start_angle: float = 0.0,
    angle_range: float = 360.0,
    detector_sizex: int = 256,
    pixel_size: float = 1.0,
    cor_offset: float = 0.0,
    rotation_center_slope: float = 0.0,
    endpoint: bool = False,
    verbose: bool = False
) -> np.ndarray:
    """
    1x4 projection matrices of a circular scan for the kct-pb2d-* tools, stored in output_file.

    The matrices are pykct.PB2D.circular_trajectory_projection_matrices, pykct.PM.projection_matrix1D
    in the material CT convention (a = [cos, sin, slope, (N-1)/2 + cor_offset] with the pixel size
    1), the rotation axis is imaged in the detector row r at the pixel
    (N-1)/2 + cor_offset + rotation_center_slope * r; the kct-pb2d-* tools evaluate r at the center
    row of the processed slab, see pykct.PM.projection_matrix1D.

    Parameters
    ----------
    output_file : str
        Output DEN file path.
    n_angles, start_angle, angle_range, endpoint
        Angles in degrees np.linspace(start_angle, start_angle + angle_range, n_angles, endpoint).
    detector_sizex : int
        N, number of detector pixels.
    pixel_size : float
        Detector pixel size.
    cor_offset : float
        CoR offset in pixels in the row r = 0.
    rotation_center_slope : float
        Shift of the CoR in pixels per detector row r, see pykct.PM.projection_matrix1D.
    verbose : bool
        Print information.

    Returns
    -------
    np.ndarray
        Projection matrices of shape (n_angles, 1, 4).
    """
    matrices = circular_trajectory_projection_matrices(n_angles, detector_sizex, start_angle, angle_range,
                                                       pixel_size, cor_offset, rotation_center_slope, endpoint)
    DEN.storeNdarrayAsDEN(output_file, matrices, force=True)
    if verbose:
        print(f"Created projection matrices: {output_file}")
        print(f"  Angles: {n_angles} in [{start_angle:.1f}°, {start_angle + angle_range:.1f}°)")
        print(f"  Detector: {detector_sizex} pixels of size {pixel_size}")
        print(f"  CoR offset: {cor_offset:+.3f} px ({cor_offset * pixel_size:+.3f} units), "
              f"slope {rotation_center_slope:+.6f} px/row")
    return matrices


def project(
    volume_file: str,
    matrices_file: str,
    output_file: str,
    detector_sizex: int = 256,
    detector_sizey: Optional[int] = None,
    platform_id: Optional[str] = None,
    force: bool = True,
    verbose: bool = False
) -> np.ndarray:
    """
    Projection of the volume DEN file by the matrices DEN file into output_file.

    detector_sizey is only checked against the number of slices of the volume, kct-pb2d-projector
    creates one detector row per slice. Returns the sinogram (nz, n_angles, detector_sizex). With
    force False an existing output_file raises FileExistsError.
    """
    import os
    if not force and os.path.exists(output_file):
        raise FileExistsError(output_file)
    projector = pb2d_projector(platform_id=platform_id, verbose=verbose)
    return projector.project(volume_file, matrices_file, detector_sizex=detector_sizex, detector_sizey=detector_sizey,
                             output_file=output_file)


def project_image(
    image: np.ndarray,
    n_angles: int = 512,
    start_angle: float = 0.0,
    angle_range: float = 360.0,
    detector_sizex: int = 256,
    detector_sizey: Optional[int] = None,
    cor_offset: float = 0.0,
    rotation_center_slope: float = 0.0,
    pixel_size: float = 1.0,
    platform_id: Optional[str] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Projection of an image (ny, nx) or volume (nz, ny, nx) in a circular scan, see
    pykct.PB2D.pb2d_projector.project.

    pixel_size is the detector pixel size (voxels have size 1), cor_offset the CoR offset in pixels
    and rotation_center_slope its shift in pixels per slice, the rotation axis is imaged at the
    detector pixel index (N-1)/2 + cor_offset + rotation_center_slope * (nz - 1) / 2, the slope is
    evaluated at the center slice, see pykct.PB2D. platform_id: OpenCL
    "platform:device", leave None to let the C++ runtime choose (recommended).

    Returns
    -------
    np.ndarray
        Sinogram of shape (nz, n_angles, detector_sizex); sinogram slice z corresponds to volume slice z.

    Example
    -------
    >>> from pykct import project_image
    >>> sino = project_image(image, n_angles=360, detector_sizex=256, cor_offset=5.0)
    """
    projector = pb2d_projector(platform_id=platform_id, verbose=verbose)
    return projector.project(image, n_angles=n_angles, start_angle=start_angle, angle_range=angle_range,
                             detector_sizex=detector_sizex, detector_sizey=detector_sizey, cor_offset=cor_offset,
                             rotation_center_slope=rotation_center_slope, pixel_size=pixel_size)
