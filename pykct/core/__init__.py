#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core KCT C++ Projector bindings and convenience functions.

This module provides:
- pb2d_projector: Wrapper for kct-pb2d-projector binary
- create_projection_matrices: Automatic matrix generation
- project: Execute projection
- project_image: High-level convenience function
"""

import numpy as np
from denpy import DEN
import subprocess
import os
import tempfile
import shutil
from typing import Optional

from pykct.PM import projection_matrix2D


class pb2d_projector:
    """
    Wrapper of the KCT parallel beam 2D projector binary kct-pb2d-projector
    (OpenCL). Named after the binary; other KCT projectors (cone beam, 3D)
    are separate tools.

    Projection matrices are created by :func:`pykct.PM.projection_matrix2D`
    in the material CT convention (a = [cos, sin, 0]/PX, b = [0, 0, 1]/PY),
    that of createCameraMatricesForCircularScanTrajectoryParallelRay3D.py
    ``--material-ct-convention``.

    CoR convention: ``cor_offset`` is the signed distance in detector pixels
    of the principal ray (the ray through the rotation axis) from the
    geometric detector centre, i.e. the rotation axis is imaged at
    detector pixel index (N-1)/2 + cor_offset. This equals
    createProjectionMatrices.py ``--rotation-center-offset-pix`` and
    ``--detector-center-offsetvx = -cor_offset`` of the geometry script
    (which shifts the detector centre, not the axis).
    
    Parameters
    ----------
    platform_id : str, optional
        OpenCL "platform:device" passed to the binary as ``-p``.
        Leave it None (recommended): the C++ runtime then selects the
        platform and device by its own heuristic, which prefers a
        platform expected to perform well (e.g. the first NVIDIA platform)
        and its last device, to avoid clashing with the card that is
        typically used for display or by other jobs. Specify it only
        when a particular OpenCL device must do the work (e.g. "1:2");
        do not pass a dummy "0:0", which overrides that heuristic and
        may select a slow or busy device.
    verbose : bool
        Print progress information.
    
    Example
    -------
    >>> projector = pb2d_projector()  # Let C++ runtime auto-detect
    >>> sinogram = projector.project_phantom(
    ...     image, n_angles=512, detector_sizex=256, cor_offset=0.0
    ... )
    """
    
    def __init__(self, platform_id: Optional[str] = None, verbose: bool = False):
        self.platform_id = platform_id
        self.verbose = verbose
    
    def create_projection_matrices(
        self,
        output_file: str,
        n_angles: int = 512,
        start_angle: float = 0.0,
        angle_range: float = 360.0,
        detector_sizex: int = 256,
        detector_sizey: int = 1,
        pixel_sizex: float = 1.0,
        pixel_sizey: float = 1.0,
        cor_offset: float = 0.0,
        endpoint: bool = False
    ) -> np.ndarray:
        """
        Create projection matrices for circular scan trajectory.
        
        Implements the same logic as createCameraMatricesForCircularScanTrajectoryParallelRay3D.py
        
        Parameters
        ----------
        output_file : str
            Output DEN file path.
        n_angles : int
            Number of projection angles.
        start_angle : float
            Starting angle in degrees.
        angle_range : float
            Angular range in degrees (default: 360°).
        detector_sizex : int
            Number of detector pixels in X.
        detector_sizey : int
            Number of detector pixels in Y (typically 1 for 2D).
        pixel_sizex : float
            Pixel size in X direction.
        pixel_sizey : float
            Pixel size in Y direction.
        cor_offset : float
            CoR offset in pixels: the rotation axis is imaged at detector
            pixel index (N-1)/2 + cor_offset (see :class:`pb2d_projector`).
        endpoint : bool
            Include endpoint in angle range.
        
        Returns
        -------
        np.ndarray
            Camera matrices of shape (n_angles, 2, 4).
        
        Notes
        -----
        Geometry follows KCT convention:
        - Geometric center at (N-1)/2 for N pixels
        - Rotation axis imaged at (N-1)/2 + cor_offset
        - --material-ct-convention: b = [0, 0, 1/PY] (pb2d ignores the b row)
        - Output shape: (n_angles, 2, 4) where each row is [a_x, a_y, a_z, u0] and [b_x, b_y, b_z, v0]
        """
        # Create angles
        angles = np.linspace(
            start_angle, start_angle + angle_range,
            n_angles, endpoint=endpoint
        )
        angles_rad = np.radians(angles)
        
        # Rotation axis imaged at (N-1)/2 + cor_offset, see pykct.PM.projection_matrix2D
        matrices = np.array([
            projection_matrix2D(theta, detector_sizex, detector_sizey, pixel_sizex, pixel_sizey,
                                rotation_center_offset_pix=cor_offset, material_ct_convention=True)
            for theta in angles_rad
        ], dtype=np.float64).reshape((n_angles, 2, 4))
        cor_offset_physical = cor_offset * pixel_sizex

        # Save to DEN file
        DEN.storeNdarrayAsDEN(output_file, matrices, force=True)
        
        if self.verbose:
            print(f"Created projection matrices: {output_file}")
            print(f"  Angles: {n_angles} in [{start_angle:.1f}°, {start_angle + angle_range:.1f}°)")
            print(f"  Detector: {detector_sizex} x {detector_sizey}")
            print(f"  CoR offset: {cor_offset:+.3f} px ({cor_offset_physical:+.3f} units)")
        
        return matrices
    
    def project(
        self,
        volume_file: str,
        matrices_file: str,
        output_file: str,
        detector_sizex: int = 256,
        detector_sizey: Optional[int] = None,
        force: bool = True
    ) -> np.ndarray:
        """
        Project volume using C++ binary with OpenCL.

        The binary projects all slices of the volume, so the projections have
        one detector row per volume slice.

        Parameters
        ----------
        volume_file : str
            Path to volume DEN file (shape: 1 x ny x nx).
        matrices_file : str
            Path to projection matrices DEN file.
        output_file : str
            Path for output projection DEN file.
        detector_sizex : int
            X dimension of detector (required by C++ binary).
        detector_sizey : int, optional
            Number of detector rows. kct-pb2d-projector derives it from the
            volume (one row per slice) and no longer accepts
            --projection-sizey, so this is only checked against the number of
            slices of the volume; a mismatch raises ValueError.
        force : bool
            Overwrite output file if exists.
        
        Returns
        -------
        np.ndarray
            Projected sinogram as numpy array of shape (1, n_angles, ndet).
        """
        if detector_sizey is not None:
            header = DEN.readHeader(volume_file)
            # dimspec is (x, y, z, ...) for the extended DEN format, zdim for the legacy one
            slices = header["dimspec"][2] if "dimspec" in header and len(header["dimspec"]) > 2 \
                else header.get("zdim", 1)
            if int(detector_sizey) != int(slices):
                raise ValueError(
                    f"detector_sizey={detector_sizey} differs from the {slices} slices of "
                    f"{volume_file}; kct-pb2d-projector creates one detector row per slice."
                )

        # Build command
        cmd = ['kct-pb2d-projector']
        
        # Add platform ID only if specified
        if self.platform_id is not None:
            cmd.extend(['-p', self.platform_id])
        
        cmd.extend([
            '--cvp',
            '--force' if force else '',
            '--projection-sizex', str(detector_sizex),
            volume_file,
            matrices_file,
            output_file
        ])
        cmd = [arg for arg in cmd if arg]  # Remove empty strings
        
        if self.verbose:
            print(f"Running: {' '.join(cmd)}")
        
        # Run
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            raise RuntimeError(
                f"Projection failed (return code {result.returncode}):\n"
                f"STDERR: {result.stderr[:500]}"
            )
        
        # Read output
        sinogram = DEN.getNumpyArray(output_file)
        
        # KCT outputs shape: (n_angles, n_slices, ndet)
        # Convert to: (n_slices, n_angles, ndet)
        # This is the standard sinogram convention
        if sinogram.ndim == 3:
            # KCT: (angle, slice, detector)
            # Want: (slice, angle, detector)
            sinogram = np.transpose(sinogram, (1, 0, 2))
        
        if self.verbose:
            print(f"Projection complete: {output_file}")
            print(f"  Shape: {sinogram.shape}")
            print(f"  Range: [{sinogram.min():.4f}, {sinogram.max():.4f}]")
        
        return sinogram
    
    def project_phantom(
        self,
        image: np.ndarray,
        n_angles: int = 512,
        start_angle: float = 0.0,
        angle_range: float = 360.0,
        detector_sizex: int = 256,
        detector_sizey: Optional[int] = None,
        cor_offset: float = 0.0,
        pixel_size: float = 1.0,
        use_temp_dir: bool = True
    ) -> np.ndarray:
        """
        Project image with all parameters in one call.
        
        Convenience function that creates temporary files and manages cleanup.
        
        Parameters
        ----------
        image : np.ndarray
            2D image (ny, nx) or volume (nz, ny, nx); each z slice is
            projected independently (parallel beam).
        n_angles : int
            Number of projection angles.
        start_angle : float
            Starting angle in degrees.
        angle_range : float
            Angular range in degrees (default: 360°).
        detector_sizex : int
            Detector X size.
        detector_sizey : int, optional
            Detector Y size, defaults to the number of slices nz.
        cor_offset : float
            CoR offset in pixels; rotation axis is imaged at detector
            pixel index (N-1)/2 + cor_offset.
        pixel_size : float
            Pixel size (used for both X and Y).
        use_temp_dir : bool
            Use temporary directory for intermediate files.
        
        Returns
        -------
        np.ndarray
            Sinogram of shape (detector_sizey, n_angles, detector_sizex);
            sinogram slice z corresponds to volume slice z.
        
        Example
        -------
        >>> import numpy as np
        >>> from pykct import pb2d_projector
        >>> 
        >>> # Create simple phantom
        >>> image = np.zeros((128, 128))
        >>> image[60:70, 60:70] = 1.0
        >>> 
        >>> # Project with CoR offset
        >>> projector = pb2d_projector()
        >>> sinogram = projector.project_phantom(
        ...     image, n_angles=360, detector_sizex=256, cor_offset=5.0
        ... )
        """
        # Create temporary files
        if use_temp_dir:
            tmpdir = tempfile.mkdtemp(prefix='kct_project_')
            volume_file = os.path.join(tmpdir, 'volume.den')
            matrices_file = os.path.join(tmpdir, 'matrices.den')
            output_file = os.path.join(tmpdir, 'projection.den')
            cleanup_dir = True
        else:
            volume_file = 'temp_volume.den'
            matrices_file = 'temp_matrices.den'
            output_file = 'temp_projection.den'
            cleanup_dir = False
        
        try:
            # Write volume (shape: nz x ny x nx for DEN)
            volume_3d = np.asarray(image, dtype=np.float32)
            if volume_3d.ndim == 2:
                volume_3d = volume_3d[np.newaxis, :, :]
            if volume_3d.ndim != 3:
                raise ValueError(
                    f"Expected 2D image or (nz, ny, nx) volume, got shape {volume_3d.shape}"
                )
            if detector_sizey is None:
                detector_sizey = volume_3d.shape[0]
            DEN.storeNdarrayAsDEN(volume_file, volume_3d, force=True)
            
            # Create matrices
            self.create_projection_matrices(
                matrices_file,
                n_angles=n_angles,
                start_angle=start_angle,
                angle_range=angle_range,
                detector_sizex=detector_sizex,
                detector_sizey=detector_sizey,
                pixel_sizex=pixel_size,
                pixel_sizey=pixel_size,
                cor_offset=cor_offset
            )
            
            # Project
            sinogram = self.project(
                volume_file,
                matrices_file,
                output_file,
                detector_sizex=detector_sizex,
                detector_sizey=detector_sizey
            )
            
            return sinogram
            
        finally:
            # Cleanup
            if cleanup_dir:
                try:
                    shutil.rmtree(tmpdir)
                except:
                    pass
            else:
                for f in [volume_file, matrices_file, output_file]:
                    if os.path.exists(f):
                        os.unlink(f)


def create_projection_matrices(
    output_file: str,
    n_angles: int = 512,
    start_angle: float = 0.0,
    angle_range: float = 360.0,
    detector_sizex: int = 256,
    detector_sizey: int = 1,
    pixel_sizex: float = 1.0,
    pixel_sizey: float = 1.0,
    cor_offset: float = 0.0,
    endpoint: bool = False,
    platform_id: Optional[str] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Convenience function to create projection matrices.
    
    Parameters
    ----------
    output_file : str
        Output DEN file path.
    n_angles : int
        Number of projection angles.
    start_angle : float
        Starting angle in degrees.
    angle_range : float
        Angular range in degrees (default: 360°).
    detector_sizex : int
        Number of detector pixels in X.
    detector_sizey : int
        Number of detector pixels in Y (typically 1 for 2D).
    pixel_sizex : float
        Pixel size in X direction.
    pixel_sizey : float
        Pixel size in Y direction.
    cor_offset : float
        CoR offset in pixels: the rotation axis is imaged at detector
        pixel index (N-1)/2 + cor_offset (see :class:`pb2d_projector`).
    endpoint : bool
        Include endpoint in angle range.
    platform_id : str, optional
        OpenCL "platform:device". Leave None to let the C++ runtime choose
        (recommended); see :class:`pb2d_projector`.
    verbose : bool
        Print progress information.
    
    Returns
    -------
    np.ndarray
        Camera matrices of shape (n_angles, 2, 4).
    """
    projector = pb2d_projector(platform_id=platform_id, verbose=verbose)
    return projector.create_projection_matrices(
        output_file=output_file,
        n_angles=n_angles,
        start_angle=start_angle,
        angle_range=angle_range,
        detector_sizex=detector_sizex,
        detector_sizey=detector_sizey,
        pixel_sizex=pixel_sizex,
        pixel_sizey=pixel_sizey,
        cor_offset=cor_offset,
        endpoint=endpoint
    )


def project(
    volume_file: str,
    matrices_file: str,
    output_file: str,
    detector_sizex: int = 256,
    detector_sizey: int = 1,
    platform_id: Optional[str] = None,
    force: bool = True,
    verbose: bool = False
) -> np.ndarray:
    """
    Convenience function to execute projection.
    
    Parameters
    ----------
    volume_file : str
        Path to volume DEN file (shape: 1 x ny x nx).
    matrices_file : str
        Path to projection matrices DEN file.
    output_file : str
        Path for output projection DEN file.
    detector_sizex : int
        X dimension of detector.
    detector_sizey : int
        Y dimension of detector.
    platform_id : str, optional
        OpenCL "platform:device". Leave None to let the C++ runtime choose
        (recommended); see :class:`pb2d_projector`.
    force : bool
        Overwrite output file if exists.
    verbose : bool
        Print progress information.
    
    Returns
    -------
    np.ndarray
        Projected sinogram.
    """
    projector = pb2d_projector(platform_id=platform_id, verbose=verbose)
    return projector.project(
        volume_file=volume_file,
        matrices_file=matrices_file,
        output_file=output_file,
        detector_sizex=detector_sizex,
        detector_sizey=detector_sizey,
        force=force
    )


def project_image(
    image: np.ndarray,
    n_angles: int = 512,
    start_angle: float = 0.0,
    angle_range: float = 360.0,
    detector_sizex: int = 256,
    detector_sizey: Optional[int] = None,
    cor_offset: float = 0.0,
    pixel_size: float = 1.0,
    platform_id: Optional[str] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    High-level convenience function for projecting images.
    
    Similar to radon function but with full KCT C++ backend support.
    
    Parameters
    ----------
    image : np.ndarray
        2D image (ny, nx) or volume (nz, ny, nx); each z slice is
        projected independently (parallel beam).
    n_angles : int
        Number of projection angles.
    start_angle : float
        Starting angle in degrees.
    angle_range : float
        Angular range in degrees (default: 360°).
    detector_sizex : int
        Detector X size.
    detector_sizey : int, optional
        Detector Y size, defaults to the number of slices nz.
    cor_offset : float
        CoR offset in pixels; rotation axis is imaged at detector
        pixel index (N-1)/2 + cor_offset.
    pixel_size : float
        Pixel size (used for both X and Y).
    platform_id : str, optional
        OpenCL "platform:device". Leave None to let the C++ runtime choose
        (recommended); see :class:`pb2d_projector`.
    verbose : bool
        Print progress information.
    
    Returns
    -------
    np.ndarray
        Sinogram of shape (detector_sizey, n_angles, detector_sizex);
        sinogram slice z corresponds to volume slice z.
    
    Example
    -------
    >>> from pykct import project_image
    >>> import numpy as np
    >>> 
    >>> # Create phantom
    >>> image = np.zeros((128, 128))
    >>> image[50:80, 50:80] = 1.0
    >>> 
    >>> # Simple projection
    >>> sino = project_image(image, n_angles=360, detector_sizex=256)
    >>> # Shape: (1, 360, 256)
    >>> 
    >>> # With CoR offset
    >>> sino_offset = project_image(
    ...     image, n_angles=360, detector_sizex=256, cor_offset=5.0
    ... )
    """
    projector = pb2d_projector(platform_id=platform_id, verbose=verbose)
    return projector.project_phantom(
        image=image,
        n_angles=n_angles,
        start_angle=start_angle,
        angle_range=angle_range,
        detector_sizex=detector_sizex,
        detector_sizey=detector_sizey,
        cor_offset=cor_offset,
        pixel_size=pixel_size
    )

