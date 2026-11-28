#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wrappers of the KCT 2D parallel beam tools kct-pb2d-projector, kct-pb2d-backprojector and
kct-pb2d-krylov (OpenCL).

- pb2d_projector.project: forward projection of a volume to a sinogram;
- pb2d_backprojector.backproject: backprojection of a sinogram, with the scalings of the binary;
- pb2d_backprojector.fbp: filtered backprojection, ramp filter in Python and backprojection with
  FBP scaling;
- pb2d_krylov.reconstruct: iterative reconstruction, CGLS or PDHG.

Data conventions
----------------
- volume: numpy array (nz, ny, nx), or (ny, nx) for one slice; each slice z is a 2D problem;
- sinogram: numpy array (nz, n_angles, n_det), or (n_angles, n_det) for one slice; the sinogram
  slice z belongs to the volume slice z. KCT stores projections as (n_angles, nz, n_det), the
  wrappers transpose;
- matrices: the 1x4 projection matrices of the kct-pb2d-* tools, see pykct.PM.projection_matrix1D,
  as a numpy array (n_angles, 1, 4) or (n_angles, 4), the 2x4 matrices (n_angles, 2, 4) of which
  the first row is used, or a DEN file. Without matrices a circular scan is created by
  circular_trajectory_projection_matrices from n_angles, start_angle, angle_range, endpoint,
  pixel_size, cor_offset and rotation_center_slope.

Arrays may be replaced by DEN files (path strings) in the layout of the binaries; the results are
always returned as arrays, with output_file also kept in that file.

Units of length are those of pixel_size and the voxel sizes, the projections are line integrals in
these units. The rotation axis of a circular scan is imaged at the detector pixel
(n_det - 1) / 2 + cor_offset + rotation_center_slope * r. The binaries evaluate the matrices once per
call (block of rows), at the center row r = z_level2d_origin + slab_from + (slab_size - 1) / 2, so the
slope gives a constant shift of the processed slab; resolve the skew by calls on thin slabs, see
pykct.PM.projection_matrix1D.

The binaries are taken from PATH or from bin_dir. platform_id is the OpenCL "platform:device"
passed as -p; leave it None (recommended), the C++ runtime then selects the platform and device by
its own heuristic, which prefers a well performing platform and its last device. Do not pass a
dummy "0:0".
"""

import os
import shutil
import subprocess
import tempfile
from typing import Optional, Sequence

import numpy as np
from denpy import DEN

from pykct.PM import projection_matrix1D

FILTERS = ("ram-lak", "shepp-logan", "cosine", "hamming", "hann")
SCALINGS = ("none", "fbp", "natural", "kaczmarz")
ALGORITHMS = ("cgls", "pdhg")


# ================================================================================================
# Functions
# ================================================================================================


def circular_trajectory_projection_matrices(n_angles, detector_sizex, start_angle=0.0, angle_range=360.0,
                                            pixel_size=1.0, cor_offset=0.0, rotation_center_slope=0.0,
                                            endpoint=False):
    """
    1x4 projection matrices of a circular scan trajectory, shape (n_angles, 1, 4).

    The angles theta (degrees) are np.linspace(start_angle, start_angle + angle_range, n_angles,
    endpoint), the matrices pykct.PM.projection_matrix1D in the material CT convention with the
    rotation axis imaged in the detector row r at the pixel

        (detector_sizex - 1) / 2 + cor_offset + rotation_center_slope * r,

    the kct-pb2d-* tools evaluate r at the center row of the processed slab, see the module
    documentation.
    """
    theta = np.deg2rad(np.linspace(start_angle, start_angle + angle_range, n_angles, endpoint=endpoint))
    return np.array([projection_matrix1D(t, detector_sizex, pixel_size, rotation_center_offset_pix=cor_offset,
                                         rotation_center_slope=rotation_center_slope,
                                         material_ct_convention=True) for t in theta],
                    dtype=np.float64).reshape((n_angles, 1, 4))


def ramp_filter(sinogram, pixel_size=1.0, filter="ram-lak"):
    """
    Ramp filtered sinogram of the filtered backprojection, along the last axis (detector).

    The discrete Ram-Lak kernel h[0] = 1 / (4 w^2), h[n] = -1 / (pi n w)^2 for odd n, 0 for even
    n != 0 (Kak and Slaney, Principles of Computerized Tomographic Imaging, eq. 3.61), w the
    detector pixel size, is convolved with the projections and multiplied by w, so that the result
    is the projection filtered by |omega| in physical units. The convolution is done by FFT with zero
    padding to at least twice the detector size. filter multiplies the frequency response by a
    window: ram-lak (none), shepp-logan, cosine, hamming or hann.
    """
    if filter not in FILTERS:
        raise ValueError("filter %r is none of %s" % (filter, FILTERS))
    p = np.asarray(sinogram, dtype=np.float64)
    n = p.shape[-1]
    size = max(64, int(2 ** np.ceil(np.log2(2 * n))))
    k = np.concatenate([np.arange(0, size // 2 + 1), np.arange(-(size // 2) + 1, 0)])
    h = np.zeros(size, dtype=np.float64)
    h[0] = 0.25 / pixel_size ** 2
    odd = (k % 2) == 1
    h[odd] = -1.0 / (np.pi * k[odd] * pixel_size) ** 2
    response = np.real(np.fft.fft(h)) * pixel_size
    if filter != "ram-lak":
        omega = 2.0 * np.pi * np.fft.fftfreq(size)  # [-pi, pi)
        if filter == "shepp-logan":
            window = np.sinc(omega / (2.0 * np.pi))
        elif filter == "cosine":
            window = np.cos(omega / 2.0)
        elif filter == "hamming":
            window = 0.54 + 0.46 * np.cos(omega)
        else:  # hann
            window = 0.5 + 0.5 * np.cos(omega)
        response = response * window
    padded = np.zeros(p.shape[:-1] + (size,), dtype=np.float64)
    padded[..., :n] = p
    q = np.real(np.fft.ifft(np.fft.fft(padded, axis=-1) * response, axis=-1))[..., :n]
    return q.astype(np.float32)


# ================================================================================================
# Classes
# ================================================================================================


class _pb2d_tool:
    """Running a kct-pb2d-* binary: platform, binary location, temporary files, DEN conversions."""

    binary = None

    def __init__(self, platform_id: Optional[str] = None, verbose: bool = False, bin_dir: Optional[str] = None,
                 barrier: bool = False, relaxed: bool = False, timeout: Optional[float] = None,
                 extra_args: Sequence[str] = ()):
        """
        Parameters
        ----------
        platform_id : str, optional
            OpenCL "platform:device" passed as -p, leave None to let the binary choose (recommended).
        verbose : bool
            Print the commands and their results.
        bin_dir : str, optional
            Directory of the binaries, by default they are taken from PATH.
        barrier, relaxed : bool
            --barrier (CVP with barrier calls, local memory) and --relaxed (relaxed OpenCL math).
        timeout : float, optional
            Timeout of a call in seconds, None waits.
        extra_args : sequence of str
            Further arguments passed to every call.
        """
        self.platform_id = platform_id
        self.verbose = verbose
        self.bin_dir = bin_dir
        self.barrier = barrier
        self.relaxed = relaxed
        self.timeout = timeout
        self.extra_args = list(extra_args)

    def command(self):
        exe = self.binary if self.bin_dir is None else os.path.join(self.bin_dir, self.binary)
        cmd = [exe, "--cvp"]
        if self.platform_id is not None:
            cmd += ["-p", str(self.platform_id)]
        if self.barrier:
            cmd.append("--barrier")
        if self.relaxed:
            cmd.append("--relaxed")
        return cmd + self.extra_args

    def run(self, args, cwd=None):
        """Run the binary with args in the directory cwd, the temporary directory of the call, so that
        files the binary writes into its working directory (e.g. debug volumes of PDHG) end there."""
        cmd = self.command() + [str(a) for a in args]
        if self.verbose:
            print("Running: %s" % " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, cwd=cwd)
        if result.returncode != 0:
            raise RuntimeError("%s failed (return code %d):\nSTDOUT: %s\nSTDERR: %s"
                               % (self.binary, result.returncode, result.stdout[-1000:], result.stderr[-1000:]))
        if self.verbose and result.stdout:
            print(result.stdout[-2000:])
        return result

    # ---------------------------------------------------------------- files

    @staticmethod
    def write_volume(volume, path):
        """Volume array (nz, ny, nx) or (ny, nx) to a DEN file [nx, ny, nz]; a path is returned as it is."""
        if isinstance(volume, (str, os.PathLike)):
            return os.path.abspath(os.fspath(volume))
        v = np.asarray(volume, dtype=np.float32)
        if v.ndim == 2:
            v = v[np.newaxis]
        if v.ndim != 3:
            raise ValueError("Expected a 2D image or a (nz, ny, nx) volume, got shape %s" % (v.shape,))
        DEN.storeNdarrayAsDEN(path, np.ascontiguousarray(v), force=True)
        return path

    @staticmethod
    def write_sinogram(sinogram, path):
        """Sinogram array (nz, n_angles, n_det) or (n_angles, n_det) to a DEN file of KCT projections
        [n_det, nz, n_angles]; a path is returned as it is."""
        if isinstance(sinogram, (str, os.PathLike)):
            return os.path.abspath(os.fspath(sinogram))
        s = np.asarray(sinogram, dtype=np.float32)
        if s.ndim == 2:
            s = s[np.newaxis]
        if s.ndim != 3:
            raise ValueError("Expected a (n_angles, n_det) or (nz, n_angles, n_det) sinogram, got shape %s" % (s.shape,))
        DEN.storeNdarrayAsDEN(path, np.ascontiguousarray(np.transpose(s, (1, 0, 2))), force=True)
        return path

    @staticmethod
    def read_sinogram(path):
        """KCT projections [n_det, nz, n_angles] as a sinogram array (nz, n_angles, n_det)."""
        s = DEN.getNumpyArray(path)
        if s.ndim == 2:  # one projection row: (n_angles, n_det)
            return s[np.newaxis]
        return np.transpose(s, (1, 0, 2))

    @staticmethod
    def sinogram_shape(sinogram):
        """(nz, n_angles, n_det) of a sinogram array or of the KCT projections in a DEN file."""
        if isinstance(sinogram, (str, os.PathLike)):
            dimspec = DEN.readHeader(os.fspath(sinogram))["dimspec"]
            n_det, nz, n_angles = (list(dimspec) + [1, 1])[:3]
            return int(nz), int(n_angles), int(n_det)
        s = np.shape(sinogram)
        return (1,) + tuple(s) if len(s) == 2 else tuple(s)

    @staticmethod
    def write_matrices(matrices, path):
        """Projection matrices array (n, 4), (n, 1, 4) or (n, 2, 4) to a DEN file; a path is returned as
        it is."""
        if isinstance(matrices, (str, os.PathLike)):
            return os.path.abspath(os.fspath(matrices))
        m = np.asarray(matrices, dtype=np.float64)
        if m.ndim == 2 and m.shape[1] == 4:
            m = m[:, np.newaxis, :]
        if m.ndim != 3 or m.shape[1] not in (1, 2) or m.shape[2] != 4:
            raise ValueError("Expected projection matrices of shape (n, 4), (n, 1, 4) or (n, 2, 4), got %s" % (m.shape,))
        DEN.storeNdarrayAsDEN(path, np.ascontiguousarray(m), force=True)
        return path

    @staticmethod
    def detector_pixel_size(matrices):
        """Detector pixel size 1 / |(a0, a1)| of the first projection matrix."""
        if isinstance(matrices, (str, os.PathLike)):
            matrices = DEN.getNumpyArray(os.fspath(matrices))
        m = np.asarray(matrices, dtype=np.float64).reshape((-1, np.shape(matrices)[-1]))
        return 1.0 / np.hypot(m[0, 0], m[0, 1])

    @staticmethod
    def geometry_args(volume_sizex=None, volume_sizey=None, voxel_sizex=1.0, voxel_sizey=None, volume_centerx=0.0,
                      volume_centery=0.0, z_level2d_origin=0.0, slab_from=None, slab_size=None):
        args = []
        if volume_sizex is not None:
            args += ["--volume-sizex", int(volume_sizex), "--volume-sizey",
                     int(volume_sizex if volume_sizey is None else volume_sizey)]
        args += ["--voxel-sizex", repr(float(voxel_sizex)),
                 "--voxel-sizey", repr(float(voxel_sizex if voxel_sizey is None else voxel_sizey))]
        if volume_centerx != 0.0 or volume_centery != 0.0:
            args += ["--volume-centerx", repr(float(volume_centerx)), "--volume-centery", repr(float(volume_centery))]
        if z_level2d_origin != 0.0:
            args += ["--z-level2d-origin", repr(float(z_level2d_origin))]
        if slab_from is not None:
            args += ["--slab-from", int(slab_from)]
        if slab_size is not None:
            args += ["--slab-size", int(slab_size)]
        return args

    class _workdir:
        """Temporary directory removed on exit."""

        def __enter__(self):
            self.path = tempfile.mkdtemp(prefix="pykct_pb2d_")
            return self.path

        def __exit__(self, *exc):
            shutil.rmtree(self.path, ignore_errors=True)

    @staticmethod
    def output_path(tmp, output_file, name):
        return os.path.abspath(os.fspath(output_file)) if output_file is not None else os.path.join(tmp, name)


class pb2d_projector(_pb2d_tool):
    """
    Wrapper of kct-pb2d-projector, see the module documentation for the conventions and the
    constructor parameters.

    Example
    -------
    >>> projector = pb2d_projector()
    >>> sinogram = projector.project(image, n_angles=360, detector_sizex=256, cor_offset=5.0)
    """

    binary = "kct-pb2d-projector"

    def project(self, volume, matrices=None, detector_sizex: int = 256, n_angles: int = 512,
                start_angle: float = 0.0, angle_range: float = 360.0, cor_offset: float = 0.0,
                rotation_center_slope: float = 0.0, pixel_size: float = 1.0, endpoint: bool = False, detector_sizey: Optional[int] = None,
                voxel_sizex: float = 1.0, voxel_sizey: Optional[float] = None, volume_centerx: float = 0.0,
                volume_centery: float = 0.0, z_level2d_origin: float = 0.0, slab_from: Optional[int] = None,
                slab_size: Optional[int] = None, output_file: Optional[str] = None) -> np.ndarray:
        """
        Forward projection of a volume.

        Parameters
        ----------
        volume : numpy.ndarray or str
            Volume (nz, ny, nx) or image (ny, nx), or a DEN file [nx, ny, nz].
        matrices : numpy.ndarray or str, optional
            Projection matrices, see the module documentation; without them those of a circular scan,
            circular_trajectory_projection_matrices(n_angles, detector_sizex, start_angle,
            angle_range, pixel_size, cor_offset, rotation_center_slope, endpoint).
        detector_sizex : int
            Number of detector pixels.
        n_angles, start_angle, angle_range, cor_offset, rotation_center_slope, pixel_size, endpoint
            The circular scan without matrices: angles in degrees, cor_offset in pixels in the row
            r = 0, rotation_center_slope in pixels per row r (applied at the center row of the
            slab, see the module documentation), pixel_size the detector pixel size.
        detector_sizey : int, optional
            Number of detector rows, checked against the number of processed slices.
        voxel_sizex, voxel_sizey, volume_centerx, volume_centery : float
            Voxel size (voxel_sizey defaults to voxel_sizex) and the center of the volume.
        z_level2d_origin, slab_from, slab_size
            Row protocol and the slices to process, see pykct.PM.projection_matrix1D and
            kct-pb2d-projector.
        output_file : str, optional
            Keep the projections in this DEN file [n_det, nz, n_angles].

        Returns
        -------
        numpy.ndarray
            Sinogram (nz, n_angles, detector_sizex).
        """
        with self._workdir() as tmp:
            volume_file = self.write_volume(volume, os.path.join(tmp, "volume.den"))
            if matrices is None:
                matrices = circular_trajectory_projection_matrices(n_angles, detector_sizex, start_angle, angle_range,
                                                                   pixel_size, cor_offset, rotation_center_slope,
                                                                   endpoint)
            matrices_file = self.write_matrices(matrices, os.path.join(tmp, "matrices.den"))
            if detector_sizey is not None:
                nz = int(DEN.readHeader(volume_file)["dimspec"][2]) if slab_size is None else int(slab_size)
                if slab_size is None and slab_from is not None:
                    nz -= int(slab_from)
                if int(detector_sizey) != nz:
                    raise ValueError("detector_sizey=%d differs from the %d processed slices; kct-pb2d-projector "
                                     "creates one detector row per slice." % (detector_sizey, nz))
            out = self.output_path(tmp, output_file, "projections.den")
            args = ["--force", "--projection-sizex", int(detector_sizex)]
            args += self.geometry_args(None, None, voxel_sizex, voxel_sizey, volume_centerx, volume_centery,
                                       z_level2d_origin, slab_from, slab_size)
            self.run(args + [volume_file, matrices_file, out], cwd=tmp)
            return self.read_sinogram(out)


class pb2d_backprojector(_pb2d_tool):
    """
    Wrapper of kct-pb2d-backprojector, see the module documentation for the conventions and the
    constructor parameters.

    Example
    -------
    >>> backprojector = pb2d_backprojector()
    >>> image = backprojector.fbp(sinogram, angle_range=180.0, cor_offset=5.0)[0]
    """

    binary = "kct-pb2d-backprojector"

    def backproject(self, sinogram, matrices=None, volume_sizex: Optional[int] = None,
                    volume_sizey: Optional[int] = None, scaling: str = "none", start_angle: float = 0.0,
                    angle_range: float = 360.0, cor_offset: float = 0.0, rotation_center_slope: float = 0.0,
                    pixel_size: float = 1.0, endpoint: bool = False, voxel_sizex: float = 1.0, voxel_sizey: Optional[float] = None,
                    volume_centerx: float = 0.0, volume_centery: float = 0.0, z_level2d_origin: float = 0.0,
                    slab_from: Optional[int] = None, slab_size: Optional[int] = None,
                    output_file: Optional[str] = None) -> np.ndarray:
        """
        Backprojection of a sinogram.

        Parameters
        ----------
        sinogram : numpy.ndarray or str
            Sinogram (nz, n_angles, n_det) or (n_angles, n_det), or a DEN file of KCT projections.
        matrices : numpy.ndarray or str, optional
            Projection matrices, see the module documentation; without them those of a circular scan
            with the n_angles and n_det of the sinogram.
        volume_sizex, volume_sizey : int, optional
            Size of the volume, volume_sizex defaults to n_det, volume_sizey to volume_sizex.
        scaling : str
            none (the adjoint of the projector), fbp (times pi / n_angles), natural or kaczmarz, see
            kct-pb2d-backprojector.
        start_angle, angle_range, cor_offset, rotation_center_slope, pixel_size, endpoint
            The circular scan without matrices, see pb2d_projector.project.
        voxel_sizex, voxel_sizey, volume_centerx, volume_centery, z_level2d_origin, slab_from, slab_size
            See pb2d_projector.project, the slab selects detector rows.
        output_file : str, optional
            Keep the volume in this DEN file [nx, ny, nz].

        Returns
        -------
        numpy.ndarray
            Volume (nz, volume_sizey, volume_sizex).
        """
        if scaling not in SCALINGS:
            raise ValueError("scaling %r is none of %s" % (scaling, SCALINGS))
        nz, n_angles, n_det = self.sinogram_shape(sinogram)
        if volume_sizex is None:
            volume_sizex = n_det
        with self._workdir() as tmp:
            sinogram_file = self.write_sinogram(sinogram, os.path.join(tmp, "projections.den"))
            if matrices is None:
                matrices = circular_trajectory_projection_matrices(n_angles, n_det, start_angle, angle_range, pixel_size,
                                                                   cor_offset, rotation_center_slope, endpoint)
            matrices_file = self.write_matrices(matrices, os.path.join(tmp, "matrices.den"))
            out = self.output_path(tmp, output_file, "volume.den")
            args = ["--force", "--backprojector-%s-scaling" % ("no" if scaling == "none" else scaling)]
            args += self.geometry_args(volume_sizex, volume_sizey, voxel_sizex, voxel_sizey, volume_centerx,
                                       volume_centery, z_level2d_origin, slab_from, slab_size)
            self.run(args + [sinogram_file, matrices_file, out], cwd=tmp)
            v = DEN.getNumpyArray(out)
            return v[np.newaxis] if v.ndim == 2 else v

    def fbp(self, sinogram, matrices=None, volume_sizex: Optional[int] = None, volume_sizey: Optional[int] = None,
            filter: str = "ram-lak", start_angle: float = 0.0, angle_range: float = 360.0, cor_offset: float = 0.0,
            rotation_center_slope: float = 0.0, pixel_size: float = 1.0, endpoint: bool = False, voxel_sizex: float = 1.0,
            voxel_sizey: Optional[float] = None, volume_centerx: float = 0.0, volume_centery: float = 0.0,
            z_level2d_origin: float = 0.0, slab_from: Optional[int] = None, slab_size: Optional[int] = None,
            output_file: Optional[str] = None) -> np.ndarray:
        """
        Filtered backprojection of a sinogram of a scan of uniformly distributed angles over 180 or 360
        degrees.

        The projections are filtered by ramp_filter with the detector pixel size (pixel_size, or that
        of the first matrix), multiplied by pixel_size / (voxel_sizex * voxel_sizey), as the
        backprojection of the cutting voxel projector adds per angle the voxel area / pixel size, and
        backprojected with the FBP scaling pi / n_angles. Parameters as backproject, filter see
        ramp_filter.

        Returns
        -------
        numpy.ndarray
            Volume (nz, volume_sizey, volume_sizex).
        """
        if isinstance(sinogram, (str, os.PathLike)):
            sinogram = self.read_sinogram(os.fspath(sinogram))
        if matrices is not None:
            pixel_size = self.detector_pixel_size(matrices)
        vy = voxel_sizex if voxel_sizey is None else voxel_sizey
        q = ramp_filter(sinogram, pixel_size, filter) * (pixel_size / (voxel_sizex * vy))
        return self.backproject(q, matrices, volume_sizex, volume_sizey, "fbp", start_angle, angle_range, cor_offset,
                                rotation_center_slope, pixel_size, endpoint, voxel_sizex, voxel_sizey, volume_centerx, volume_centery,
                                z_level2d_origin, slab_from, slab_size, output_file)


class pb2d_krylov(_pb2d_tool):
    """
    Wrapper of kct-pb2d-krylov, see the module documentation for the conventions and the constructor
    parameters.

    Example
    -------
    >>> krylov = pb2d_krylov()
    >>> image = krylov.reconstruct(sinogram, max_iterations=40, cor_offset=5.0)[0]
    """

    binary = "kct-pb2d-krylov"

    def reconstruct(self, sinogram, matrices=None, volume_sizex: Optional[int] = None,
                    volume_sizey: Optional[int] = None, algorithm: str = "cgls", max_iterations: int = 40,
                    stopping_relative_error: Optional[float] = None, x0=None,
                    tikhonov_lambda_l2: Optional[float] = None, tikhonov_lambda_v2: Optional[float] = None,
                    tikhonov_lambda_laplace: Optional[float] = None, pdhg_mu: Optional[float] = None,
                    max_iterations_pdhg: Optional[int] = None, start_angle: float = 0.0, angle_range: float = 360.0,
                    cor_offset: float = 0.0, rotation_center_slope: float = 0.0, pixel_size: float = 1.0,
                    endpoint: bool = False, voxel_sizex: float = 1.0, voxel_sizey: Optional[float] = None, volume_centerx: float = 0.0,
                    volume_centery: float = 0.0, z_level2d_origin: float = 0.0, slab_from: Optional[int] = None,
                    slab_size: Optional[int] = None, args: Sequence[str] = (),
                    output_file: Optional[str] = None) -> np.ndarray:
        """
        Iterative reconstruction of a sinogram.

        Parameters
        ----------
        sinogram, matrices, volume_sizex, volume_sizey
            See pb2d_backprojector.backproject.
        algorithm : str
            cgls or pdhg (TV regularized), the algorithms of kct-pb2d-krylov for this geometry.
        max_iterations : int
            --max-iterations.
        stopping_relative_error : float, optional
            --stopping-relative-error.
        x0 : numpy.ndarray or str, optional
            Initial vector, volume (nz, ny, nx) of the slab or of the whole output, or a DEN file.
        tikhonov_lambda_l2, tikhonov_lambda_v2, tikhonov_lambda_laplace : float, optional
            Tikhonov regularization of CGLS.
        pdhg_mu, max_iterations_pdhg : optional
            TV weight and inner iterations of PDHG.
        start_angle, angle_range, cor_offset, rotation_center_slope, pixel_size, endpoint
            The circular scan without matrices, see pb2d_projector.project.
        voxel_sizex, voxel_sizey, volume_centerx, volume_centery, z_level2d_origin, slab_from, slab_size
            See pb2d_backprojector.backproject.
        args : sequence of str
            Further arguments of kct-pb2d-krylov, e.g. ["--gradient-2d"].
        output_file : str, optional
            Keep the volume in this DEN file [nx, ny, nz].

        Returns
        -------
        numpy.ndarray
            Volume (nz, volume_sizey, volume_sizex).
        """
        if algorithm not in ALGORITHMS:
            raise ValueError("algorithm %r is none of %s" % (algorithm, ALGORITHMS))
        nz, n_angles, n_det = self.sinogram_shape(sinogram)
        if volume_sizex is None:
            volume_sizex = n_det
        with self._workdir() as tmp:
            sinogram_file = self.write_sinogram(sinogram, os.path.join(tmp, "projections.den"))
            if matrices is None:
                matrices = circular_trajectory_projection_matrices(n_angles, n_det, start_angle, angle_range, pixel_size,
                                                                   cor_offset, rotation_center_slope, endpoint)
            matrices_file = self.write_matrices(matrices, os.path.join(tmp, "matrices.den"))
            out = self.output_path(tmp, output_file, "volume.den")
            cmd = ["--force", "--%s" % algorithm, "--max-iterations", int(max_iterations)]
            for option, value in (("--stopping-relative-error", stopping_relative_error),
                                  ("--tikhonov-lambda-l2", tikhonov_lambda_l2),
                                  ("--tikhonov-lambda-v2", tikhonov_lambda_v2),
                                  ("--tikhonov-lambda-laplace", tikhonov_lambda_laplace),
                                  ("--pdhg-mu", pdhg_mu), ("--max-iterations-pdhg", max_iterations_pdhg)):
                if value is not None:
                    cmd += [option, value]
            if x0 is not None:
                cmd += ["--x0", self.write_volume(x0, os.path.join(tmp, "x0.den"))]
            cmd += self.geometry_args(volume_sizex, volume_sizey, voxel_sizex, voxel_sizey, volume_centerx,
                                      volume_centery, z_level2d_origin, slab_from, slab_size)
            self.run(cmd + list(args) + [sinogram_file, matrices_file, out], cwd=tmp)
            v = DEN.getNumpyArray(out)
            return v[np.newaxis] if v.ndim == 2 else v
