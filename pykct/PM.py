#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Projection matrices of the KCT parallel ray geometry.

Port of the matrix construction of the KCT script
createCameraMatricesForCircularScanTrajectoryParallelRay3D.py. The theory is in
https://kulvait.github.io/KCT_doc/posts/working-with-kct-cbct-5-parallel-beam-geometry.html
(general model), in
https://kulvait.github.io/KCT_doc/posts/working-with-kct-cbct-6-parallel-beam-geometry-2d.html
(orthogonal detector, 2D -> 1D geometry and skewed rotation axis) and the
conventions in
https://kulvait.github.io/KCT_doc/posts/tomographic-notes-1-geometric-conventions.html

Two formats are provided:

- 2x4 matrices [[a, px0], [b, py0]] mapping a point (x, y, z) of the volume
  in units of length to the detector coordinates (PX, PY), used by the 3D
  parallel beam tools of KCT (kct-pb-*).
    - projection_matrix2D_vectors: general model given by the detector
      vectors VX, VY, which need not be orthogonal, rows a, b are their dual
      basis computed by means of the Gram matrix.
    - projection_matrix2D: orthogonal detector aligned with the geometry,
      the rotation axis is parallel to the detector columns.
    - projection_matrix2D_skew: as projection_matrix2D with the y axis of
      the detector skewed so that the image of the rotation axis moves by
      rotation_center_slope columns per row, every row images a plane of
      constant z.
    - projection_matrix2D_tilt: as projection_matrix2D with the detector
      rotated within its plane so that the image of the rotation axis moves
      by rotation_center_slope columns per row, the axes stay orthogonal.
- projection_matrix1D: 1x4 matrix [[a0, a1, a2, a3]] mapping (x, y, r) to
  PX, where r is the detector row index, used by the 2D parallel beam tools
  of KCT (kct-pb2d-*). The coefficient a2 = rotation_center_slope encodes
  a skewed rotation axis as a shift of the rotation center per detector
  row. The tools evaluate the matrix once per block of rows processed
  together, at its center row r = --z-level2d-origin + slab-from
  + (slab-size - 1) / 2, so the shift is constant within the block.

The kct-pb2d-* tools accept also the 2x4 matrices and use their first row,
which equals the 1x4 matrix without skew.

Units of length are by convention millimeters.
"""

import numpy as np


def projection_matrix2D(theta,
                        detector_sizex,
                        detector_sizey,
                        pixel_sizex,
                        pixel_sizey,
                        detector_centerx=0.0,
                        detector_centery=0.0,
                        detector_centerz=0.0,
                        detector_center_offsetvx=0.0,
                        detector_center_offsetvy=0.0,
                        rotation_center_offset_pix=0.0,
                        material_ct_convention=True):
    """
    Projection matrix of a single view of the parallel ray geometry, 2x4 format.

    The matrix maps a point x = (x, y, z) of the volume to the detector
    coordinates (PX, PY), measured in pixels,

        PX = a . x + px0
        PY = b . x + py0,

    and it is returned as the 2x4 array [[a, px0], [b, py0]]. The detector
    coordinate k is the center of the pixel with index k, so that the pixel k
    covers [k - 0.5, k + 0.5] and the geometric center of a detector with
    N = detector_sizex columns and M = detector_sizey rows is at
    ((N - 1) / 2, (M - 1) / 2).

    Geometry
    --------
    The rotation axis is the z axis and theta is the angle of the Radon
    transform, measured counterclockwise from the x axis. DX and DY are the
    unit vectors along increasing PX and PY,

        material CT convention (default):   DX = (cos theta,  sin theta, 0)
                                            DY = (0, 0,  1)
        medical CT convention:              DX = (cos theta, -sin theta, 0)
                                            DY = (0, 0, -1)

    In the material convention the detector rows PY grow with z. The medical
    convention mirrors z so that the rows grow towards -z, and since
    mirroring one axis reverses the orientation, the detector rotates by
    -theta in the xy plane. Rays are orthogonal to both DX and DY, so the
    rows of the matrix are

        a = DX / pixel_sizex,    b = DY / pixel_sizey.

    A row of a parallel ray projection matrix only fixes the direction of
    the detector, the position along the rays does not change the projection.

    Detector position
    -----------------
    The detector center C is

        C = (detector_centerx, detector_centery, detector_centerz)
            + detector_center_offsetvx * DX + detector_center_offsetvy * DY,

    the offsets thus move the center along the detector directions by a
    physical length, independent of theta. C is imaged at the center of the
    detector shifted by rotation_center_offset_pix pixels in PX,

        px0 = (N - 1) / 2 - a . C + rotation_center_offset_pix
        py0 = (M - 1) / 2 - b . C.

    With the default C = 0 the rotation axis (x = y = 0) is imaged at
    the column

        PX = (N - 1) / 2 + rotation_center_offset_pix
             - detector_center_offsetvx / pixel_sizex,

    positive rotation_center_offset_pix moves the image of the rotation axis
    towards higher column indices, which is the convention of
    --rotation-center-offset-pix of createProjectionMatrices.py and of the
    rotation center detection scripts.

    The detector axes DX and DY are orthogonal and the rotation axis is
    parallel to DY, so the matrix has no skew, a[2] = 0. For a skewed
    rotation axis use the 1x4 format of projection_matrix1D.

    Parameters
    ----------
    theta : float
        Angle of the Radon transform in radians.
    detector_sizex : int
        N, number of detector pixels in x (columns).
    detector_sizey : int
        M, number of detector pixels in y (rows).
    pixel_sizex : float
        Pixel size in x.
    pixel_sizey : float
        Pixel size in y.
    detector_centerx, detector_centery, detector_centerz : float
        Coordinates of the center of the detector before the offsets.
    detector_center_offsetvx, detector_center_offsetvy : float
        Offsets of the center of the detector along DX and DY in units of
        length, see above.
    rotation_center_offset_pix : float
        Shift of the image of the rotation axis in pixels, see above.
    material_ct_convention : bool
        If True (default), PY grows with z, otherwise the medical convention
        is used.

    Returns
    -------
    numpy.ndarray
        Array of shape (2, 4) and dtype float64, [[a, px0], [b, py0]].
    """
    N = float(detector_sizex)
    M = float(detector_sizey)
    # Unit direction vectors along increasing PX and PY coordinates
    if material_ct_convention:
        DX = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)
        DY = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        DX = np.array([np.cos(theta), -np.sin(theta), 0.0], dtype=np.float64)
        DY = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    a = DX / pixel_sizex
    b = DY / pixel_sizey
    detectorCenter = np.array([detector_centerx, detector_centery, detector_centerz], dtype=np.float64) \
        + DX * detector_center_offsetvx + DY * detector_center_offsetvy
    px0 = N * 0.5 - 0.5 - detectorCenter.dot(a) + rotation_center_offset_pix
    py0 = M * 0.5 - 0.5 - detectorCenter.dot(b)
    return np.array([np.append(a, px0), np.append(b, py0)], dtype=np.float64)


def projection_matrix1D(theta,
                        detector_size,
                        pixel_size,
                        detector_centerx=0.0,
                        detector_centery=0.0,
                        detector_center_offsetvx=0.0,
                        rotation_center_offset_pix=0.0,
                        rotation_center_slope=0.0,
                        material_ct_convention=True):
    """
    Projection matrix of a single view of the parallel ray geometry, 1x4 format
    of the kct-pb2d-* tools.

    The geometry in the xy plane and the parameters are those of
    projection_matrix2D, the matrix is its first row [a0, a1, a2, a3] with a2
    replaced by rotation_center_slope. The third coordinate is not the
    coordinate z but the detector row index r, the matrix maps the point
    (x, y) of the slice imaged by the row r to the detector column

        PX = a0 * x + a1 * y + a2 * r + a3,
        a2 = rotation_center_slope.

    So the rotation axis is imaged in the row r at the column

        PX = (N - 1) / 2 + rotation_center_offset_pix
             + rotation_center_slope * r - detector_center_offsetvx / pixel_size,

    where N = detector_size. The skew is thus not a property of the volume
    geometry, it is a protocol between the caller and the kct-pb2d-* tools.
    They do not evaluate the matrix per row but once per block of rows
    processed together, at the center row of the block

        r = --z-level2d-origin + slab-from + (slab-size - 1) / 2,

    the whole block is reconstructed with the rotation center of this row
    (the tools may split a slab into several blocks to fit the device
    memory, each uses its own center row). The skew is thus resolved slab by
    slab, exactly with slabs of one row. A linear fit offset(r) = A + B * r
    of the rotation center in pixels is encoded by
    rotation_center_offset_pix = A and rotation_center_slope = B, and
    --z-level2d-origin is the row 0 of the processed projections in the row
    coordinates of the fit, 0 if they are the same. Without skew, the matrix
    is the first row of projection_matrix2D and r is irrelevant.

    Parameters
    ----------
    theta : float
        Angle of the Radon transform in radians.
    detector_size : int
        N, number of detector pixels in x (columns).
    pixel_size : float
        Pixel size in x.
    detector_centerx, detector_centery : float
        Coordinates of the center of the detector before the offset.
    detector_center_offsetvx : float
        Offset of the center of the detector along DX in units of length.
    rotation_center_offset_pix : float
        Shift of the image of the rotation axis in pixels in the row r = 0.
    rotation_center_slope : float
        Shift of the image of the rotation axis in pixels per detector row r.
    material_ct_convention : bool
        If True (default), the detector rotates by theta, otherwise by -theta
        in the xy plane, see projection_matrix2D.

    Returns
    -------
    numpy.ndarray
        Array of shape (1, 4) and dtype float64, [[a0, a1, a2, a3]].
    """
    pm2d = projection_matrix2D(theta, detector_size, 1, pixel_size, 1.0,
                               detector_centerx=detector_centerx,
                               detector_centery=detector_centery,
                               detector_center_offsetvx=detector_center_offsetvx,
                               rotation_center_offset_pix=rotation_center_offset_pix,
                               material_ct_convention=material_ct_convention)
    pm1d = pm2d[0].copy()
    # Third coordinate is the detector row index, PX changes by rotation_center_slope per row
    pm1d[2] = rotation_center_slope
    return pm1d.reshape((1, 4))


def projection_matrix2D_vectors(VX, VY, detector_center, detector_sizex, detector_sizey, ray=None):
    """
    Projection matrix of the general parallel ray geometry, 2x4 format.

    The detector is given by the vectors VX and VY between the centers of
    neighboring pixels along increasing PX and PY and by the center of the
    detector, which is imaged at ((N - 1) / 2, (M - 1) / 2), where
    N = detector_sizex and M = detector_sizey. The vectors need not be
    orthogonal to each other nor to the ray direction.

    The rows a and b of the matrix satisfy

        (a, ray) = 0,  (a, VX) = 1,  (a, VY) = 0,
        (b, ray) = 0,  (b, VX) = 0,  (b, VY) = 1,

    they are the dual basis to the projections VX0, VY0 of VX, VY onto the
    plane orthogonal to the ray,

        [a; b] = G^-1 [VX0; VY0],   G = Gram matrix of VX0, VY0,

    and px0 = (N - 1) / 2 - (a, d), py0 = (M - 1) / 2 - (b, d), see
    https://kulvait.github.io/KCT_doc/posts/working-with-kct-cbct-5-parallel-beam-geometry.html

    Parameters
    ----------
    VX, VY : array_like
        Vectors of length 3 between the centers of neighboring pixels.
    detector_center : array_like
        Point d of length 3 imaged at the center of the detector.
    detector_sizex : int
        N, number of detector pixels in x (columns).
    detector_sizey : int
        M, number of detector pixels in y (rows).
    ray : array_like, optional
        Ray direction, defaults to VX x VY, i.e. the detector orthogonal to
        the rays. Specify it for a detector tilted with respect to the rays.

    Returns
    -------
    numpy.ndarray
        Array of shape (2, 4) and dtype float64, [[a, px0], [b, py0]].
    """
    VX = np.asarray(VX, dtype=np.float64)
    VY = np.asarray(VY, dtype=np.float64)
    d = np.asarray(detector_center, dtype=np.float64)
    r = np.cross(VX, VY) if ray is None else np.asarray(ray, dtype=np.float64)
    rr = r.dot(r)
    if rr == 0.0:
        raise ValueError("Ray direction is zero, VX and VY are parallel or ray is zero.")
    V = np.array([VX - VX.dot(r) / rr * r, VY - VY.dot(r) / rr * r])
    G = V @ V.T
    if abs(np.linalg.det(G)) <= 1e-12 * G[0, 0] * G[1, 1]:
        raise ValueError("Projections of VX and VY onto the plane orthogonal to the rays are parallel.")
    (a, b) = np.linalg.solve(G, V)
    px0 = float(detector_sizex) * 0.5 - 0.5 - d.dot(a)
    py0 = float(detector_sizey) * 0.5 - 0.5 - d.dot(b)
    return np.array([np.append(a, px0), np.append(b, py0)], dtype=np.float64)


def _detector_directions(theta, material_ct_convention):
    """Unit vectors DX, DY along increasing PX, PY, see projection_matrix2D."""
    if material_ct_convention:
        DX = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)
        DY = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        DX = np.array([np.cos(theta), -np.sin(theta), 0.0], dtype=np.float64)
        DY = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return (DX, DY)


def projection_matrix2D_skew(theta,
                             detector_sizex,
                             detector_sizey,
                             pixel_sizex,
                             pixel_sizey,
                             detector_centerx=0.0,
                             detector_centery=0.0,
                             detector_centerz=0.0,
                             detector_center_offsetvx=0.0,
                             detector_center_offsetvy=0.0,
                             rotation_center_offset_pix=0.0,
                             rotation_center_slope=0.0,
                             material_ct_convention=True):
    """
    Projection matrix with the skewed y axis of the detector, 2x4 format.

    The parameters are those of projection_matrix2D with the additional
    rotation_center_slope, the number of columns, by which the image of the
    rotation axis moves per detector row of increasing PY. The detector has
    the vectors

        VX = pixel_sizex * DX
        VY = pixel_sizey * DY + s * pixel_sizex * DX,   s = -rotation_center_slope,

    so that moving by one row on the detector moves by s columns. Note the
    sign, the skew s of the detector axis and the observed slope of the image
    of the rotation axis have opposite signs. The rows of the matrix are

        a = DX / pixel_sizex + rotation_center_slope * DY / pixel_sizey
        b = DY / pixel_sizey,

    i.e. a[2] = rotation_center_slope * b[2], and every detector row images a
    plane of constant z. The center of the detector is
    (detector_centerx, detector_centery, detector_centerz)
    + detector_center_offsetvx * DX + detector_center_offsetvy * DY
    - rotation_center_offset_pix * VX, so rotation_center_offset_pix shifts PX
    by this number of pixels in all rows. The image of the rotation axis is in
    the detector row PY at the column

        PX = (N - 1) / 2 + rotation_center_offset_pix
             + rotation_center_slope * (PY - (M - 1) / 2)
             - detector_center_offsetvx / pixel_sizex

    (with detector_centerx = detector_centery = 0), so rotation_center_offset_pix
    refers to the center row (M - 1) / 2. It is also the row py0 imaging the
    plane z = 0 only if detector_center_offsetvy = detector_centerz = 0. With
    rotation_center_slope = 0 the matrix equals projection_matrix2D. The
    substitution z = (r - py0) / b[2] gives projection_matrix1D with
    rotation_center_offset_pix shifted by -rotation_center_slope * (M - 1) / 2,
    see
    https://kulvait.github.io/KCT_doc/posts/working-with-kct-cbct-6-parallel-beam-geometry-2d.html

    Returns
    -------
    numpy.ndarray
        Array of shape (2, 4) and dtype float64, [[a, px0], [b, py0]].
    """
    (DX, DY) = _detector_directions(theta, material_ct_convention)
    VX = pixel_sizex * DX
    VY = pixel_sizey * DY - rotation_center_slope * pixel_sizex * DX
    detectorCenter = np.array([detector_centerx, detector_centery, detector_centerz], dtype=np.float64) \
        + DX * detector_center_offsetvx + DY * detector_center_offsetvy - rotation_center_offset_pix * VX
    return projection_matrix2D_vectors(VX, VY, detectorCenter, detector_sizex, detector_sizey)


def projection_matrix2D_tilt(theta,
                             detector_sizex,
                             detector_sizey,
                             pixel_sizex,
                             pixel_sizey,
                             detector_centerx=0.0,
                             detector_centery=0.0,
                             detector_centerz=0.0,
                             detector_center_offsetvx=0.0,
                             detector_center_offsetvy=0.0,
                             rotation_center_offset_pix=0.0,
                             rotation_center_slope=0.0,
                             material_ct_convention=True):
    """
    Projection matrix with the detector rotated within its plane, 2x4 format.

    The parameters are those of projection_matrix2D_skew. The detector axes
    stay orthogonal with the pixel sizes pixel_sizex and pixel_sizey and
    they are rotated by the angle phi within the detector plane

        VX = pixel_sizex * (cos(phi) * DX - sin(phi) * DY)
        VY = pixel_sizey * (sin(phi) * DX + cos(phi) * DY),

    where tan(phi) = -rotation_center_slope * pixel_sizex / pixel_sizey, so
    that the image of the rotation axis moves by rotation_center_slope columns
    per detector row. In general both rows of the matrix have nonzero x, y and
    z components and a detector row does not image a plane of constant z,
    so this model can not be expressed by projection_matrix1D. The center of
    the detector is defined as in projection_matrix2D_skew, so
    rotation_center_offset_pix refers to the row py0, which images the plane
    z = 0. With rotation_center_slope = 0 the matrix equals
    projection_matrix2D, see
    https://kulvait.github.io/KCT_doc/posts/working-with-kct-cbct-6-parallel-beam-geometry-2d.html

    Returns
    -------
    numpy.ndarray
        Array of shape (2, 4) and dtype float64, [[a, px0], [b, py0]].
    """
    (DX, DY) = _detector_directions(theta, material_ct_convention)
    phi = np.arctan(-rotation_center_slope * pixel_sizex / pixel_sizey)
    VX = pixel_sizex * (np.cos(phi) * DX - np.sin(phi) * DY)
    VY = pixel_sizey * (np.sin(phi) * DX + np.cos(phi) * DY)
    detectorCenter = np.array([detector_centerx, detector_centery, detector_centerz], dtype=np.float64) \
        + DX * detector_center_offsetvx + DY * detector_center_offsetvy - rotation_center_offset_pix * VX
    return projection_matrix2D_vectors(VX, VY, detectorCenter, detector_sizex, detector_sizey)
