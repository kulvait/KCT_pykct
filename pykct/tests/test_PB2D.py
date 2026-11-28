"""Tests of pykct.PB2D: ramp filter, and with the kct-pb2d-* binaries in PATH adjointness, FBP,
CGLS/PDHG, matrix formats, slabs and files."""

import shutil

import numpy as np
import pytest

from pykct import PB2D

binaries = pytest.mark.skipif(
    any(shutil.which(b) is None for b in ("kct-pb2d-projector", "kct-pb2d-backprojector", "kct-pb2d-krylov")),
    reason="kct-pb2d-* binaries not in PATH")

N = 96
YY, XX = np.mgrid[:N, :N] - (N - 1) / 2
DISC = ((XX ** 2 + YY ** 2) < 30 ** 2).astype(np.float32)
INNER = (XX ** 2 + YY ** 2) < 22 ** 2


def test_circular_trajectory_projection_matrices():
    m = PB2D.circular_trajectory_projection_matrices(8, 101, start_angle=10.0, angle_range=180.0, pixel_size=0.5,
                                                     cor_offset=2.0, rotation_center_slope=0.25)
    assert m.shape == (8, 1, 4)
    theta = np.deg2rad(np.linspace(10.0, 190.0, 8, endpoint=False))
    np.testing.assert_allclose(m[:, 0, 0], np.cos(theta) / 0.5)
    np.testing.assert_allclose(m[:, 0, 1], np.sin(theta) / 0.5)
    np.testing.assert_allclose(m[:, 0, 2], 0.25)  # slope in pixels per row r
    np.testing.assert_allclose(m[:, 0, 3], 50.0 + 2.0)  # axis at (N-1)/2 + cor_offset in the row r = 0
    assert PB2D._pb2d_tool.detector_pixel_size(m) == pytest.approx(0.5)


def test_ramp_filter_kernel():
    # A centered delta gives the Ram-Lak kernel times the pixel size: 1/(4w), 0, -1/(pi^2 n^2 w)
    w = 0.5
    p = np.zeros((1, 3, 65))
    p[..., 32] = 1.0
    q = PB2D.ramp_filter(p, w)
    np.testing.assert_allclose(q[0, 1, 32], 1.0 / (4 * w), rtol=1e-6)
    np.testing.assert_allclose(q[0, 1, 33], -1.0 / (np.pi ** 2 * w), rtol=1e-6)
    np.testing.assert_allclose(q[0, 1, 34], 0.0, atol=1e-7)
    for f in PB2D.FILTERS:
        assert PB2D.ramp_filter(p, w, f).shape == p.shape
    with pytest.raises(ValueError):
        PB2D.ramp_filter(p, w, "parzen")


@binaries
def test_projector_backprojector_adjoint():
    rng = np.random.default_rng(1)
    x = rng.random((2, 40, 40)).astype(np.float32)
    y = rng.random((2, 50, 60)).astype(np.float32)
    geometry = dict(cor_offset=1.5, voxel_sizex=0.7, pixel_size=0.9)
    Ax = PB2D.pb2d_projector().project(x, detector_sizex=60, n_angles=50, **geometry)
    Aty = PB2D.pb2d_backprojector().backproject(y, volume_sizex=40, **geometry)
    assert Ax.shape == (2, 50, 60) and Aty.shape == (2, 40, 40)
    lhs, rhs = (Ax.astype(np.float64) * y).sum(), (x.astype(np.float64) * Aty).sum()
    assert lhs == pytest.approx(rhs, rel=1e-5)


@binaries
@pytest.mark.parametrize("angle_range", [180.0, 360.0])
def test_fbp_reconstructs_values(angle_range):
    vol = np.stack([DISC, 2 * DISC])
    sino = PB2D.pb2d_projector().project(vol, detector_sizex=144, n_angles=360, angle_range=angle_range, cor_offset=3.0)
    rec = PB2D.pb2d_backprojector().fbp(sino, volume_sizex=N, angle_range=angle_range, cor_offset=3.0)
    assert rec.shape == (2, N, N)
    np.testing.assert_allclose([rec[0][INNER].mean(), rec[1][INNER].mean()], [1.0, 2.0], rtol=2e-3)


@binaries
def test_fbp_physical_units_and_matrices():
    geometry = dict(pixel_size=2.0, voxel_sizex=0.5)
    sino = PB2D.pb2d_projector().project(DISC, detector_sizex=72, n_angles=360, **geometry)
    rec = PB2D.pb2d_backprojector().fbp(sino, volume_sizex=N, **geometry)
    assert rec[0][INNER].mean() == pytest.approx(1.0, rel=3e-3)
    # the pixel size is taken from the matrices, 2x4 matrices use their first row
    m1 = PB2D.circular_trajectory_projection_matrices(360, 72, pixel_size=2.0)
    m2 = np.concatenate([m1, np.tile([[[0.0, 0.0, 0.5, 0.0]]], (360, 1, 1))], axis=1)
    rec2 = PB2D.pb2d_backprojector().fbp(sino, m2, volume_sizex=N, voxel_sizex=0.5)
    np.testing.assert_allclose(rec2, rec, atol=1e-5)


@binaries
@pytest.mark.parametrize("algorithm", PB2D.ALGORITHMS)
def test_krylov(algorithm):
    vol = np.stack([DISC, 2 * DISC])
    sino = PB2D.pb2d_projector().project(vol, detector_sizex=144, n_angles=180, angle_range=180.0)
    rec = PB2D.pb2d_krylov().reconstruct(sino, volume_sizex=N, algorithm=algorithm, max_iterations=30,
                                         angle_range=180.0)
    assert rec.shape == (2, N, N)
    np.testing.assert_allclose([rec[0][INNER].mean(), rec[1][INNER].mean()], [1.0, 2.0], rtol=5e-3)
    with pytest.raises(ValueError):
        PB2D.pb2d_krylov().reconstruct(sino, algorithm="sirt")


@binaries
def test_rotation_center_slope():
    # The binaries evaluate the 1x4 matrix once per call at the slab center row
    # r = z_level2d_origin + slab_from + (slab_size - 1) / 2, the axis is at (N-1)/2 + cor_offset + slope * r
    vol = np.stack([DISC] * 3)
    P, B = PB2D.pb2d_projector(), PB2D.pb2d_backprojector()
    scan = dict(n_angles=180, angle_range=180.0)

    def ref(cor):
        return P.project(DISC, detector_sizex=144, cor_offset=cor, **scan)[0]

    sino = P.project(vol, detector_sizex=144, cor_offset=1.0, rotation_center_slope=2.0, **scan)
    for z in range(3):  # r = 1
        np.testing.assert_allclose(sino[z], ref(3.0), rtol=1e-5, atol=1e-4)
    slab = P.project(vol, detector_sizex=144, cor_offset=1.0, rotation_center_slope=2.0, slab_from=1, **scan)
    np.testing.assert_allclose(slab, np.stack([ref(4.0)] * 2), rtol=1e-5, atol=1e-4)  # r = 1.5
    level = P.project(vol[:1], detector_sizex=144, cor_offset=1.0, rotation_center_slope=2.0, z_level2d_origin=2.5,
                      **scan)
    np.testing.assert_allclose(level[0], ref(6.0), rtol=1e-5, atol=1e-4)  # r = 2.5
    # the backprojector uses the same row, here the sinogram rows 1, 2 given with z_level2d_origin = 1
    rec = B.fbp(slab, volume_sizex=N, cor_offset=1.0, rotation_center_slope=2.0, z_level2d_origin=1.0,
                angle_range=180.0)
    np.testing.assert_allclose([r[INNER].mean() for r in rec], [1.0] * 2, rtol=2e-3)
    np.testing.assert_allclose(rec, B.fbp(slab, volume_sizex=N, cor_offset=4.0, angle_range=180.0), rtol=1e-5, atol=1e-4)


@binaries
def test_slabs_files_and_errors(tmp_path):
    rng = np.random.default_rng(2)
    vol = rng.random((3, 32, 32)).astype(np.float32)
    P, B = PB2D.pb2d_projector(), PB2D.pb2d_backprojector()
    full = P.project(vol, detector_sizex=48, n_angles=20, output_file=str(tmp_path / "p.den"))
    np.testing.assert_array_equal(PB2D._pb2d_tool.read_sinogram(str(tmp_path / "p.den")), full)
    slab = P.project(vol, detector_sizex=48, n_angles=20, slab_from=1, slab_size=2)
    np.testing.assert_allclose(slab, full[1:], rtol=1e-5, atol=1e-5)
    # a file as input gives the same result as the array
    bp_file = B.backproject(str(tmp_path / "p.den"), volume_sizex=32)
    bp = B.backproject(full, volume_sizex=32)
    np.testing.assert_array_equal(bp_file, bp)
    np.testing.assert_allclose(B.backproject(full, volume_sizex=32, slab_from=2, slab_size=1), bp[2:], rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError):
        P.project(vol, detector_sizex=48, n_angles=20, detector_sizey=2)
    with pytest.raises(ValueError):
        B.backproject(full, scaling="cosine")
    with pytest.raises(RuntimeError):
        PB2D.pb2d_projector(extra_args=["--no-such-option"]).project(vol, detector_sizex=48, n_angles=20)


@binaries
def test_relative_paths_and_clean_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sino = PB2D.pb2d_projector().project(DISC, detector_sizex=144, n_angles=90, angle_range=180.0, output_file="p.den")
    rec = PB2D.pb2d_krylov().reconstruct("p.den", volume_sizex=N, algorithm="pdhg", max_iterations=3, angle_range=180.0,
                                         output_file="r.den")
    assert rec.shape == (1, N, N) and sino.shape == (1, 90, 144)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["p.den", "r.den"]
