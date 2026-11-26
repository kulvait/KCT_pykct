#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for pykct core functionality.
"""

import pytest
import numpy as np
import os
import tempfile
from denpy import DEN


class TestPb2dProjector:
    """Tests for pb2d_projector class."""

    def test_platform_detection(self):
        """Test automatic OpenCL platform detection."""
        from pykct import pb2d_projector
        
        # Should not raise; None lets the C++ runtime auto-detect
        projector = pb2d_projector(verbose=False)
        assert projector.platform_id is None
    
    def test_projection_matrices_creation(self):
        """Test projection matrix creation."""
        from pykct import pb2d_projector
        
        projector = pb2d_projector(verbose=False)
        
        with tempfile.NamedTemporaryFile(suffix='.den', delete=False) as f:
            temp_file = f.name
        
        try:
            matrices = projector.create_projection_matrices(
                output_file=temp_file,
                n_angles=36,
                detector_sizex=64,
                cor_offset=0.0
            )
            
            assert matrices.shape == (36, 2, 4)
            assert DEN.readHeader(temp_file)['shape'] == (36, 2, 4)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_simple_projection(self):
        """Test simple projection."""
        from pykct import pb2d_projector
        
        # Create simple image
        image = np.zeros((64, 64), dtype=np.float32)
        image[30:34, 30:34] = 1.0
        
        projector = pb2d_projector(verbose=False)
        
        sinogram = projector.project_phantom(
            image,
            n_angles=36,
            detector_sizex=64,
            cor_offset=0.0
        )
        
        assert sinogram.shape == (1, 36, 64)
        assert sinogram.max() > 0
    
    def test_cor_offset(self):
        """Test projection with CoR offset."""
        from pykct import pb2d_projector
        
        # Create centered ball
        size = 64
        image = np.zeros((size, size), dtype=np.float32)
        center = (size - 1) / 2
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= 5**2
        image[mask] = 1.0
        
        projector = pb2d_projector(verbose=False)
        
        # Project with CoR offset
        sino_offset = projector.project_phantom(
            image,
            n_angles=36,
            detector_sizex=64,
            cor_offset=5.0
        )
        
        # Should have different sinogram than CoR=0
        sino_zero = projector.project_phantom(
            image,
            n_angles=36,
            detector_sizex=64,
            cor_offset=0.0
        )
        
        # Not equal due to CoR offset
        assert not np.allclose(sino_offset, sino_zero)


class TestMultiSlice:
    """Projection of (nz, ny, nx) volumes."""

    def test_slices_project_independently_in_order(self):
        from pykct import Circle, create_phantom2d, project_image

        v = np.stack([
            create_phantom2d(64, Circle(4, x_offset=-15)),
            create_phantom2d(64, Circle(6, x_offset=10, y_offset=5)),
        ])
        sino = project_image(v, n_angles=16, detector_sizex=64, cor_offset=2.5)
        assert sino.shape == (2, 16, 64)
        for z in range(2):
            single = project_image(v[z], n_angles=16, detector_sizex=64, cor_offset=2.5)
            np.testing.assert_allclose(sino[z], single[0], rtol=1e-5, atol=1e-5)

    def test_bad_ndim(self):
        from pykct import project_image

        with pytest.raises(ValueError):
            project_image(np.zeros((2, 2, 8, 8)), n_angles=4, detector_sizex=8)


def _centroid(image):
    y, x = np.indices(image.shape)
    m = image.sum()
    return (x * image).sum() / m, (y * image).sum() / m


class TestShapes:
    """Shapes drawn by create_phantom2d."""

    def test_circle_binary(self):
        from pykct import Circle, create_phantom2d

        img = create_phantom2d(64, Circle(radius=5))
        assert img.shape == (64, 64) and img.dtype == np.float32
        assert img.max() == 1.0
        assert img[31, 31] == 1.0 and img[0, 0] == 0.0
        cx, cy = _centroid(img)
        assert abs(cx - 31.5) < 1e-9 and abs(cy - 31.5) < 1e-9

    @pytest.mark.parametrize("size", [63, 64])
    @pytest.mark.parametrize("k", [1, 2, 3, 5])
    def test_centred_shapes_are_point_symmetric(self, size, k):
        """Pixels exactly on a boundary are treated alike on both sides."""
        from pykct import Circle, Rectangle, create_phantom2d

        kw = dict(supersampling=k > 1, supersampling_factor=k)
        for shape in (Circle(5), Circle(7.5), Rectangle(10, 6)):
            img = create_phantom2d(size, shape, **kw)
            np.testing.assert_array_equal(img, img[::-1, ::-1])

    def test_position_center_and_offset(self):
        from pykct import Circle, create_phantom2d

        a = create_phantom2d(64, Circle(5, x_center=20, y_center=40, x_offset=1.5))
        b = create_phantom2d(64, Circle(5, x_offset=20 + 1.5 - 31.5, y_offset=40 - 31.5))
        np.testing.assert_array_equal(a, b)
        cx, cy = _centroid(create_phantom2d(64, Circle(5, x_offset=7, y_offset=-3),
                                            antialias=True, antialias_factor=8))
        assert abs(cx - 38.5) < 0.02 and abs(cy - 28.5) < 0.02

    def test_values_add_up(self):
        from pykct import Circle, create_phantom2d

        img = create_phantom2d(64, [Circle(10), Circle(4, value=0.5),
                                    Circle(3, x_offset=20, y_offset=20, value=2.0)])
        assert np.isclose(img[32, 32], 1.5)
        assert np.isclose(img.max(), 2.0)

    def test_hollow_circle_complements_circle(self):
        from pykct import Circle, HollowCircle, create_phantom2d

        ring = create_phantom2d(64, HollowCircle(radius=20, inner_radius=8, x_offset=3.3))
        inner = create_phantom2d(64, Circle(8, x_offset=3.3))
        np.testing.assert_array_equal(ring + inner, create_phantom2d(64, Circle(20, x_offset=3.3)))
        with pytest.raises(ValueError):
            create_phantom2d(16, HollowCircle(radius=4, inner_radius=4))

    def test_ellipse_rectangle_triangle_areas(self):
        from pykct import Ellipse, Rectangle, Triangle, create_phantom2d

        aa = dict(antialias=True, antialias_factor=8)
        e = create_phantom2d(128, Ellipse(30, 12, angle_deg=25), **aa)
        assert abs(e.sum() / (np.pi * 30 * 12) - 1) < 2e-3
        r = create_phantom2d(64, Rectangle(30, 10, angle_deg=90), **aa)
        assert abs(r.sum() - 300) < 0.5
        rows = np.flatnonzero(r.sum(axis=1) > 0.5)
        cols = np.flatnonzero(r.sum(axis=0) > 0.5)
        assert np.ptp(rows) > np.ptp(cols)        # rotated by 90 deg: tall
        t = create_phantom2d(64, Triangle(((0, -10), (10, 10), (-10, 10))), **aa)
        assert abs(t.sum() - 200) < 0.5
        t_rev = create_phantom2d(64, Triangle(((0, -10), (-10, 10), (10, 10))), **aa)
        np.testing.assert_array_equal(t, t_rev)   # vertex order does not matter

    def test_rotation_direction(self):
        """Positive angle turns +x towards +y (row index increasing)."""
        from pykct import Rectangle, create_phantom2d

        img = create_phantom2d(65, Rectangle(40, 2, angle_deg=45, value=1.0))
        assert img[32 + 10, 32 + 10] == 1.0 and img[32 - 10, 32 + 10] == 0.0

    def test_gaussian_dot_mass(self):
        from pykct import GaussianDot, create_phantom2d

        g = create_phantom2d(64, GaussianDot(sigma=3, x_offset=5, value=2.0))
        assert abs(g.sum() / (2 * 2 * np.pi * 9) - 1) < 1e-6
        assert np.isclose(g.max(), 2.0, rtol=1e-6) or g.max() < 2.0

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_supersampling_same_phantom_finer_grid(self, k):
        from pykct import Circle, GaussianDot, create_phantom2d

        img = create_phantom2d(64, [Circle(6, x_offset=10.3, y_offset=-7.2),
                                    GaussianDot(2.0, x_offset=-12, y_offset=9, value=0.0)],
                               antialias=True, antialias_factor=4,
                               supersampling=True, supersampling_factor=k)
        n = 64 * k
        assert img.shape == (n, n)
        cx, cy = _centroid(img)
        # back to grid pixels: the disc stays at (41.8, 24.3)
        assert abs((cx - (n - 1) / 2) / k + 31.5 - 41.8) < 0.02
        assert abs((cy - (n - 1) / 2) / k + 31.5 - 24.3) < 0.02
        assert abs(img.sum() / k ** 2 / (np.pi * 36) - 1) < 0.02

    def test_factor_validation(self):
        from pykct import Circle, create_phantom2d

        for bad in (dict(antialias=True), dict(antialias_factor=4),
                    dict(supersampling=True), dict(supersampling_factor=2),
                    dict(supersampling=True, supersampling_factor=2.5)):
            with pytest.raises(ValueError):
                create_phantom2d(8, Circle(2), **bad)
        with pytest.raises(TypeError):
            create_phantom2d(8, (1, 2, 3))


class TestPhantomObjects:
    """draw / add_to of the object classes and the image objects."""

    def test_draw_equals_create_phantom2d(self):
        from pykct import Circle, Ellipse, create_phantom2d

        e = Ellipse(20, 8, angle_deg=30, x_offset=5.5)
        np.testing.assert_array_equal(e.draw(64), create_phantom2d(64, e))
        np.testing.assert_array_equal(
            e.draw(64, antialias_factor=4, supersampling_factor=2),
            create_phantom2d(64, e, antialias=True, antialias_factor=4,
                             supersampling=True, supersampling_factor=2))
        canvas = np.zeros((64, 64))
        Circle(10).add_to(canvas, 64)
        e.add_to(canvas, 64)
        np.testing.assert_allclose(canvas, create_phantom2d(64, [Circle(10), e]))

    def test_validation_at_construction(self):
        from pykct import Circle, GaussianDot, HollowCircle, Rectangle, Triangle

        for make in (lambda: Circle(0), lambda: HollowCircle(3, 5), lambda: Rectangle(1, -1),
                     lambda: Triangle(((0, 0), (1, 1), (2, 2))), lambda: GaussianDot(0)):
            with pytest.raises(ValueError):
                make()
        with pytest.raises(ValueError):
            Circle(3).draw(16, antialias_factor=0)

    def test_objects_are_immutable_and_keyword_friendly(self):
        import dataclasses
        from pykct import Circle

        c = Circle(radius=5, x_offset=2, value=0.5)
        assert c == Circle(5, x_offset=2, value=0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.radius = 3

    def test_custom_shape_subclass(self):
        """A new shape only needs profile and extent."""
        from dataclasses import dataclass
        from pykct import Circle, PhantomShape, create_phantom2d

        @dataclass(frozen=True)
        class Square(PhantomShape):
            side: float

            def extent(self):
                return self.side

            def profile(self, x, y):
                return ((np.abs(x) <= self.side / 2) & (np.abs(y) <= self.side / 2)).astype(float)

        img = create_phantom2d(32, [Square(10, x_offset=-5), Circle(3)])
        assert img.max() == 2.0 and img[15, 11] == 1.0

    @pytest.mark.parametrize("n", [64, 65])
    def test_denpy_phantom_object(self, n):
        from denpy import PHANTOM
        from pykct import DenpyPhantom2D, create_phantom2d

        obj = DenpyPhantom2D(size_px=n, phantom="Toft2D")
        np.testing.assert_array_equal(create_phantom2d(n, obj),
                                      PHANTOM.phantom2d("Toft2D", n).astype(np.float32))
        fine = obj.draw(n, supersampling_factor=2)
        np.testing.assert_array_equal(fine, PHANTOM.phantom2d("Toft2D", 2 * n).astype(np.float32))
        # smaller object (same parity as the grid) placed off-centre
        m = 32 + n % 2
        img = create_phantom2d(n, DenpyPhantom2D(size_px=m, x_offset=10, y_offset=-4))
        r0, c0 = (n - m) // 2 - 4, (n - m) // 2 + 10
        np.testing.assert_array_equal(img[r0:r0 + m, c0:c0 + m],
                                      PHANTOM.phantom2d("SheppLogan2D", m).astype(np.float32))
        assert img.sum() == pytest.approx(img[r0:r0 + m, c0:c0 + m].sum())

    def test_image_object_alignment(self):
        from pykct import DenpyPhantom2D, create_phantom2d

        with pytest.raises(ValueError):          # half-pixel shift
            create_phantom2d(64, DenpyPhantom2D(size_px=32, x_offset=0.5))
        with pytest.raises(ValueError):          # odd object on an even grid, centred
            create_phantom2d(64, DenpyPhantom2D(size_px=31))
        # half a grid pixel is a whole canvas pixel with supersampling 2
        img = create_phantom2d(64, DenpyPhantom2D(size_px=32, x_offset=0.5),
                               supersampling=True, supersampling_factor=2)
        assert img.shape == (128, 128)

    def test_array_image(self):
        from pykct import ArrayImage, Circle, create_phantom2d

        arr = np.arange(16, dtype=float).reshape(4, 4)
        img = create_phantom2d(8, [ArrayImage(array=arr, x_offset=-2, y_offset=-2)])
        np.testing.assert_array_equal(img[0:4, 0:4], arr)
        fine = create_phantom2d(8, ArrayImage(array=arr), supersampling=True,
                                supersampling_factor=3)
        np.testing.assert_array_equal(fine[6:18, 6:18], np.kron(arr, np.ones((3, 3))))
        # antialiasing leaves a piecewise constant image unchanged
        np.testing.assert_allclose(
            create_phantom2d(8, ArrayImage(array=arr), antialias=True, antialias_factor=2),
            create_phantom2d(8, ArrayImage(array=arr)))
        # mixed with a shape
        mixed = create_phantom2d(8, [ArrayImage(array=np.ones((8, 8))), Circle(1, value=2.0)])
        assert mixed.max() == 3.0


class TestSheppLogan:

    def test_shepp_logan_phantom(self):
        from pykct import create_shepp_logan_phantom2d

        phantom = create_shepp_logan_phantom2d(size=64)
        assert phantom.shape == (64, 64)
        assert phantom.dtype == np.float32
        # Original Shepp-Logan: skull 2.0, brain 1.02
        assert np.isclose(phantom.max(), 2.0)

    @pytest.mark.parametrize("n", [64, 129])
    def test_shepp_logan_variants_match_denpy(self, n):
        """The Ellipse shapes reproduce denpy.PHANTOM.phantom2d exactly."""
        from pykct import SHEPP_LOGAN_VARIANTS, create_shepp_logan_phantom2d
        from denpy import PHANTOM

        for name in SHEPP_LOGAN_VARIANTS:
            expected = PHANTOM.phantom2d(phantom=name, n=n).astype(np.float32)
            np.testing.assert_array_equal(create_shepp_logan_phantom2d(n, phantom=name), expected)

    def test_shepp_logan_supersampling(self):
        from pykct import create_shepp_logan_phantom2d

        a = create_shepp_logan_phantom2d(64, antialias=True, antialias_factor=8)
        b = create_shepp_logan_phantom2d(64, supersampling=True, supersampling_factor=4)
        assert b.shape == (256, 256)
        b_binned = b.reshape(64, 4, 64, 4).mean(axis=(1, 3))
        assert np.abs(a - b_binned).mean() < 0.01

    def test_shepp_logan_unknown_variant(self):
        from pykct import create_shepp_logan_phantom2d

        with pytest.raises(ValueError):
            create_shepp_logan_phantom2d(64, phantom="nonsense")

    def test_legacy_shepp_logan_phantom(self):
        """Legacy phantom is reproducible and differs from the standard one."""
        from pykct import create_legacy_shepp_logan_phantom2d

        with pytest.warns(FutureWarning):
            legacy = create_legacy_shepp_logan_phantom2d(64)
        assert legacy.shape == (64, 64)
        assert np.isclose(legacy.max(), 1.2)
        # Head is wider than tall (swapped semi-axes)
        rows = np.flatnonzero(legacy.any(axis=1))
        cols = np.flatnonzero(legacy.any(axis=0))
        assert np.ptp(cols) > np.ptp(rows)


class TestSinogramGeneration:

    def test_sinogram2d_shapes(self, tmp_path):
        from pykct import Circle, create_sinogram2d, shepp_logan_shapes2d

        sino, img = create_sinogram2d(64, Circle(5), n_angles=36, detector_size=80,
                                      return_phantom=True)
        assert sino.shape == (36, 80) and img.shape == (64, 64)
        out = tmp_path / "s.den"
        sino = create_sinogram2d(64, shepp_logan_shapes2d(64), n_angles=36, output_file=str(out))
        assert sino.shape == (36, 64)
        assert DEN.getNumpyArray(str(out)).shape == (1, 36, 64)

    @pytest.mark.parametrize("k", [1, 2, 5])
    @pytest.mark.parametrize("size,ndet", [(64, 64), (63, 64), (64, 63)])
    def test_supersampled_sinogram_geometry(self, k, size, ndet):
        """Centred disc stays on the axis, an off-centre disc follows
        cor + x cos + y sin, line integrals in grid pixel units, for every k."""
        from pykct import Circle, create_sinogram2d

        disc = Circle(5, x_offset=12, y_offset=-7)
        kw = dict(antialias=True, antialias_factor=4, supersampling=k > 1, supersampling_factor=k)
        sino = create_sinogram2d(size, disc, n_angles=360, detector_size=ndet,
                                 cor_offset=1.5, **kw)
        theta = np.radians(np.arange(360))
        centroid = (sino * np.arange(ndet)).sum(axis=1) / sino.sum(axis=1)
        expected = (ndet - 1) / 2 + 1.5 + 12 * np.cos(theta) - 7 * np.sin(theta)
        assert np.abs(centroid - expected).max() < 0.02
        assert abs(sino.sum(axis=1).mean() / (np.pi * 25) - 1) < 0.02
        for aa in (False, True):     # binary and antialiased discs
            kw.update(antialias=aa, antialias_factor=4 if aa else 1)
            centred = create_sinogram2d(size, Circle(5), n_angles=36, detector_size=ndet, **kw)
            c = (centred * np.arange(ndet)).sum(axis=1) / centred.sum(axis=1)
            assert np.abs(c - (ndet - 1) / 2).max() < 1e-3


class TestGeometry:
    """Rotation axis offset convention of the projector."""

    @pytest.mark.parametrize("cor_offset", [0.0, 1.3, -3.0])
    def test_matrices_match_createProjectionMatrices(self, cor_offset, tmp_path):
        """Rows equal createProjectionMatrices.py --material-ct-convention
        --rotation-center-offset-pix cor_offset: a = [cos, sin, 0]/PX,
        px0 = (N-1)/2 + cor_offset, b = [0, 0, 1]/PY."""
        from pykct import create_projection_matrices

        N, M = 64, 3
        m = create_projection_matrices(str(tmp_path / "m.den"), n_angles=12,
                                       detector_sizex=N, detector_sizey=M,
                                       cor_offset=cor_offset)
        theta = np.radians(np.linspace(0, 360, 12, endpoint=False))
        np.testing.assert_allclose(m[:, 0, 0], np.cos(theta), atol=1e-12)
        np.testing.assert_allclose(m[:, 0, 1], np.sin(theta), atol=1e-12)
        np.testing.assert_allclose(m[:, 0, 3], (N - 1) / 2 + cor_offset, atol=1e-12)
        np.testing.assert_allclose(m[:, 1, :3], [[0, 0, 1]] * 12)
        np.testing.assert_allclose(m[:, 1, 3], (M - 1) / 2)

    @pytest.mark.parametrize("cor_offset", [0.0, 1.3, -3.0, 5.0])
    def test_rotation_axis_position(self, cor_offset):
        """Rotation axis is imaged at detector position (N-1)/2 + cor_offset.

        For a full 360 degree parallel-beam sinogram the mass-weighted mean
        detector position equals the image of the rotation axis.
        """
        from pykct import Circle, calculate_centroid, create_phantom2d, project_image

        phantom = create_phantom2d(128, Circle(8, x_offset=15), antialias=True, antialias_factor=4)
        sino = project_image(phantom, n_angles=180, detector_sizex=128,
                             cor_offset=cor_offset)
        cx, _ = calculate_centroid(sino)
        assert abs(cx - (63.5 + cor_offset)) < 0.01


class TestCentroid:
    """Tests for centroid calculation."""

    def test_centroid_centered(self, tmp_path):
        from pykct import Circle, calculate_centroid, create_phantom2d

        phantom = create_phantom2d(64, Circle(5))
        f = str(tmp_path / "c.den")
        DEN.storeNdarrayAsDEN(f, phantom[None], force=True)
        cx, cy = calculate_centroid(f)
        assert abs(cx - 31.5) < 0.1
        assert abs(cy - 31.5) < 0.1

    def test_centroid_array(self):
        """calculate_centroid accepts arrays and handles empty frames."""
        from pykct import Circle, calculate_centroid, create_phantom2d

        phantom = create_phantom2d(64, Circle(5, x_offset=7, y_offset=-3))
        cx, cy = calculate_centroid(phantom)
        assert abs(cx - (31.5 + 7)) < 0.1
        assert abs(cy - (31.5 - 3)) < 0.1
        assert calculate_centroid(np.zeros((8, 8))) is None


class TestScripts:
    """Smoke tests of command line scripts."""

    def _run(self, module, *args):
        import subprocess
        import sys
        return subprocess.run([sys.executable, "-m", module, *args],
                              capture_output=True, text=True, check=True)

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_shepp_logan_script_uses_library(self, tmp_path):
        from pykct import create_legacy_shepp_logan_phantom2d, create_shepp_logan_phantom2d

        out = tmp_path / "sl.den"
        self._run("pykct.scripts.create_shepp_logan", str(out), "-n", "32",
                  "--supersampling-factor", "2")
        np.testing.assert_array_equal(
            DEN.getNumpyArray(str(out)).reshape(64, 64),
            create_shepp_logan_phantom2d(32, supersampling=True, supersampling_factor=2))
        out = tmp_path / "legacy.den"
        self._run("pykct.scripts.create_shepp_logan", str(out), "-n", "32", "--phantom", "legacy")
        np.testing.assert_array_equal(DEN.getNumpyArray(str(out)).reshape(32, 32),
                                      create_legacy_shepp_logan_phantom2d(32))

    def test_ball_scripts_and_centroid(self, tmp_path):
        from pykct import Circle, create_phantom2d

        phantom_file = tmp_path / "ball.den"
        self._run("pykct.scripts.create_ball_phantom", str(phantom_file),
                  "-n", "32", "-r", "4", "--x-offset", "3", "--antialias")
        np.testing.assert_array_equal(
            DEN.getNumpyArray(str(phantom_file)).reshape(32, 32),
            create_phantom2d(32, Circle(4, x_offset=3), antialias=True, antialias_factor=4))

        sino_file = tmp_path / "sino.den"
        self._run("pykct.scripts.create_ball_sinogram", str(sino_file),
                  "-n", "32", "-r", "4", "--n-angles", "16",
                  "--detector-size", "32", "--cor-offset", "2")
        assert DEN.getNumpyArray(str(sino_file)).shape == (1, 16, 32)

        csv = tmp_path / "c.csv"
        res = self._run("pykct.scripts.den_centroid", str(sino_file), "--csv", str(csv))
        assert "Offset X" in res.stdout
        offset_x = np.loadtxt(str(csv), delimiter=",", skiprows=1)[3]
        assert abs(offset_x - 2.0) < 0.05

    def test_detector_center_offsetvx_is_minus_cor_offset(self, tmp_path):
        a, b = tmp_path / "a.den", tmp_path / "b.den"
        common = ["-n", "32", "-r", "4", "--x-offset", "5", "--n-angles", "16",
                  "--detector-size", "32"]
        self._run("pykct.scripts.create_ball_sinogram", str(a), *common, "--cor-offset", "3")
        self._run("pykct.scripts.create_ball_sinogram", str(b), *common,
                  "--detector-center-offsetvx", "-3")
        # OpenCL projection is reproducible only to float32 rounding
        np.testing.assert_allclose(DEN.getNumpyArray(str(a)), DEN.getNumpyArray(str(b)),
                                   rtol=1e-5, atol=1e-5)

    def test_refuses_overwrite(self, tmp_path):
        import subprocess
        import sys

        out = tmp_path / "x.den"
        out.write_bytes(b"")
        res = subprocess.run([sys.executable, "-m", "pykct.scripts.create_shepp_logan",
                              str(out), "-n", "8"], capture_output=True, text=True)
        assert res.returncode == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
