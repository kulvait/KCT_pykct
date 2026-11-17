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
        from pykct import create_ball_phantom, project_image

        v = np.concatenate([
            create_ball_phantom(64, 4, x_offset=-15),
            create_ball_phantom(64, 6, x_offset=10, y_offset=5),
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


class TestPhantoms:
    """Tests for phantom generation."""
    
    def test_ball_phantom(self):
        """Test ball phantom creation."""
        from pykct import create_ball_phantom
        
        phantom = create_ball_phantom(
            size=64, radius=5, x_offset=0.0, y_offset=0.0,
            antialias=False
        )
        
        assert phantom.shape == (1, 64, 64)
        assert phantom.max() == 1.0
    
    def test_multi_ball_phantom(self):
        from pykct import create_ball_phantom, create_multi_ball_phantom

        single = create_ball_phantom(64, 5, x_offset=7, y_offset=-3)
        np.testing.assert_array_equal(
            create_multi_ball_phantom(64, [(7, -3, 5)]), single)

        multi = create_multi_ball_phantom(
            64, [(0, 0, 10), (0, 0, 4, 0.5),
                 {"x_offset": 20, "y_offset": 20, "radius": 3, "value": 2.0}],
            antialias=False)
        assert multi.shape == (1, 64, 64)
        assert multi.dtype == np.float32
        assert np.isclose(multi[0, 32, 32], 1.5)     # overlapping discs add
        assert np.isclose(multi.max(), 2.0)
        with pytest.raises(ValueError):
            create_multi_ball_phantom(64, [(0, 0)])

    def test_shepp_logan_phantom(self):
        """Test Shepp-Logan phantom creation."""
        from pykct import create_shepp_logan_phantom
        
        phantom = create_shepp_logan_phantom(size=64)
        
        assert phantom.shape == (64, 64)
        assert phantom.dtype == np.float32
        # Original Shepp-Logan: skull 2.0, brain 1.02
        assert np.isclose(phantom.max(), 2.0)

    def test_shepp_logan_variants_match_denpy(self):
        """Standard variants are delegated to denpy.PHANTOM.phantom2d."""
        from pykct import create_shepp_logan_phantom
        from denpy import PHANTOM

        for name in ("SheppLogan2D", "Octave2D", "Toft2D"):
            expected = PHANTOM.phantom2d(phantom=name, n=64).astype(np.float32)
            np.testing.assert_array_equal(
                create_shepp_logan_phantom(64, phantom=name), expected)

    def test_shepp_logan_unknown_variant(self):
        from pykct import create_shepp_logan_phantom

        with pytest.raises(ValueError):
            create_shepp_logan_phantom(64, phantom="nonsense")

    def test_legacy_shepp_logan_phantom(self):
        """Legacy phantom is reproducible and differs from the standard one."""
        from pykct import create_shepp_logan_phantom, create_legacy_shepp_logan_phantom

        with pytest.warns(FutureWarning):
            legacy = create_legacy_shepp_logan_phantom(64)
        with pytest.warns(FutureWarning):
            via_name = create_shepp_logan_phantom(64, phantom="legacy")
        np.testing.assert_array_equal(via_name, legacy)
        assert legacy.shape == (64, 64)
        assert np.isclose(legacy.max(), 1.2)
        # Head is wider than tall (swapped semi-axes)
        rows = np.flatnonzero(legacy.any(axis=1))
        cols = np.flatnonzero(legacy.any(axis=0))
        assert np.ptp(cols) > np.ptp(rows)


class TestSinogramGeneration:
    """Tests for sinogram generation functions."""
    
    def test_ball_sinogram(self):
        """Test ball sinogram creation."""
        from pykct import create_ball_sinogram
        
        sino = create_ball_sinogram(
            size=64, radius=5,
            n_angles=36, detector_size=64,
            cor_offset=0.0
        )
        
        assert sino.shape == (1, 36, 64)
    
    def test_shepp_logan_sinogram(self):
        """Test Shepp-Logan sinogram creation."""
        from pykct import create_shepp_logan_sinogram
        
        sino = create_shepp_logan_sinogram(
            size=64,
            n_angles=36, detector_size=64
        )
        
        assert sino.shape == (1, 36, 64)


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
        from pykct import create_ball_phantom, project_image, calculate_centroid

        phantom = create_ball_phantom(size=128, radius=8, x_offset=15)
        sino = project_image(phantom, n_angles=180, detector_sizex=128,
                             cor_offset=cor_offset)
        cx, _ = calculate_centroid(sino)
        assert abs(cx - (63.5 + cor_offset)) < 0.01


class TestCentroid:
    """Tests for centroid calculation."""
    
    def test_centroid_centered(self):
        """Test centroid of centered circle."""
        from pykct import calculate_centroid, create_ball_phantom
        from denpy import DEN
        
        # Create centered ball
        phantom = create_ball_phantom(
            size=64, radius=5, antialias=False
        )
        
        with tempfile.NamedTemporaryFile(suffix='.den', delete=False) as f:
            temp_file = f.name
        
        try:
            DEN.storeNdarrayAsDEN(temp_file, phantom, force=True)
            
            cx, cy = calculate_centroid(temp_file)
            
            # Center should be at (63/2, 63/2) = (31.5, 31.5)
            expected_center = (64 - 1) / 2.0
            assert abs(cx - expected_center) < 0.1
            assert abs(cy - expected_center) < 0.1
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


    def test_centroid_array(self):
        """calculate_centroid accepts arrays and handles empty frames."""
        from pykct import calculate_centroid, create_ball_phantom

        phantom = create_ball_phantom(size=64, radius=5, x_offset=7, y_offset=-3,
                                      antialias=False)
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
        from pykct import create_shepp_logan_phantom

        for variant in ("SheppLogan2D", "legacy"):
            out = tmp_path / f"{variant}.den"
            self._run("pykct.scripts.create_shepp_logan", str(out),
                      "-n", "32", "--phantom", variant)
            np.testing.assert_array_equal(
                DEN.getNumpyArray(str(out))[0],
                create_shepp_logan_phantom(32, phantom=variant))

    def test_ball_scripts_and_centroid(self, tmp_path):
        from pykct import create_ball_phantom

        phantom_file = tmp_path / "ball.den"
        self._run("pykct.scripts.create_ball_phantom", str(phantom_file),
                  "-n", "32", "-r", "4", "--x-offset", "3", "--antialias")
        np.testing.assert_array_equal(
            DEN.getNumpyArray(str(phantom_file)).reshape(1, 32, 32),
            create_ball_phantom(32, 4, x_offset=3))

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
