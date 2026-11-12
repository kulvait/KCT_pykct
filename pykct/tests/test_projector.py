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


class TestKCTProjector:
    """Tests for KCTProjector class."""
    
    def test_platform_detection(self):
        """Test automatic OpenCL platform detection."""
        from pykct import KCTProjector
        
        # Should not raise
        projector = KCTProjector(verbose=False)
        assert projector.platform_id is not None
    
    def test_projection_matrices_creation(self):
        """Test projection matrix creation."""
        from pykct import KCTProjector
        
        projector = KCTProjector(verbose=False)
        
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
        from pykct import KCTProjector
        
        # Create simple image
        image = np.zeros((64, 64), dtype=np.float32)
        image[30:34, 30:34] = 1.0
        
        projector = KCTProjector(verbose=False)
        
        sinogram = projector.project_phantom(
            image,
            n_angles=36,
            detector_sizex=64,
            cor_offset=0.0
        )
        
        assert sinogram.shape == (36, 1, 64)
        assert sinogram.max() > 0
    
    def test_cor_offset(self):
        """Test projection with CoR offset."""
        from pykct import KCTProjector
        
        # Create centered ball
        size = 64
        image = np.zeros((size, size), dtype=np.float32)
        center = (size - 1) / 2
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= 5**2
        image[mask] = 1.0
        
        projector = KCTProjector(verbose=False)
        
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
    
    def test_shepp_logan_phantom(self):
        """Test Shepp-Logan phantom creation."""
        from pykct import create_shepp_logan_phantom
        
        phantom = create_shepp_logan_phantom(size=64)
        
        assert phantom.shape == (64, 64)
        assert phantom.max() == 1.0


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
        
        assert sino.shape == (36, 1, 64)
    
    def test_shepp_logan_sinogram(self):
        """Test Shepp-Logan sinogram creation."""
        from pykct import create_shepp_logan_sinogram
        
        sino = create_shepp_logan_sinogram(
            size=64,
            n_angles=36, detector_size=64
        )
        
        assert sino.shape == (36, 1, 64)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
