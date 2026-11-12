# PyKCT Installation and Usage Guide

## Overview

PyKCT is a clean Python package for KCT C++ parallel beam 2D projector integration. It provides:
- Easy-to-use projection functions (similar to `radon` from scikit-image)
- Automatic projection matrix creation (hidden from users)
- Phantom generation utilities
- Command-line tools for batch processing

## Installation

```bash
cd /workspace/pykct
pip install -e .
```

This installs:
- `pykct` package with all modules
- Command-line scripts in your PATH

## Quick Start

### Python API

```python
import numpy as np
from pykct import project_image, create_ball_sinogram, create_shepp_logan_phantom

# Method 1: Project arbitrary image (like radon)
image = np.zeros((128, 128))
image[50:80, 50:80] = 1.0
sinogram = project_image(image, n_angles=360, detector_sizex=256)

# Method 2: Create sinogram from phantom (simplest!)
ball_sino = create_ball_sinogram(
    size=128, radius=10, n_angles=360, detector_size=256, cor_offset=5.0
)

# Method 3: Use phantom functions
phantom = create_shepp_logan_phantom(size=256)
sl_sino = create_shepp_logan_sinogram(size=256, n_angles=512, detector_size=256)
```

### Command Line

```bash
# Create phantom
pykct-create-ball-phantom --radius 10 --antialias -o ball.den
pykct-create-shepp-logan -o sl.den

# Create sinogram
pykct-create-ball-sinogram \
    --radius 10 --n-angles 360 --detector-size 256 \
    --cor-offset 5.0 --sinogram-file ball_sino.den

pykct-create-shepp-logan-sinogram \
    --n-angles 512 --detector-size 256 \
    --sinogram-file sl_sino.den

# Calculate centroid
pykct-den-centroid sinogram.den
```

## Features

### 1. Simple Projection (No Matrices!)

Just specify the geometry parameters:

```python
from pykct import project_image

sinogram = project_image(
    image=my_image,
    n_angles=512,           # Number of angles
    start_angle=0.0,        # Starting angle (degrees)
    angle_range=360.0,      # Angular range (degrees)
    detector_sizex=256,     # Detector size
    cor_offset=0.0,         # CoR offset (pixels)
    pixel_size=1.0          # Pixel size
)
```

**Projection matrices are created automatically and temporarily!**

### 2. Phantom Generation

**Ball phantom:**
```python
from pykct import create_ball_phantom

# Centered ball
ball = create_ball_phantom(size=256, radius=10)

# Off-center ball
ball = create_ball_phantom(
    size=256, radius=10,
    x_offset=20.0, y_offset=-15.0
)

# With antialiasing
ball = create_ball_phantom(
    size=256, radius=10,
    antialias=True, antialias_factor=4
)
```

**Shepp-Logan phantom:**
```python
from pykct import create_shepp_logan_phantom

sl = create_shepp_logan_phantom(size=256)
```

### 3. Direct Sinogram Generation

Combine phantom creation and projection:

```python
from pykct import create_ball_sinogram, create_shepp_logan_sinogram

# Ball sinogram
ball_sino = create_ball_sinogram(
    size=256, radius=10,          # Phantom
    n_angles=512, detector_size=256,  # Projection
    cor_offset=5.0                 # CoR offset
)

# Shepp-Logan sinogram
sl_sino = create_shepp_logan_sinogram(
    size=256, n_angles=512, detector_size=256
)
```

### 4. Advanced Usage (Custom Matrices)

For full control over projection matrices:

```python
from pykct import KCTProjector
from denpy import DEN

projector = KCTProjector()

# Create custom matrices
matrices = projector.create_projection_matrices(
    output_file="matrices.den",
    n_angles=512,
    start_angle=0.0,
    angle_range=360.0,
    detector_sizex=256,
    cor_offset=0.0
)

# Project with your own volume
volume = DEN.getNumpyArray("volume.den")
sinogram = projector.project(
    volume_file="volume.den",
    matrices_file="matrices.den",
    output_file="projection.den",
    detector_sizex=256
)
```

## Geometry Reference

### Coordinate System

- **Geometric center**: `(N-1)/2` for N pixels
- **Example**: For 256 pixels, center = 127.5

### CoR Offset Convention

- **Positive**: CoR to the RIGHT of detector center
- **Negative**: CoR to the LEFT of detector center

### Sinogram Shift Relationship

For parallel beam geometry with 360° scan and no truncation:

```
sinogram(cor + Δcor) = shift(sinogram(cor), -Δcor)
```

Where `shift` is along the detector axis (x-axis).

## Testing

Run tests:

```bash
cd /workspace/pykct
pytest pykct/tests/ -v
```

## Package Structure

```
pykct/
├── pykct/
│   ├── __init__.py           # Main exports
│   ├── core/
│   │   └── __init__.py       # Projector wrapper and functions
│   ├── phantoms/
│   │   └── __init__.py       # Phantom generation
│   ├── scripts/
│   │   ├── create_ball_phantom.py
│   │   ├── create_shepp_logan.py
│   │   ├── create_ball_sinogram.py
│   │   ├── create_shepp_logan_sinogram.py
│   │   └── den_centroid.py
│   └── tests/
│       └── test_projector.py
├── setup.py
├── README.md
└── INSTALLATION_AND_USAGE.md
```

## Comparison with Previous Approach

### Old way (manual matrices)

```python
# Create matrices with external script
os.system("createCameraMatricesForCircularScanTrajectoryParallelRay3D.py matrices.den")

# Load matrices
from denpy import DEN
matrices = DEN.getNumpyArray("matrices.den")

# Project
os.system("kct-pb2d-projector volume.den matrices.den proj.den ...")
```

### New way (pykct)

```python
from pykct import project_image

sinogram = project_image(
    image=my_image,
    n_angles=512,
    detector_sizex=256,
    cor_offset=5.0
)
```

Much simpler!

## Common Use Cases

### 1. CoR Detection Testing

```python
from pykct import create_ball_sinogram
import numpy as np

# Create sinograms with known CoR offsets
for cor in [-10.0, -5.0, 0.0, 5.0, 10.0]:
    sino = create_ball_sinogram(
        size=128, radius=10,
        n_angles=512, detector_size=256,
        cor_offset=cor
    )
    # Save and test your CoR detection algorithm
```

### 2. Interpolation Validation

```python
from pykct import create_ball_sinogram
from scipy import ndimage

# Project with CoR=0
sino_zero = create_ball_sinogram(
    size=128, radius=10, n_angles=360, detector_size=256, cor_offset=0.0
)

# Shift manually
shifted = ndimage.shift(sino_zero[0, :, :], -10.2, order=3)

# Project with actual CoR=10.2
sino_offset = create_ball_sinogram(
    size=128, radius=10, n_angles=360, detector_size=256, cor_offset=10.2
)

# Compare
diff = shifted - sino_offset[0, :, :]
print(f"L2 error: {np.sqrt(np.sum(diff**2)):.2f}")
```

### 3. Batch Sinogram Generation

```python
from pykct.scripts.create_ball_sinogram import main as ball_sino_main
import sys

# Generate many sinograms
configs = [
    {"radius": 5, "cor_offset": 0.0},
    {"radius": 5, "cor_offset": 5.0},
    {"radius": 10, "cor_offset": 0.0},
    # ... more configs
]

for i, cfg in enumerate(configs):
    # Use command line interface
    sys.argv = [
        "create_ball_sinogram",
        f"--radius={cfg['radius']}",
        f"--cor-offset={cfg['cor_offset']}",
        "--n-angles=512",
        "--detector-size=256",
        f"--sinogram-file=sino_{i}.den"
    ]
    ball_sino_main()
```

## Troubleshooting

### OpenCL platform not detected

```python
# Specify platform explicitly
from pykct import KCTProjector

projector = KCTProjector(platform_id="0:0")  # or "0:1", etc.
```

### Projection errors

Check:
1. Volume size matches expected detector size
2. No truncation (object fully visible)
3. Image values are reasonable (not NaN/Inf)

## Next Steps

After this package is installed, you can:
1. Create comprehensive CoR detection test datasets
2. Benchmark different CoR detection algorithms
3. Validate projection/reconstruction pipelines
4. Generate synthetic CT data for research

All with simple, consistent interfaces!
