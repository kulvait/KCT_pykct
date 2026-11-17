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
pykct-create-ball-phantom --radius 10 --antialias ball.den
pykct-create-shepp-logan sl.den                    # --phantom Toft2D|Octave2D|legacy

# Create sinogram
pykct-create-ball-sinogram ball_sino.den \
    --radius 10 --n-angles 360 --detector-size 256 --cor-offset 5.0

pykct-create-shepp-logan-sinogram sl_sino.den \
    --n-angles 512 --detector-size 256

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
    cor_offset=0.0,         # CoR offset (pixels), axis imaged at (N-1)/2 + cor_offset
    pixel_size=1.0          # Pixel size
)
```

**Projection matrices are created automatically and temporarily!**

Volumes `(nz, ny, nx)` are accepted as well; each slice is projected
independently and `detector_sizey` defaults to `nz`:

```python
import numpy as np
volume = np.concatenate([create_ball_phantom(256, 10), create_shepp_logan_phantom(256)[None]])
sino = project_image(volume, n_angles=360, detector_sizex=256)   # (2, 360, 256)
```

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

**Several balls:** `(x_offset, y_offset, radius[, value])` tuples or dicts,
offsets from the geometric centre, overlapping discs add up:
```python
from pykct import create_multi_ball_phantom

balls = create_multi_ball_phantom(
    size=256,
    balls=[(0, 0, 20), (60, -30, 8, 0.5), (-70, 40, 5)],
)   # shape (1, 256, 256)
```

**Shepp-Logan phantom:**
```python
from pykct import create_shepp_logan_phantom

sl = create_shepp_logan_phantom(size=256)                       # SheppLogan2D (denpy)
toft = create_shepp_logan_phantom(size=256, phantom="Toft2D")
```

`phantom="legacy"` (or `create_legacy_shepp_logan_phantom`) reproduces the
non-standard phantom of pykct <= 0.1.0 (swapped semi-axes, no rotations,
centred at size/2). It is deprecated, emits a `FutureWarning` and will be
removed in a later version; use it only to reproduce earlier experiments.

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
from pykct import pb2d_projector
from denpy import DEN

projector = pb2d_projector()

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

`cor_offset` is the signed distance, in detector pixels, of the principal ray
(the ray through the rotation axis) from the geometric detector centre:

```
axis_position = (N-1)/2 + cor_offset
```

This is `createProjectionMatrices.py --rotation-center-offset-pix`. The
`--detector-center-offsetvx` of
`createCameraMatricesForCircularScanTrajectoryParallelRay3D.py` shifts the
detector centre instead, so it equals `-cor_offset`. Estimators reporting the
axis position in pixel-index units with `(N-1)/2` as detector centre
(e.g. algotom `find_center_vo`) return `(N-1)/2 + cor_offset`.

Geometry otherwise equals the `--material-ct-convention` of the geometry script;
the pb2d projector does not use the detector y row.

### Sinogram Shift Relationship

For parallel beam geometry with 360° scan and no truncation:

```
sinogram(cor + Δcor) = shift(sinogram(cor), +Δcor)
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
│   ├── scripts/              # Thin CLI wrappers over the API above
│   │   ├── _common.py
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

### OpenCL platform selection

Normally leave `platform_id=None`: the C++ runtime picks a platform expected
to perform well (e.g. the first NVIDIA platform) and its last device, to avoid
clashing with the display card or other jobs. Do not pass a dummy `"0:0"`.
Only if that choice fails or a specific device must be used:

```python
from pykct import pb2d_projector

# platform 1, device 2 as listed by `clinfo -l`
projector = pb2d_projector(platform_id="1:2")
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
