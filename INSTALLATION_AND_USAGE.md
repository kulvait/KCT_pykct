# PyKCT Installation and Usage Guide

## Overview

PyKCT is a clean Python package for KCT C++ parallel beam 2D projector integration. It provides:
- Easy-to-use projection functions (similar to `radon` from scikit-image)
- Automatic projection matrix creation for the projection functions
- `pykct.PM` for explicit parallel ray projection matrices of KCT
- `pykct.PB2D` for projection, backprojection, FBP and CGLS/PDHG by the kct-pb2d-* tools
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
from pykct import (Circle, create_phantom2d, create_shepp_logan_phantom2d,
                   create_sinogram2d, project_image, shepp_logan_shapes2d)

# Method 1: Project arbitrary image (like radon)
image = np.zeros((128, 128))
image[50:80, 50:80] = 1.0
sinogram = project_image(image, n_angles=360, detector_sizex=256)

# Method 2: Sinogram of shapes drawn on the grid (simplest!)
ball_sino = create_sinogram2d(128, Circle(radius=10), n_angles=360, detector_size=256,
                              cor_offset=5.0, antialias=True, antialias_factor=4)

# Method 3: Phantom image, then project
phantom = create_shepp_logan_phantom2d(size=256)
sl_sino = create_sinogram2d(256, shepp_logan_shapes2d(256), n_angles=512, detector_size=256)
```

### Command Line

```bash
# Create phantom
pykct-create-ball-phantom --radius 10 --antialias ball.den
pykct-create-shepp-logan sl.den                    # --phantom Toft2D|Octave2D|legacy
pykct-create-shepp-logan sl5.den --supersampling-factor 5    # (1, 1280, 1280)

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
    pixel_size=1.0          # Detector pixel size in voxels
)
```

**Projection matrices are created automatically and temporarily!**

Volumes `(nz, ny, nx)` are accepted as well; each slice is projected
independently and `detector_sizey` defaults to `nz`:

```python
import numpy as np
volume = np.stack([create_phantom2d(256, Circle(10)), create_shepp_logan_phantom2d(256)])
sino = project_image(volume, n_angles=360, detector_sizex=256)   # (2, 360, 256)
```

### 2. Phantom Generation

**Shapes:** immutable phantom objects in pixels of the `size` grid, placed relative to the
image centre `(size-1)/2` (the rotation axis) or at absolute `x_center`,
`y_center`; overlapping shapes add up their `value`:
```python
from pykct import (Circle, HollowCircle, Ellipse, Rectangle, Triangle, GaussianDot,
                   create_phantom2d)

ball = create_phantom2d(256, Circle(radius=10))                        # centred
ball = create_phantom2d(256, Circle(radius=10, x_offset=20.0, y_offset=-15.0),
                        antialias=True, antialias_factor=4)            # 4x4 samples per pixel
balls = create_phantom2d(256, [Circle(20), Circle(8, x_offset=60, y_offset=-30, value=0.5),
                               Circle(5, x_offset=-70, y_offset=40)])
mixed = create_phantom2d(256, [HollowCircle(100, 92), Ellipse(30, 12, angle_deg=30),
                               Rectangle(40, 10, y_offset=60),
                               Triangle(((0, -10), (10, 8), (-10, 8)), x_offset=-60),
                               GaussianDot(3, x_offset=50, value=2.0)])
fine = create_phantom2d(256, Circle(10), supersampling=True, supersampling_factor=5)  # (1280, 1280)
```

**Shepp-Logan phantom:**
```python
from pykct import create_shepp_logan_phantom2d

sl = create_shepp_logan_phantom2d(size=256)                      # SheppLogan2D (= denpy)
toft = create_shepp_logan_phantom2d(size=256, phantom="Toft2D")
```

`create_legacy_shepp_logan_phantom2d` reproduces the non-standard phantom of
pykct <= 0.1.0 (swapped semi-axes, no rotations, centred at size/2). It is
deprecated, emits a `FutureWarning` and will be removed in a later version;
use it only to reproduce earlier experiments.

### 3. Direct Sinogram Generation

Draw shapes and project them with KCT in one call. With supersampling the
image is k times finer, the projection matrices use detector pixel size k
voxels and the line integrals are divided by k, so geometry and values stay
in pixels of the `size` grid:

```python
from pykct import Circle, create_sinogram2d, shepp_logan_shapes2d

ball_sino = create_sinogram2d(256, Circle(10), n_angles=512, detector_size=256,
                              cor_offset=5.0)                                  # (512, 256)
sl_sino = create_sinogram2d(256, shepp_logan_shapes2d(256), n_angles=512,
                            supersampling=True, supersampling_factor=5)
```

### 4. Advanced Usage (Custom Matrices)

For full control over projection matrices, pass them (array or DEN file, 1x4
or 2x4) to the tools of `pykct.PB2D`:

```python
from pykct import PB2D
from denpy import DEN

matrices = PB2D.circular_trajectory_projection_matrices(512, 256, angle_range=360.0, cor_offset=0.0)
volume = DEN.getNumpyArray("volume.den")
sinogram = PB2D.pb2d_projector().project(volume, matrices, detector_sizex=256,
                                         output_file="projection.den")
fbp = PB2D.pb2d_backprojector().fbp(sinogram, matrices, volume_sizex=256)
cgls = PB2D.pb2d_krylov().reconstruct("projection.den", matrices, volume_sizex=256)
```

The matrices of a single view come from `pykct.PM`, e.g. with a skewed
rotation axis for kct-pb2d-* (1x4 matrices, the third coordinate is the
detector row):

```python
import numpy as np
from pykct import PM

angles = np.deg2rad(np.linspace(0, 360, 512, endpoint=False))
P = np.array([PM.projection_matrix1D(t, 256, 1.0, rotation_center_offset_pix=2.0,
                                       rotation_center_slope=0.01) for t in angles])  # (512, 1, 4)
```

See the README for all functions of `pykct.PM` and the script
`createCameraMatricesForCircularScanTrajectoryPB`.

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

This is `rotation_center_offset_pix` of `pykct.PM` and
`--rotation-center-offset-pix` of `createProjectionMatrices.py` and
`createCameraMatricesForCircularScanTrajectoryPB`. The
`--detector-center-offsetvx` of
`createCameraMatricesForCircularScanTrajectoryParallelRay3D.py` shifts the
detector centre instead, so it equals `-cor_offset * pixel_size`. Estimators reporting the
axis position in pixel-index units with `(N-1)/2` as detector centre
(e.g. algotom `find_center_vo`) return `(N-1)/2 + cor_offset`.

Matrices are created by `pykct.PM.projection_matrix2D` in the
`--material-ct-convention` of the geometry script; the pb2d projector does not
use the detector y row.

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
│   ├── PM.py                 # Projection matrices of the parallel ray geometry
│   ├── PB2D.py               # kct-pb2d-* tools: projector, backprojector (FBP), krylov
│   ├── core/
│   │   └── __init__.py       # Projector wrapper and functions
│   ├── phantoms/
│   │   └── __init__.py       # Phantom generation
│   ├── scripts/              # Thin CLI wrappers over the API above
│   │   ├── _common.py
│   │   ├── create_ball_phantom.py         # Circle via create_phantom2d
│   │   ├── create_shepp_logan.py
│   │   ├── create_ball_sinogram.py
│   │   ├── create_shepp_logan_sinogram.py
│   │   ├── createCameraMatricesForCircularScanTrajectoryPB.py  # port of the KCT geometry script
│   │   └── den_centroid.py
│   └── tests/
│       ├── test_PB2D.py
│       ├── test_PM.py
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
from pykct import Circle, create_sinogram2d

# Sinograms with known CoR offsets
for cor in [-10.0, -5.0, 0.0, 5.0, 10.0]:
    sino = create_sinogram2d(128, Circle(radius=10, x_offset=20), n_angles=512,
                             detector_size=256, cor_offset=cor,
                             antialias=True, antialias_factor=4)
    # Save and test your CoR detection algorithm
```

### 2. Interpolation Validation

```python
import numpy as np
from scipy import ndimage
from pykct import Circle, create_sinogram2d

ball = Circle(radius=10, x_offset=20)
sino_zero = create_sinogram2d(128, ball, n_angles=360, detector_size=256, cor_offset=0.0)

# Shift manually
shifted = ndimage.shift(sino_zero, (0, 10.2), order=3)

# Project with actual CoR offset 10.2
sino_offset = create_sinogram2d(128, ball, n_angles=360, detector_size=256, cor_offset=10.2)

diff = shifted - sino_offset
print(f"L2 error: {np.sqrt(np.sum(diff**2)):.2f}")
```

### 3. Batch Sinogram Generation

```python
from pykct import Circle, create_sinogram2d

configs = [
    {"radius": 5, "cor_offset": 0.0},
    {"radius": 5, "cor_offset": 5.0},
    {"radius": 10, "cor_offset": 0.0},
    # ... more configs
]
for i, cfg in enumerate(configs):
    create_sinogram2d(256, Circle(radius=cfg["radius"]), n_angles=512, detector_size=256,
                      cor_offset=cfg["cor_offset"], output_file=f"sino_{i}.den")
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
