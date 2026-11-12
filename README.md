# PyKCT - Python bindings for KCT C++ Projector

Python package for parallel beam 2D CT projection using the KCT C++ projector.

## Installation

### Development installation

```bash
cd pykct
pip install -e .
```

### Install dependencies

```bash
pip install numpy scipy denpy matplotlib
```

## Quick Start

### Python API

```python
import numpy as np
from pykct import project_image

# Create simple phantom
image = np.zeros((128, 128))
image[50:80, 50:80] = 1.0

# Project with default parameters
sinogram = project_image(image, n_angles=360, detector_sizex=256)
# Shape: (1, 360, 256) = (slices, angles, detector)

# Project with CoR offset
sinogram_offset = project_image(
    image, n_angles=360, detector_sizex=256, cor_offset=5.0
)
```

### Using Shepp-Logan phantom from denpy

```python
from denpy import PHANTOM
from pykct import project_image

# Create Shepp-Logan phantom using denpy's specification
phantom = PHANTOM.phantom2d(n=256)

# Project
sinogram = project_image(
    phantom, n_angles=512, detector_sizex=256, cor_offset=5.0
)
```

### Creating ball phantom

```python
from pykct.scripts.create_ball_phantom import create_circle
import numpy as np
from denpy import DEN

# Create ball phantom
image = create_circle(size=256, radius=10, antialias=4)
volume = image[np.newaxis, :, :].astype(np.float32)
DEN.storeNdarrayAsDEN('ball.den', volume, force=True)
```

## Command Line Tools

All scripts require a **positional output file parameter** (no `-o`/`--output`).

### Create ball phantom

```bash
pykct-create-ball-phantom ball.den --radius 10 --antialias
```

### Create Shepp-Logan phantom

```bash
pykct-create-shepp-logan shepp_logan.den
```

### Create ball sinogram

```bash
pykct-create-ball-sinogram ball_sino.den --radius 10 --n-angles 360 \
    --detector-size 256 --detector-center-offsetvx 5.0

# --cor-offset is an alias for --detector-center-offsetvx
pykct-create-ball-sinogram ball_sino.den --radius 10 --n-angles 360 \
    --detector-size 256 --cor-offset 5.0
```

### Create Shepp-Logan sinogram

```bash
pykct-create-shepp-logan-sinogram sl_sino.den --n-angles 512 \
    --detector-size 256 --detector-center-offsetvx 5.0
```

### Calculate centroid

```bash
pykct-den-centroid sinogram.den
```

- For 2D images: shows 2D centroid

### OpenCL platform

By default, the C++ runtime auto-detects the OpenCL platform. To specify a platform:

```bash
pykct-create-ball-sinogram ball_sino.den --radius 10 --platform 0:0 \
    --n-angles 360 --detector-size 256 --detector-center-offsetvx 5.0
```

## API Reference

### Core Functions

#### `project_image(image, n_angles=512, start_angle=0.0, angle_range=360.0, detector_sizex=256, detector_sizey=1, cor_offset=0.0, pixel_size=1.0, platform_id=None, verbose=False)`

High-level convenience function for projecting images.

**Parameters:**
- `image` (np.ndarray): 2D image of shape (ny, nx)
- `n_angles` (int): Number of projection angles (default: 512)
- `start_angle` (float): Starting angle in degrees (default: 0.0)
- `angle_range` (float): Angular range in degrees (default: 360.0)
- `detector_sizex` (int): Detector X size (default: 256)
- `detector_sizey` (int): Detector Y size (default: 1)
- `cor_offset` (float): CoR offset in pixels (default: 0.0)
- `pixel_size` (float): Pixel size (default: 1.0)
- `platform_id` (str, optional): OpenCL platform ID. If None, C++ runtime auto-detects.
- `verbose` (bool): Print progress information

**Returns:** Sinogram of shape (n_slices, n_angles, detector_sizex)
- For 2D input: (1, n_angles, detector_sizex)

### Phantom Functions

#### `create_circle(size, radius, x_center=None, y_center=None, antialias=1)`

Create a 2D circular phantom with supersampling antialiasing.

**Parameters:**
- `size` (int): Image size
- `radius` (float): Ball radius
- `x_center` (float, optional): X center (default: center of image)
- `y_center` (float, optional): Y center (default: center of image)
- `antialias` (int): Antialiasing factor (default: 1 = no antialiasing)

**Returns:** 2D image of shape (size, size)

## Geometry Reference

### Coordinate System

- **Geometric center**: `(N-1)/2` for N pixels (0-indexed convention)
- **Example**: For 256 pixels, center = 127.5

### CoR Offset Convention

- **Positive**: CoR to the RIGHT of detector center
- **Negative**: CoR to the LEFT of detector center
- **Relationship**: `sinogram(cor + Δ) = shift(sinogram(cor), -Δ)`

### Sinogram Shape Convention

- **Input**: 2D image `(ny, nx)`
- **Output**: Sinogram `(n_slices, n_angles, n_det)`
  - For 2D input: `(1, n_angles, n_det)`
  - For 3D volume `(nz, ny, nx)`: Output `(nz, n_angles, n_det)`

**Note**: KCT outputs `(n_angles, n_slices, n_det)`, which is automatically transposed to the standard sinogram convention.

## Testing

Run tests:

```bash
pytest pykct/tests/ -v
```

## Package Structure

```
pykct/
├── pykct/
│   ├── __init__.py           # Main API exports
│   ├── core/__init__.py      # Projector wrapper and functions
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
└── QUICK_START.md
```

## Dependencies

- NumPy >= 1.19.0
- SciPy >= 1.7.0
- DENpy >= 0.5.0 (provides PHANTOM module for Shepp-Logan phantom)

Optional:
- Matplotlib >= 3.4 (for visualization)

## License

GNU GPL3 (same as KCT)

## References

- KCT Projector: https://github.com/Kulvait/KCT
- denpy: https://github.com/Kulvait/denpy
- Parallel beam geometry: See standard CT textbooks
