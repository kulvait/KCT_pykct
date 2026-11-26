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

### Phantoms from shapes

Phantom objects are immutable dataclasses in pixels of the `size` x `size` grid, placed
relative to the image centre `(size-1)/2` (which is also the rotation axis)
or at absolute `x_center`, `y_center`. Overlapping shapes add up their
`value`. `create_phantom2d` draws them on the grid.

```python
from pykct import (Circle, HollowCircle, Ellipse, Rectangle, Triangle, GaussianDot,
                   create_phantom2d)

image = create_phantom2d(256, [
    HollowCircle(radius=100, inner_radius=92),             # a tube wall
    Circle(radius=20, x_offset=40, value=0.5),
    Ellipse(radius_x=30, radius_y=12, angle_deg=30, y_offset=-50),
    Rectangle(width=40, height=10, angle_deg=-15, y_offset=60),
    Triangle(vertices=((0, -10), (10, 8), (-10, 8)), x_offset=-60),
    GaussianDot(sigma=3, x_center=200, y_center=100, value=2.0),
], antialias=True, antialias_factor=4)                     # (256, 256)

# The same phantom on a 5x finer grid, (1280, 1280)
fine = create_phantom2d(256, Circle(radius=20, x_offset=40),
                        supersampling=True, supersampling_factor=5)
```

* `x`, `y` follow the image columns and rows (row index increasing, i.e.
  downwards with `imshow(origin="upper")`); positive `angle_deg` turns +x
  towards +y.
* `antialias`, `antialias_factor`: every pixel is the mean of
  `factor x factor` samples (otherwise the pixel centre is sampled).
* `supersampling`, `supersampling_factor` k: the image has `size*k` pixels
  per side and all shape parameters are scaled by k.
* A factor only takes effect with its switch, a switch needs a factor >= 2;
  anything else raises `ValueError`.

Every object draws itself: `obj.draw(size, antialias_factor=1,
supersampling_factor=1)` returns its image, `obj.add_to(canvas, size, ...)`
adds it to a canvas; `create_phantom2d` just loops over `add_to`.

```python
from pykct import ArrayImage, DenpyPhantom2D, HollowCircle

ring = HollowCircle(radius=30, inner_radius=25).draw(128, antialias_factor=4)

# objects given as an image, placed on whole canvas pixels
head = DenpyPhantom2D(size_px=128, phantom="Toft2D")        # denpy.PHANTOM.phantom2d
img = create_phantom2d(256, [head, Circle(5, x_offset=90)])
tile = ArrayImage(array=my_array, x_offset=-64)             # given 2D array
```

New shapes subclass `PhantomShape` and implement `profile(x, y)` (fraction of
`value` at a point in the shape's own coordinates) and `extent()`; image
objects subclass `PhantomImageObject` and implement `render(n)`.

### Shepp-Logan phantom

```python
from pykct import create_shepp_logan_phantom2d, shepp_logan_shapes2d

phantom = create_shepp_logan_phantom2d(256)                    # == denpy.PHANTOM.phantom2d
toft = create_shepp_logan_phantom2d(256, phantom="Toft2D", supersampling=True,
                                    supersampling_factor=4)    # (1024, 1024)
ellipses = shepp_logan_shapes2d(256)                           # list of Ellipse shapes
```

### Sinograms

`create_sinogram2d` draws the shapes and projects them with KCT. The geometry
is in pixels of the `size` grid for every supersampling factor: a detector
pixel is one grid pixel (the projection matrices use detector pixel size k
voxels) and the line integrals are divided by k.

```python
from pykct import Circle, create_sinogram2d

sino, image = create_sinogram2d(256, [Circle(10, x_offset=30), Circle(5, y_offset=-60)],
                                n_angles=360, detector_size=256, cor_offset=3.5,
                                antialias=True, antialias_factor=4,
                                supersampling=True, supersampling_factor=5,
                                return_phantom=True)
# sino (360, 256), image (1280, 1280)
```

## Command Line Tools

All scripts require a **positional output file parameter** (no `-o`/`--output`).

### Create ball phantom

```bash
pykct-create-ball-phantom ball.den --radius 10 --antialias
# 5x finer grid: (1, 1280, 1280)
pykct-create-ball-phantom ball5.den --radius 10 --supersampling-factor 5
```

### Create Shepp-Logan phantom

```bash
pykct-create-shepp-logan shepp_logan.den
```

### Create ball sinogram

```bash
# Rotation axis imaged at (N-1)/2 + 5 detector pixels
pykct-create-ball-sinogram ball_sino.den --radius 10 --n-angles 360 \
    --detector-size 256 --cor-offset 5.0

# Same geometry given as detector centre shift of the geometry script
# (--detector-center-offsetvx d is equivalent to --cor-offset -d)
pykct-create-ball-sinogram ball_sino.den --radius 10 --n-angles 360 \
    --detector-size 256 --detector-center-offsetvx -5.0
```

### Create Shepp-Logan sinogram

```bash
pykct-create-shepp-logan-sinogram sl_sino.den --n-angles 512 \
    --detector-size 256 --cor-offset 5.0
```

### Calculate centroid

```bash
pykct-den-centroid sinogram.den
```

- For 2D images: shows 2D centroid

### OpenCL platform

By default the C++ runtime selects the OpenCL platform and device by its own
heuristic: it prefers a platform expected to perform well (e.g. the first
NVIDIA platform) and uses its last device to avoid clashes with the card used
for display or by other jobs. Leave `--platform` / `platform_id` unset unless
a specific OpenCL device must do the work; do not pass a dummy `0:0`, which
overrides the heuristic.

```bash
# Only when a specific device is wanted, e.g. platform 1, device 2 (see clinfo -l)
pykct-create-ball-sinogram ball_sino.den --radius 10 --platform 1:2 \
    --n-angles 360 --detector-size 256 --cor-offset 5.0
```

## API Reference

### Core Functions

#### `project_image(image, n_angles=512, start_angle=0.0, angle_range=360.0, detector_sizex=256, detector_sizey=None, cor_offset=0.0, pixel_size=1.0, platform_id=None, verbose=False)`

High-level convenience function for projecting images.

**Parameters:**
- `image` (np.ndarray): 2D image (ny, nx) or volume (nz, ny, nx)
- `n_angles` (int): Number of projection angles (default: 512)
- `start_angle` (float): Starting angle in degrees (default: 0.0)
- `angle_range` (float): Angular range in degrees (default: 360.0)
- `detector_sizex` (int): Detector X size (default: 256)
- `detector_sizey` (int, optional): Detector Y size (default: nz)
- `cor_offset` (float): CoR offset in pixels; rotation axis imaged at `(N-1)/2 + cor_offset` (default: 0.0)
- `pixel_size` (float): Detector pixel size in voxels (default: 1.0); `create_sinogram2d` sets it to the supersampling factor
- `platform_id` (str, optional): OpenCL `platform:device`. Leave None (recommended) so the C++ runtime picks the device, see [OpenCL platform](#opencl-platform).
- `verbose` (bool): Print progress information

**Returns:** Sinogram of shape (n_slices, n_angles, detector_sizex)
- For 2D input: (1, n_angles, detector_sizex)

### Shapes

All shapes have `x_center=None`, `y_center=None` (None: image centre
`(size-1)/2`), `x_offset=0`, `y_offset=0` and `value=1.0`; lengths in pixels.

| shape | parameters |
|---|---|
| `Circle` | `radius` |
| `HollowCircle` | `radius`, `inner_radius` (0 <= inner < radius) |
| `Ellipse` | `radius_x`, `radius_y`, `angle_deg=0` |
| `Rectangle` | `width`, `height`, `angle_deg=0` |
| `Triangle` | `vertices` (three (x, y) relative to the shape centre), `angle_deg=0` |
| `GaussianDot` | `sigma` (`value` is the peak) |
| `DenpyPhantom2D` | `size_px`, `phantom` (name in denpy.PHANTOM) |
| `ArrayImage` | `array` (square, one array pixel per grid pixel) |

### Phantom Functions

#### `create_phantom2d(size, objects, antialias=False, antialias_factor=1, supersampling=False, supersampling_factor=1, dtype=float32)`

Draw one shape or a list of shapes. **Returns:** 2D image `(size*k, size*k)`,
k = supersampling factor.

#### `create_shepp_logan_phantom2d(size=256, phantom="SheppLogan2D", antialias=..., supersampling=...)`

Shepp-Logan variants `SHEPP_LOGAN_VARIANTS` = SheppLogan2D, Octave2D, Toft2D,
drawn from `shepp_logan_shapes2d(size, phantom)` (Ellipse shapes). Without
antialiasing and supersampling equal to `denpy.PHANTOM.phantom2d`.

#### `create_legacy_shepp_logan_phantom2d(size=256)`

Deprecated non-standard phantom of pykct <= 0.1.0 (FutureWarning).

#### `create_sinogram2d(size, objects, n_angles=512, start_angle=0.0, angle_range=360.0, detector_size=None, cor_offset=0.0, antialias=..., supersampling=..., output_file=None, platform_id=None, verbose=False, return_phantom=False)`

Draw and project with KCT. **Returns:** sinogram `(n_angles, detector_size)`
in grid pixel units (and the image with `return_phantom=True`).

## Geometry Reference

### Coordinate System

- **Geometric center**: `(N-1)/2` for N pixels (0-indexed convention)
- **Example**: For 256 pixels, center = 127.5

### CoR Offset Convention

- `cor_offset`: distance of the principal ray (through the rotation axis) from the geometric detector centre
- **Rotation axis image**: detector pixel index `(N-1)/2 + cor_offset`
- Same as `createProjectionMatrices.py --rotation-center-offset-pix`; `--detector-center-offsetvx` of `createCameraMatricesForCircularScanTrajectoryParallelRay3D.py` is `-cor_offset`
- **Relationship**: `sinogram(cor + Δ) = shift(sinogram(cor), +Δ)`

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
