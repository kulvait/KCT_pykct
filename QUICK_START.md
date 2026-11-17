# PyKCT Quick Start Guide

## Overview

PyKCT is a clean Python package for KCT C++ parallel beam 2D projector integration. It provides easy-to-use functions for:
- Phantom generation (ball, Shepp-Logan via denpy)
- Sinogram projection with automatic matrix creation
- CoR offset support
- Centroid calculation

## Installation

```bash
cd /workspace/pykct
pip install -e .
```

## Quick Examples

### 1. Simple Projection (Like radon)

```python
from pykct import project_image
import numpy as np

# Create image
image = np.zeros((128, 128))
image[50:80, 50:80] = 1.0

# Project
sinogram = project_image(
    image,
    n_angles=360,
    detector_sizex=256,
    cor_offset=5.0  # CoR offset
)
# Shape: (1, 360, 256) = (slices, angles, detector)
```

### 2. Shepp-Logan Phantom (Using denpy)

```python
from pykct import create_shepp_logan_phantom, project_image

# Shepp-Logan phantom from denpy.PHANTOM specification;
# phantom= "SheppLogan2D" (default) | "Octave2D" | "Toft2D" | "legacy"
phantom = create_shepp_logan_phantom(size=256)

# Project
sinogram = project_image(
    phantom,
    n_angles=512,
    detector_sizex=256,
    cor_offset=0.0
)
```

### 3. Ball Phantom

```python
from pykct import create_circle
from pykct import project_image

# Create ball phantom with antialiasing
image = create_circle(
    size=256,
    radius=10,
    x_center=None,  # Center (127.5, 127.5)
    y_center=None,
    antialias=4
)

# Project
sinogram = project_image(
    image,
    n_angles=512,
    detector_sizex=256,
    cor_offset=5.0
)
```

## Command Line Tools

**All scripts require a positional output file parameter.**

### Create Ball Phantom

```bash
# Positional output file (required)
pykct-create-ball-phantom ball.den --radius 10 --antialias
```

### Create Shepp-Logan Phantom

```bash
# Positional output file (required)
pykct-create-shepp-logan shepp_logan.den
```

### Create Ball Sinogram

```bash
# Rotation axis imaged at (N-1)/2 + 5
pykct-create-ball-sinogram ball_sino.den \
    --radius 10 --n-angles 512 --detector-size 256 \
    --cor-offset 5.0

# Same geometry as detector centre shift (--detector-center-offsetvx d == --cor-offset -d)
pykct-create-ball-sinogram ball_sino.den \
    --radius 10 --n-angles 512 --detector-size 256 \
    --detector-center-offsetvx -5.0
```

### Create Shepp-Logan Sinogram

```bash
# With CoR offset
pykct-create-shepp-logan-sinogram sl_sino.den \
    --n-angles 512 --detector-size 256 \
    --cor-offset 5.0
```

### Calculate Centroid

```bash
pykct-den-centroid sinogram.den
```

### OpenCL Platform

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

## Verification Test

```bash
# Test CoR shift relationship
cd /workspace/pykct

# Create sinograms with different CoR offsets
pykct-create-ball-sinogram data/ball0.den \
    --radius 5 --n-angles 36 --detector-size 64 \
    --cor-offset 0.0

pykct-create-ball-sinogram data/ball5.den \
    --radius 5 --n-angles 36 --detector-size 64 \
    --cor-offset 5.0

# Calculate centroids
pykct-den-centroid data/ball0.den
pykct-den-centroid data/ball5.den

# Expected:
# - ball0: centroid ≈ 31.5 (detector center for 64-pixel detector)
# - ball5: centroid ≈ 36.5 (shifted by +5 due to CoR offset)
# Relationship: centroid(cor+Δ) = centroid(cor) + Δ
```

## Key Points

1. **Geometric center**: `(N-1)/2` for N pixels
   - For 256 pixels: center = 127.5
   
2. **CoR offset** (`--cor-offset`): rotation axis imaged at `(N-1)/2 + cor_offset`
   - sinogram(cor+5) = shift(sinogram(cor), +5)
   - centroid(cor+5) = centroid(cor) + 5

3. **Shepp-Logan phantom**: Uses denpy.PHANTOM.phantom2d()
   - Specification lives in denpy, not re-implemented

4. **Ball phantom**: Custom implementation with antialiasing
   - Supersampling for smooth edges

5. **Output shape**: `(n_slices, n_angles, n_det)`
   - 2D input → `(1, n_angles, n_det)`

6. **OpenCL platform**: C++ runtime auto-detects by default
   - Specify `--platform` only to force a specific device; never a dummy `0:0`

## Complete Pipeline Example

```python
from pykct import create_shepp_logan_phantom, project_image, calculate_centroid

# Test CoR shift relationship
phantom = create_shepp_logan_phantom(size=256)
results = []
for cor in [0.0, 5.0, -10.0]:
    
    # Project (C++ auto-detects platform)
    sino = project_image(
        phantom,
        n_angles=512,
        detector_sizex=256,
        cor_offset=cor
    )
    
    # Mass-weighted detector centroid = image of the rotation axis
    centroid, _ = calculate_centroid(sino)
    results.append((cor, centroid))
    
for cor, centroid in results:
    print(f"CoR={cor:+.1f}: centroid = {centroid:.2f}")

# Expected output:
# CoR=+0.0: centroid = 127.50
# CoR=+5.0: centroid = 132.50
# CoR=-10.0: centroid = 117.50
```

## Troubleshooting

### Missing output file argument
All scripts require a positional output file:
```bash
# Wrong: pykct-create-ball-phantom --radius 10
# Correct:
pykct-create-ball-phantom ball.den --radius 10
```

### "np is not defined"
Fixed! All scripts now properly import numpy.

### "pykct.phantoms module not found"
The pykct/phantoms module was removed. Use:
- `denpy.PHANTOM.phantom2d()` for Shepp-Logan
- `create_circle()` from `pykct.scripts.create_ball_phantom` for ball

### Directory doesn't exist
Create the directory first:
```bash
mkdir -p data
pykct-create-ball-phantom data/ball.den --radius 10
```

### File already exists
Add `--force` to overwrite:
```bash
pykct-create-ball-phantom data/ball.den --radius 10 --force
```

## Files Generated

Each command creates DEN files:
- `*.den` - Phantom
- `*_sino*.den` - Sinogram

Use `denpy` or `pykct.scripts.den_centroid` to read them.
