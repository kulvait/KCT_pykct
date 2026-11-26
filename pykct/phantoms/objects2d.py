#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2D phantom objects that draw themselves on a pixel grid.

Class hierarchy
---------------
* :class:`PhantomObject` -- anything that can be drawn on a ``size`` x
  ``size`` grid: position (``x_center``, ``y_center``, ``x_offset``,
  ``y_offset``) and ``value``; :meth:`~PhantomObject.draw` returns an image,
  :meth:`~PhantomObject.add_to` adds the object to an existing canvas.

  * :class:`PhantomShape` -- a shape given by a formula: each subclass
    implements :meth:`~PhantomShape.profile` (its fraction of ``value`` at a
    point in the shape's own coordinates) and :meth:`~PhantomShape.extent`;
    the base class does the sampling (positioning, rotation, antialiasing,
    supersampling, bounding box).

    :class:`Circle`, :class:`HollowCircle`, :class:`Ellipse`,
    :class:`Rectangle`, :class:`Triangle`, :class:`GaussianDot`

  * :class:`PhantomImageObject` -- an object that comes as an image: each
    subclass implements :meth:`~PhantomImageObject.render` (the image at a
    requested number of pixels); the base class places it on the canvas.

    :class:`DenpyPhantom2D` (denpy.PHANTOM.phantom2d), :class:`ArrayImage`

Coordinates
-----------
Lengths and positions are in pixels of the ``size`` grid. Pixel (row i,
column j) has its centre at x = j, y = i; the image centre (size-1)/2 is the
rotation axis of the projector. x runs along columns, y along rows (row
index increasing, i.e. downwards with ``imshow(origin="upper")``). A
positive ``angle_deg`` turns +x towards +y.

Discretization (arguments of draw / add_to)
-------------------------------------------
* ``supersampling_factor`` k: the canvas has size*k pixels per side and the
  object is scaled by k, so it covers the same region for every k;
* ``antialias_factor`` a: every canvas pixel is the mean of a x a samples on
  a regular sub-grid; a = 1 samples the pixel centre.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "PhantomObject", "PhantomShape", "PhantomImageObject",
    "Circle", "HollowCircle", "Ellipse", "Rectangle", "Triangle", "GaussianDot",
    "DenpyPhantom2D", "ArrayImage",
]


def _check_factor(factor, name):
    if isinstance(factor, bool) or float(factor) != int(factor) or int(factor) < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {factor!r}")
    return int(factor)


# ---------------------------------------------------------------------------
# base classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class PhantomObject:
    """Base class: position and value of an object on a size x size grid.

    Attributes
    ----------
    x_center, y_center : float or None
        Absolute position in pixels; None means the image centre (size-1)/2.
    x_offset, y_offset : float
        Added to the position.
    value : float
        Added to the pixels covered by the object (overlapping objects add
        up, a negative value subtracts).
    """
    x_center: Optional[float] = None
    y_center: Optional[float] = None
    x_offset: float = 0.0
    y_offset: float = 0.0
    value: float = 1.0

    def offset_from_image_centre(self, size):
        """Object centre relative to the image centre (size-1)/2, in pixels.

        Computed relative to the centre (not as absolute position minus the
        centre) so that a centred object is exactly symmetric on the grid.
        """
        centre = 0.5 * (int(size) - 1)
        dx = 0.0 if self.x_center is None else float(self.x_center) - centre
        dy = 0.0 if self.y_center is None else float(self.y_center) - centre
        return dx + float(self.x_offset), dy + float(self.y_offset)

    def draw(self, size, antialias_factor=1, supersampling_factor=1, dtype=np.float32):
        """The object alone on a (size*k, size*k) canvas, k = supersampling_factor."""
        k = _check_factor(supersampling_factor, "supersampling_factor")
        canvas = np.zeros((int(size) * k, int(size) * k), dtype=np.float64)
        self.add_to(canvas, size, antialias_factor, supersampling_factor)
        return canvas.astype(dtype, copy=False)

    def add_to(self, canvas, size, antialias_factor=1, supersampling_factor=1):
        """Add the object to ``canvas`` of shape (size*k, size*k) in place."""
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class PhantomShape(PhantomObject):
    """A shape given by a formula; subclasses implement profile and extent."""

    def profile(self, x, y):
        """Fraction of ``value`` (0..1) at points (x, y) in the shape's own
        coordinates: pixels, origin at the shape centre, before rotation."""
        raise NotImplementedError

    def extent(self):
        """Radius around the shape centre outside of which profile is 0."""
        raise NotImplementedError

    def rotation_deg(self):
        """Rotation of the shape about its centre; 0 unless overridden."""
        return 0.0

    def add_to(self, canvas, size, antialias_factor=1, supersampling_factor=1):
        a = _check_factor(antialias_factor, "antialias_factor")
        k = _check_factor(supersampling_factor, "supersampling_factor")
        n = int(size) * k
        if canvas.shape != (n, n):
            raise ValueError(f"canvas must have shape {(n, n)}, got {canvas.shape}")
        cx, cy = self.offset_from_image_centre(size)
        # canvas pixel index -> coordinate relative to the image centre, in
        # grid pixels; regular a x a sub-samples inside each canvas pixel
        coord = (np.arange(n, dtype=np.float64) - 0.5 * (n - 1)) / k
        sub = ((np.arange(a) + 0.5) / a - 0.5) / k
        extent = self.extent()

        def index_range(c):     # bounding box, one pixel margin, clipped
            lo = int(np.floor((c - extent) * k + 0.5 * (n - 1))) - 1
            hi = int(np.ceil((c + extent) * k + 0.5 * (n - 1))) + 2
            return max(lo, 0), min(hi, n)
        c0, c1 = index_range(cx)
        r0, r1 = index_range(cy)
        if c0 >= c1 or r0 >= r1:
            return canvas
        t = np.radians(self.rotation_deg())
        ct, st = np.cos(t), np.sin(t)
        block = np.zeros((r1 - r0, c1 - c0), dtype=np.float64)
        for oy in sub:
            dy = (coord[r0:r1] + oy - cy)[:, None]
            for ox in sub:
                dx = (coord[c0:c1] + ox - cx)[None, :]
                # rotate back into the shape's own coordinates
                block += self.profile(ct * dx + st * dy, -st * dx + ct * dy)
        canvas[r0:r1, c0:c1] += float(self.value) * block / (a * a)
        return canvas


@dataclass(frozen=True, kw_only=True)
class PhantomImageObject(PhantomObject):
    """An object that comes as an image covering ``size_px`` x ``size_px``
    grid pixels around its centre; subclasses implement render.

    On a canvas with supersampling k and antialiasing a, the object is
    rendered at size_px*k*a pixels and block-averaged by a. Its pixels must
    line up with the canvas pixels: the object's top-left corner must fall
    on a canvas pixel boundary (integer shifts of 1/k grid pixels, and
    size_px of the same parity as size when centred), else ValueError.
    """
    size_px: int = 0

    def __post_init__(self):
        if int(self.size_px) < 1 or int(self.size_px) != self.size_px:
            raise ValueError(f"size_px must be a positive integer: {self!r}")

    def render(self, n):
        """The object image at n x n pixels (n = size_px * k * a)."""
        raise NotImplementedError

    def add_to(self, canvas, size, antialias_factor=1, supersampling_factor=1):
        a = _check_factor(antialias_factor, "antialias_factor")
        k = _check_factor(supersampling_factor, "supersampling_factor")
        n = int(size) * k
        if canvas.shape != (n, n):
            raise ValueError(f"canvas must have shape {(n, n)}, got {canvas.shape}")
        m = int(self.size_px) * k
        image = np.asarray(self.render(m * a), dtype=np.float64)
        if image.shape != (m * a, m * a):
            raise ValueError(f"render({m * a}) returned shape {image.shape}")
        if a > 1:
            image = image.reshape(m, a, m, a).mean(axis=(1, 3))
        cx, cy = self.offset_from_image_centre(size)
        # canvas index of the object's first pixel
        col0 = 0.5 * (n - 1) + cx * k - 0.5 * (m - 1)
        row0 = 0.5 * (n - 1) + cy * k - 0.5 * (m - 1)
        if abs(col0 - round(col0)) > 1e-9 or abs(row0 - round(row0)) > 1e-9:
            raise ValueError(f"{type(self).__name__} does not line up with the canvas pixels "
                             f"(first pixel at column {col0}, row {row0}); use offsets that "
                             f"are multiples of 1/supersampling_factor and a size_px of the "
                             f"same parity as size")
        col0, row0 = int(round(col0)), int(round(row0))
        rs, re_ = max(row0, 0), min(row0 + m, n)
        cs, ce = max(col0, 0), min(col0 + m, n)
        if rs < re_ and cs < ce:
            canvas[rs:re_, cs:ce] += float(self.value) * image[rs - row0:re_ - row0,
                                                               cs - col0:ce - col0]
        return canvas


# ---------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Circle(PhantomShape):
    """Disc: x^2 + y^2 <= radius^2."""
    radius: float

    def __post_init__(self):
        if not self.radius > 0:
            raise ValueError(f"Circle radius must be positive: {self!r}")

    def extent(self):
        return float(self.radius)

    def profile(self, x, y):
        r = float(self.radius)
        return (x * x + y * y <= r * r).astype(np.float64)


@dataclass(frozen=True)
class HollowCircle(PhantomShape):
    """Annulus: inner_radius^2 < x^2 + y^2 <= radius^2.

    The inner edge is excluded, so HollowCircle(r, ri) + Circle(ri) equals
    Circle(r) pixel for pixel.
    """
    radius: float
    inner_radius: float

    def __post_init__(self):
        if not (self.radius > 0 and 0 <= self.inner_radius < self.radius):
            raise ValueError(f"HollowCircle needs 0 <= inner_radius < radius: {self!r}")

    def extent(self):
        return float(self.radius)

    def profile(self, x, y):
        r, ri = float(self.radius), float(self.inner_radius)
        d2 = x * x + y * y
        return ((d2 <= r * r) & (d2 > ri * ri)).astype(np.float64)


@dataclass(frozen=True)
class Ellipse(PhantomShape):
    """Ellipse: (x / radius_x)^2 + (y / radius_y)^2 <= 1, rotated by angle_deg."""
    radius_x: float
    radius_y: float
    angle_deg: float = 0.0

    def __post_init__(self):
        if not (self.radius_x > 0 and self.radius_y > 0):
            raise ValueError(f"Ellipse radii must be positive: {self!r}")

    def rotation_deg(self):
        return float(self.angle_deg)

    def extent(self):
        return float(max(self.radius_x, self.radius_y))

    def profile(self, x, y):
        return ((x / self.radius_x) ** 2 + (y / self.radius_y) ** 2 <= 1.0).astype(np.float64)


@dataclass(frozen=True)
class Rectangle(PhantomShape):
    """Rectangle: |x| <= width/2 and |y| <= height/2, rotated by angle_deg."""
    width: float
    height: float
    angle_deg: float = 0.0

    def __post_init__(self):
        if not (self.width > 0 and self.height > 0):
            raise ValueError(f"Rectangle width and height must be positive: {self!r}")

    def rotation_deg(self):
        return float(self.angle_deg)

    def extent(self):
        return float(0.5 * np.hypot(self.width, self.height))

    def profile(self, x, y):
        return ((np.abs(x) <= 0.5 * self.width)
                & (np.abs(y) <= 0.5 * self.height)).astype(np.float64)


@dataclass(frozen=True)
class Triangle(PhantomShape):
    """Triangle with three vertices (x, y) relative to the shape centre,
    rotated by angle_deg; the vertex order does not matter."""
    vertices: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    angle_deg: float = 0.0
    _ccw: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        v = np.asarray(self.vertices, dtype=np.float64)
        if v.shape != (3, 2):
            raise ValueError(f"Triangle needs three (x, y) vertices: {self!r}")
        area2 = ((v[1, 0] - v[0, 0]) * (v[2, 1] - v[0, 1])
                 - (v[1, 1] - v[0, 1]) * (v[2, 0] - v[0, 0]))
        if area2 == 0:
            raise ValueError(f"Triangle vertices are collinear: {self!r}")
        object.__setattr__(self, "vertices", tuple(tuple(float(c) for c in p) for p in v))
        # counter-clockwise order in (x, y) for the edge test
        object.__setattr__(self, "_ccw", v if area2 > 0 else v[[0, 2, 1]])

    def rotation_deg(self):
        return float(self.angle_deg)

    def extent(self):
        return float(np.sqrt((self._ccw ** 2).sum(axis=1)).max())

    def profile(self, x, y):
        v = self._ccw
        inside = np.ones(np.broadcast(x, y).shape, dtype=bool)
        for p, q in ((v[0], v[1]), (v[1], v[2]), (v[2], v[0])):
            # (q - p) x (point - p) >= 0: point on the left of or on the edge
            inside &= (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0]) >= 0
        return inside.astype(np.float64)


@dataclass(frozen=True)
class GaussianDot(PhantomShape):
    """Isotropic Gaussian exp(-(x^2 + y^2) / (2 sigma^2)); value is the peak.

    Evaluated within EXTENT_SIGMA * sigma of the centre (relative value
    there exp(-32) ~ 1e-14).
    """
    sigma: float
    EXTENT_SIGMA = 8.0

    def __post_init__(self):
        if not self.sigma > 0:
            raise ValueError(f"GaussianDot sigma must be positive: {self!r}")

    def extent(self):
        return self.EXTENT_SIGMA * float(self.sigma)

    def profile(self, x, y):
        s = float(self.sigma)
        return np.exp(-(x * x + y * y) / (2.0 * s * s))


# ---------------------------------------------------------------------------
# image objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DenpyPhantom2D(PhantomImageObject):
    """A denpy.PHANTOM.phantom2d phantom covering size_px x size_px pixels.

    Rendered by denpy at every resolution (n = size_px * k * a samples of
    [-1, 1]), i.e. exactly what denpy produces, including its sampling of
    [-1, 1] at n points: the phantom's half-width is (n-1)/(2 k a) grid
    pixels, so it grows slightly with k and a (by less than half a pixel).
    For a phantom that scales exactly use :func:`shepp_logan_shapes2d`.
    """
    phantom: str = "SheppLogan2D"

    def __post_init__(self):
        super().__post_init__()
        from denpy import PHANTOM
        if not hasattr(PHANTOM, self.phantom):
            raise ValueError(f"denpy.PHANTOM has no phantom '{self.phantom}'")

    def render(self, n):
        from denpy import PHANTOM
        return PHANTOM.phantom2d(phantom=self.phantom, n=int(n))


@dataclass(frozen=True, eq=False)
class ArrayImage(PhantomImageObject):
    """A given 2D array, one array pixel per grid pixel (size_px = array side).

    With supersampling k every array pixel becomes k x k canvas pixels
    (piecewise constant); antialiasing does not change it.
    """
    array: np.ndarray = None
    size_px: int = field(init=False)

    def __post_init__(self):
        arr = np.asarray(self.array, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 1:
            raise ValueError(f"ArrayImage needs a square 2D array, got shape {arr.shape}")
        object.__setattr__(self, "array", arr)
        object.__setattr__(self, "size_px", arr.shape[0])

    def render(self, n):
        m = self.array.shape[0]
        if n % m:
            raise ValueError(f"ArrayImage of side {m} cannot be rendered at {n} pixels")
        r = n // m
        return np.repeat(np.repeat(self.array, r, axis=0), r, axis=1)
