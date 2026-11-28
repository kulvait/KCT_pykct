"""Tests of pykct.PM, properties stated in the docstrings."""

import numpy as np
import pytest

from pykct.PM import (projection_matrix1D, projection_matrix2D, projection_matrix2D_skew,
                        projection_matrix2D_tilt, projection_matrix2D_vectors)

N, M = 101, 7
pixel_sizex, pixel_sizey = 0.02, 0.05
THETAS = np.linspace(0.0, 2.0 * np.pi, 13)
CONVENTIONS = [False, True]
GEOMETRY = dict(detector_centerx=0.4, detector_centery=-0.7, detector_centerz=1.1,
                detector_center_offsetvx=0.3, detector_center_offsetvy=-0.15)


def project(CM, x):
    return CM[:, :3] @ np.asarray(x, dtype=np.float64) + CM[:, 3]


@pytest.mark.parametrize("material", CONVENTIONS)
@pytest.mark.parametrize("theta", THETAS)
def test_rotation_axis_imaged_at_detector_center(theta, material):
    CM = projection_matrix2D(theta, N, M, pixel_sizex, pixel_sizey, material_ct_convention=material)
    assert CM.shape == (2, 4) and CM.dtype == np.float64
    np.testing.assert_allclose(project(CM, [0.0, 0.0, 0.0]), [(N - 1) / 2, (M - 1) / 2], atol=1e-12)


@pytest.mark.parametrize("material", CONVENTIONS)
@pytest.mark.parametrize("theta", THETAS)
def test_rows_scale_and_orthogonality(theta, material):
    a, b = projection_matrix2D(theta, N, M, pixel_sizex, pixel_sizey, material_ct_convention=material)[:, :3]
    np.testing.assert_allclose(np.linalg.norm(a), 1.0 / pixel_sizex)
    np.testing.assert_allclose(b, [0.0, 0.0, (1.0 if material else -1.0) / pixel_sizey])
    assert abs(a.dot(b)) < 1e-12


@pytest.mark.parametrize("material", CONVENTIONS)
@pytest.mark.parametrize("theta", THETAS)
def test_offsets_move_axis_image_independent_of_theta(theta, material):
    offx, offy, rcop = 0.3, -0.15, 2.25
    CM = projection_matrix2D(theta, N, M, pixel_sizex, pixel_sizey, detector_center_offsetvx=offx,
                             detector_center_offsetvy=offy, rotation_center_offset_pix=rcop,
                             material_ct_convention=material)
    np.testing.assert_allclose(project(CM, [0.0, 0.0, 0.0]),
                               [(N - 1) / 2 + rcop - offx / pixel_sizex, (M - 1) / 2 - offy / pixel_sizey],
                               atol=1e-10)


@pytest.mark.parametrize("material", CONVENTIONS)
def test_rotation_center_offset_pix_equals_negative_offsetvx(material):
    for theta in THETAS:
        A = projection_matrix2D(theta, N, M, pixel_sizex, pixel_sizey, rotation_center_offset_pix=3.5,
                                material_ct_convention=material)
        B = projection_matrix2D(theta, N, M, pixel_sizex, pixel_sizey,
                                detector_center_offsetvx=-3.5 * pixel_sizex, material_ct_convention=material)
        np.testing.assert_allclose(A, B, atol=1e-10)


@pytest.mark.parametrize("material", CONVENTIONS)
def test_detector_center_is_imaged_at_center(material):
    C = [0.4, -0.7, 1.1]
    for theta in THETAS:
        CM = projection_matrix2D(theta, N, M, pixel_sizex, pixel_sizey, *C, material_ct_convention=material)
        np.testing.assert_allclose(project(CM, C), [(N - 1) / 2, (M - 1) / 2], atol=1e-10)


def test_medical_convention_mirrors_material_convention():
    for theta in THETAS:
        med = projection_matrix2D(theta, N, M, pixel_sizex, pixel_sizey, material_ct_convention=False)
        mat = projection_matrix2D(-theta, N, M, pixel_sizex, pixel_sizey, material_ct_convention=True)
        np.testing.assert_allclose(med[0], mat[0], atol=1e-12)
        np.testing.assert_allclose(med[1], mat[1] * [1, 1, -1, 1], atol=1e-12)


@pytest.mark.parametrize("material", CONVENTIONS)
@pytest.mark.parametrize("theta", THETAS)
def test_1D_without_skew_is_first_row_of_2D(theta, material):
    kw = dict(detector_centerx=0.4, detector_centery=-0.7, detector_center_offsetvx=0.3,
              rotation_center_offset_pix=0.75, material_ct_convention=material)
    CM1 = projection_matrix1D(theta, N, pixel_sizex, **kw)
    CM2 = projection_matrix2D(theta, N, M, pixel_sizex, pixel_sizey, **kw)
    assert CM1.shape == (1, 4) and CM1.dtype == np.float64
    np.testing.assert_allclose(CM1[0], CM2[0], rtol=0, atol=1e-12)


@pytest.mark.parametrize("material", CONVENTIONS)
@pytest.mark.parametrize("theta", THETAS)
def test_1D_skew_refers_to_row_zero(theta, material):
    offx, rcop, slope = 0.3, 0.75, 0.0125
    CM1 = projection_matrix1D(theta, N, pixel_sizex, detector_center_offsetvx=offx,
                              rotation_center_offset_pix=rcop, rotation_center_slope=slope,
                              material_ct_convention=material)
    for r in [0.0, 3.0, 1919.5]:
        np.testing.assert_allclose(project(CM1, [0.0, 0.0, r])[0],
                                   (N - 1) / 2 + rcop + slope * r - offx / pixel_sizex, atol=1e-10)


@pytest.mark.parametrize("material", CONVENTIONS)
@pytest.mark.parametrize("theta", THETAS)
def test_2D_has_no_skew(theta, material):
    CM = projection_matrix2D(theta, N, M, pixel_sizex, pixel_sizey, rotation_center_offset_pix=1.5,
                             material_ct_convention=material, **GEOMETRY)
    assert CM[0, 2] == 0.0


def test_skew_from_rotation_center_fit():
    """Linear fit offset(r) = A + B * r of a detectRotationCenter*.py log, bin 1."""
    pix = 0.001281815405036707
    fit_a, fit_b = 0.005630139932862342, 1.2281041144214863e-08  # length, length per row
    Nd, Md = 5120, 3840
    CM1 = projection_matrix1D(0.3, Nd, pix, rotation_center_offset_pix=fit_a / pix,
                              rotation_center_slope=fit_b / pix)
    for r in [0.0, (Md - 1) / 2, Md - 1.0]:
        expected = (Nd - 1) / 2 + (fit_a + fit_b * r) / pix
        np.testing.assert_allclose(project(CM1, [0.0, 0.0, r])[0], expected, atol=1e-9)



# General model, skewed and tilted detector

SLOPE = 0.0125
E3 = np.array([0.0, 0.0, 1.0])


def directions(theta, material):
    if material:
        return np.array([np.cos(theta), np.sin(theta), 0.0]), E3
    return np.array([np.cos(theta), -np.sin(theta), 0.0]), -E3


def test_vectors_dual_basis_for_general_detector():
    rng = np.random.default_rng(7)
    for _ in range(20):
        VX, VY, d = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
        for ray in (None, np.cross(VX, VY) + 0.3 * VX - 0.2 * VY):  # orthogonal and tilted detector
            CM = projection_matrix2D_vectors(VX, VY, d, N, M, ray=ray)
            a, b = CM[:, :3]
            r = np.cross(VX, VY) if ray is None else ray
            np.testing.assert_allclose([a @ VX, a @ VY, b @ VX, b @ VY, a @ r, b @ r], [1, 0, 0, 1, 0, 0], atol=1e-10)
            np.testing.assert_allclose(project(CM, d), [(N - 1) / 2, (M - 1) / 2], atol=1e-10)
            np.testing.assert_allclose(project(CM, d + 3 * VX - 2 * VY), [(N - 1) / 2 + 3, (M - 1) / 2 - 2], atol=1e-9)


def test_vectors_rejects_degenerate_detector():
    with pytest.raises(ValueError):
        projection_matrix2D_vectors([1, 0, 0], [2, 0, 0], [0, 0, 0], N, M)


@pytest.mark.parametrize("material", CONVENTIONS)
@pytest.mark.parametrize("theta", THETAS)
def test_vectors_orthogonal_equals_projection_matrix2D(theta, material):
    DX, DY = directions(theta, material)
    rcop = 1.5
    VX, VY = pixel_sizex * DX, pixel_sizey * DY
    C = np.array([GEOMETRY["detector_centerx"], GEOMETRY["detector_centery"], GEOMETRY["detector_centerz"]]) \
        + GEOMETRY["detector_center_offsetvx"] * DX + GEOMETRY["detector_center_offsetvy"] * DY - rcop * VX
    np.testing.assert_allclose(projection_matrix2D_vectors(VX, VY, C, N, M),
                               projection_matrix2D(theta, N, M, pixel_sizex, pixel_sizey, rotation_center_offset_pix=rcop,
                                                   material_ct_convention=material, **GEOMETRY), atol=1e-10)


@pytest.mark.parametrize("model", [projection_matrix2D_skew, projection_matrix2D_tilt])
@pytest.mark.parametrize("material", CONVENTIONS)
@pytest.mark.parametrize("theta", THETAS)
def test_zero_slope_equals_projection_matrix2D(theta, material, model):
    kw = dict(GEOMETRY, rotation_center_offset_pix=1.5, material_ct_convention=material)
    np.testing.assert_allclose(model(theta, N, M, pixel_sizex, pixel_sizey, rotation_center_slope=0.0, **kw),
                               projection_matrix2D(theta, N, M, pixel_sizex, pixel_sizey, **kw), atol=1e-10)


@pytest.mark.parametrize("model", [projection_matrix2D_skew, projection_matrix2D_tilt])
@pytest.mark.parametrize("material", CONVENTIONS)
@pytest.mark.parametrize("theta", THETAS)
def test_axis_image_moves_by_slope_per_row(theta, material, model):
    rcop = 1.5
    CM = model(theta, N, M, pixel_sizex, pixel_sizey, rotation_center_offset_pix=rcop,
               rotation_center_slope=SLOPE, material_ct_convention=material)
    for z in [-0.2, 0.0, 0.13]:
        PX, PY = project(CM, [0.0, 0.0, z])
        np.testing.assert_allclose(PX, (N - 1) / 2 + rcop + SLOPE * (PY - CM[1, 3]), atol=1e-10)
    np.testing.assert_allclose(project(CM, [0.0, 0.0, 0.0]), [(N - 1) / 2 + rcop, (M - 1) / 2], atol=1e-10)


@pytest.mark.parametrize("material", CONVENTIONS)
@pytest.mark.parametrize("theta", THETAS)
def test_skew_equals_createProjectionMatrices_fitted_matrix(theta, material):
    """Old encoding of createProjectionMatrices.py: a[2] = slope * b[2], px0 = (N-1)/2 - a.C + offset."""
    rcop = 1.5
    kw = dict(GEOMETRY, rotation_center_offset_pix=rcop, material_ct_convention=material)
    CM = projection_matrix2D_skew(theta, N, M, pixel_sizex, pixel_sizey, rotation_center_slope=SLOPE, **kw)
    DX, DY = directions(theta, material)
    a, b = DX / pixel_sizex, DY / pixel_sizey
    a[2] = SLOPE * b[2]
    C = np.array([GEOMETRY["detector_centerx"], GEOMETRY["detector_centery"], GEOMETRY["detector_centerz"]]) \
        + GEOMETRY["detector_center_offsetvx"] * DX + GEOMETRY["detector_center_offsetvy"] * DY
    expected = np.array([np.append(a, (N - 1) / 2 - a @ C + rcop), np.append(b, (M - 1) / 2 - b @ C)])
    np.testing.assert_allclose(CM, expected, atol=1e-10)


@pytest.mark.parametrize("material", CONVENTIONS)
@pytest.mark.parametrize("theta", THETAS)
def test_skew_rows_image_planes_and_match_1D(theta, material):
    kw = dict(detector_centerx=0.4, detector_centery=-0.7, detector_center_offsetvx=0.3, material_ct_convention=material)
    rcop = 0.75
    CM2 = projection_matrix2D_skew(theta, N, M, pixel_sizex, pixel_sizey, rotation_center_offset_pix=rcop,
                                   rotation_center_slope=SLOPE, **kw)
    np.testing.assert_allclose(CM2[1, :2], [0.0, 0.0], atol=1e-12)  # row index depends only on z
    CM1 = projection_matrix1D(theta, N, pixel_sizex, rotation_center_offset_pix=rcop - SLOPE * CM2[1, 3],
                              rotation_center_slope=SLOPE, **kw)
    for x in ([0.0, 0.0, 0.0], [0.3, -0.25, 0.17], [-0.4, 0.1, -0.33]):
        PX, PY = project(CM2, x)
        np.testing.assert_allclose(project(CM1, [x[0], x[1], PY])[0], PX, atol=1e-10)


@pytest.mark.parametrize("material", CONVENTIONS)
@pytest.mark.parametrize("theta", [0.3, 1.1, 2.0, 4.0])  # generic angles, cos and sin nonzero
def test_tilt_keeps_orthogonal_axes_and_mixes_coordinates(theta, material):
    CM = projection_matrix2D_tilt(theta, N, M, pixel_sizex, pixel_sizey, rotation_center_slope=SLOPE,
                                  material_ct_convention=material)
    a, b = CM[:, :3]
    assert abs(a @ b) < 1e-10
    np.testing.assert_allclose([np.linalg.norm(a), np.linalg.norm(b)], [1 / pixel_sizex, 1 / pixel_sizey])
    assert np.all(np.abs(a) > 1e-6) and np.all(np.abs(b) > 1e-6)  # rows are not slices


def test_skew_from_rotation_center_fit_2D():
    """2D skew matrix reproduces the fit offset(r) = A + B * r at the rows, offset given at the center row."""
    pix = 0.001281815405036707
    fit_a, fit_b = 0.005630139932862342, 1.2281041144214863e-08
    Nd, Md = 5120, 3840
    center_row = (Md - 1) / 2
    for model in (projection_matrix2D_skew, projection_matrix2D_tilt):
        CM = model(0.3, Nd, Md, pix, pix, rotation_center_offset_pix=(fit_a + fit_b * center_row) / pix,
                   rotation_center_slope=fit_b / pix)
        for r in [0.0, center_row, Md - 1.0]:
            z = (r - CM[1, 3]) / CM[1, 2]
            PX, PY = project(CM, [0.0, 0.0, z])
            np.testing.assert_allclose([PX, PY], [(Nd - 1) / 2 + (fit_a + fit_b * r) / pix, r], atol=1e-8)


@pytest.mark.parametrize("material", CONVENTIONS)
@pytest.mark.parametrize("theta", THETAS)
def test_skew_offset_refers_to_center_row(theta, material):
    """rotation_center_offset_pix of projection_matrix2D_skew is the offset in the center row (M - 1) / 2,
    also when detector_center_offsetvy and detector_centerz move the row py0 of z = 0."""
    rcop, vx = 0.75, 0.3
    CM = projection_matrix2D_skew(theta, N, M, pixel_sizex, pixel_sizey, detector_centerz=1.1,
                                  detector_center_offsetvx=vx, detector_center_offsetvy=-0.15,
                                  rotation_center_offset_pix=rcop, rotation_center_slope=SLOPE,
                                  material_ct_convention=material)
    assert abs(CM[1, 3] - (M - 1) / 2) > 1.0  # py0 is not the center row
    for z in (-0.3, 0.0, 0.2):
        PX, PY = project(CM, [0.0, 0.0, z])
        np.testing.assert_allclose(PX, (N - 1) / 2 + rcop + SLOPE * (PY - (M - 1) / 2) - vx / pixel_sizex, atol=1e-10)
    CM1 = projection_matrix1D(theta, N, pixel_sizex, detector_center_offsetvx=vx,
                              rotation_center_offset_pix=rcop - SLOPE * (M - 1) / 2, rotation_center_slope=SLOPE,
                              material_ct_convention=material)
    for x in ([0.3, -0.25, 0.17], [-0.4, 0.1, -0.33]):
        PX, PY = project(CM, x)
        np.testing.assert_allclose(project(CM1, [x[0], x[1], PY])[0], PX, atol=1e-10)
