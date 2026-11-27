#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create projection matrices of a circular scan in the parallel ray geometry.

Port of the KCT script createCameraMatricesForCircularScanTrajectoryParallelRay3D.py
with the same options, the matrices are computed by pykct.PB2D.projection_matrix2D,
with --flat-matrices by pykct.PB2D.projection_matrix1D.
See also
https://kulvait.github.io/KCT_doc/posts/working-with-kct-cbct-5-parallel-beam-geometry.html

Usage:
    pykct-create-camera-matrices-parallel-ray3d CM.den --number-of-angles 1800 --theta-angular-range 180
"""

import argparse
import json

import numpy as np
from denpy import DEN

from pykct.PB2D import projection_matrix1D, projection_matrix2D
from pykct.scripts._common import check_output


def petra_compatibility_transform(theta):
    """
    Map the rotation angle of a PETRA dataset to the angle theta of the Radon
    transform, so that the geometry matches the convention of the PETRA
    reconstruction scripts https://github.com/moosmann/matlab.
    """
    return -theta - 0.5 * np.pi


def main():
    # Defaults from https://confluence.desy.de/display/P5I/Detectors, KIT CMOS, no magnification
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("outputMatrixFile")
    parser.add_argument("--projection-sizex", type=int, default=5120, help="PX dimension of detector in pixel count.")
    parser.add_argument("--projection-sizey", type=int, default=3840, help="PY dimension of detector in pixel count.")
    parser.add_argument("--pixel-sizex", type=float, default=0.064, help="X pixel size.")
    parser.add_argument("--pixel-sizey", type=float, default=0.064, help="Y pixel size.")
    parser.add_argument("--detector-centerx", type=float, default=0., help="Coordinate x of the center of the detector.")
    parser.add_argument("--detector-centery", type=float, default=0., help="Coordinate y of the center of the detector.")
    parser.add_argument("--detector-centerz", type=float, default=0., help="Coordinate z of the center of the detector.")
    parser.add_argument("--detector-center-offsetvx", type=float, default=0., help="Offset of the center of the detector, detector_center_offsetvx * VX/|VX| + detector_center_offsetvy * VY/|VY| is added to the coordinates of the center of the detector for each angle.")
    parser.add_argument("--detector-center-offsetvy", type=float, default=0., help="Offset of the center of the detector, detector_center_offsetvx * VX/|VX| + detector_center_offsetvy * VY/|VY| is added to the coordinates of the center of the detector for each angle.")
    parser.add_argument("--rotation-center-offset-pix", type=float, default=0., help="Shift of the image of the rotation axis in pixels, positive towards higher PX, the rotation axis is imaged at (projection_sizex-1)/2 + ROTATION_CENTER_OFFSET_PIX - detector_center_offsetvx/pixel_sizex.")
    parser.add_argument("--flat-matrices", action="store_true", help="Store 1x4 matrices of the kct-pb2d-* tools, see pykct.PB2D.projection_matrix1D, instead of 2x4 matrices. They equal the first rows of the 2x4 matrices, --projection-sizey, --pixel-sizey, --detector-centerz and --detector-center-offsetvy do not apply.")
    parser.add_argument("--number-of-angles", type=int, default=5001, help="Number of views of the circular trajectory.")
    parser.add_argument("--theta-zero", type=float, default=0, help="Initial angle theta from Radon transform in degrees. See also https://kulvait.github.io/KCT_doc/posts/tomographic-notes-1-geometric-conventions.html")
    parser.add_argument("--theta-angular-range", type=float, default=360, help="Angular range in degrees along which positions are distributed.")
    parser.add_argument("--endpoint", action="store_true", default=False, help="Include thetaZero+thetaAngularRange as an endpoint of the discretization, by default it is not included as it often coincides with the start point.")
    parser.add_argument("--material-ct-convention", action="store_true", default=False, help="The z axis direction and PY direction will coincide, that is usually not the case in medical CT praxis. See also https://kulvait.github.io/KCT_doc/posts/tomographic-notes-1-geometric-conventions.html.")
    parser.add_argument("--petra-compatibility-transform", action="store_true", help="Interpret the angles as rotation angles of a PETRA dataset and transform them by theta = -angle - pi/2, so that the geometry matches the convention of https://github.com/moosmann/matlab.")
    parser.add_argument("--angles-mat", type=str, default=None, help="Sequence of angles in radians stored as variable angles in a mat file, overrides the circular discretization.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--write-params-file", action="store_true", help="Write parameters of the call to OUTPUTMATRIXFILE.params.")
    parser.add_argument('--_json-message', default="Created using pykct script create_camera_matrices_parallel_ray3d.py", help=argparse.SUPPRESS)
    ARG = parser.parse_args()

    check_output(ARG.outputMatrixFile, ARG.force)
    paramsFile = "%s.params" % ARG.outputMatrixFile
    if ARG.write_params_file:
        check_output(paramsFile, ARG.force)

    if ARG.angles_mat is not None:
        import scipy.io
        directionAngles = np.ravel(scipy.io.loadmat(ARG.angles_mat)["angles"])
    else:
        THETA = np.pi * ARG.theta_zero / 180
        RANGE = np.pi * ARG.theta_angular_range / 180
        directionAngles = np.linspace(THETA, THETA + RANGE, num=ARG.number_of_angles, endpoint=ARG.endpoint)
    if ARG.petra_compatibility_transform:
        directionAngles = [petra_compatibility_transform(x) for x in directionAngles]

    CameraMatrices = []
    for theta in directionAngles:
        if ARG.flat_matrices:
            CM = projection_matrix1D(float(theta), ARG.projection_sizex, ARG.pixel_sizex,
                                     detector_centerx=ARG.detector_centerx,
                                     detector_centery=ARG.detector_centery,
                                     detector_center_offsetvx=ARG.detector_center_offsetvx,
                                     rotation_center_offset_pix=ARG.rotation_center_offset_pix,
                                     material_ct_convention=ARG.material_ct_convention)
        else:
            CM = projection_matrix2D(float(theta), ARG.projection_sizex, ARG.projection_sizey,
                                     ARG.pixel_sizex, ARG.pixel_sizey,
                                     detector_centerx=ARG.detector_centerx,
                                     detector_centery=ARG.detector_centery,
                                     detector_centerz=ARG.detector_centerz,
                                     detector_center_offsetvx=ARG.detector_center_offsetvx,
                                     detector_center_offsetvy=ARG.detector_center_offsetvy,
                                     rotation_center_offset_pix=ARG.rotation_center_offset_pix,
                                     material_ct_convention=ARG.material_ct_convention)
        CameraMatrices.append(CM)
    CameraMatrices = np.array(CameraMatrices, dtype=np.float64).reshape((-1, 1 if ARG.flat_matrices else 2, 4))

    DEN.storeNdarrayAsDEN(ARG.outputMatrixFile, CameraMatrices, force=ARG.force)
    if ARG.write_params_file:
        with open(paramsFile, 'w') as f:
            json.dump(ARG.__dict__, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
