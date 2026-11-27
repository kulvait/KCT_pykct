#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create projection matrices of a circular scan in the parallel ray geometry.

Port of the KCT script createCameraMatricesForCircularScanTrajectoryParallelRay3D.py.
Two subcommands produce the two matrix formats of KCT:

    PB3D  2x4 matrices of the 3D parallel beam tools kct-pb-*,
          computed by pykct.PB2D.projection_matrix2D.
    PB2D  1x4 matrices of the 2D parallel beam tools kct-pb2d-*,
          computed by pykct.PB2D.projection_matrix1D.

See also
https://kulvait.github.io/KCT_doc/posts/working-with-kct-cbct-5-parallel-beam-geometry.html
https://kulvait.github.io/KCT_doc/posts/working-with-kct-cbct-6-parallel-beam-geometry-2d.html

The angles are either a uniform discretization of the circular scan
(--number-of-angles, --theta-zero, --theta-angular-range, --endpoint), an
explicit list --angles in degrees or a mat file --angles-mat in radians.

Usage:
    createCameraMatricesForCircularScanTrajectoryPB PB3D CM.den --number-of-angles 1800 --theta-angular-range 180
    createCameraMatricesForCircularScanTrajectoryPB PB2D CM.den --projection-sizex 2560 --angles 0,45,90.5
    createCameraMatricesForCircularScanTrajectoryPB PB2D CM.den --angles=-20,0,20
"""

import argparse
import json
import re
import sys

import numpy as np
from denpy import DEN

from pykct.PB2D import projection_matrix1D, projection_matrix2D
from pykct.scripts._common import check_output

# Defaults of the uniform discretization, the options default to None to detect their explicit use
NUMBER_OF_ANGLES = 5001
THETA_ZERO = 0.0
THETA_ANGULAR_RANGE = 360.0


def petra_compatibility_transform(theta):
    """
    Map the rotation angle of a PETRA dataset to the angle theta of the Radon
    transform, so that the geometry matches the convention of the PETRA
    reconstruction scripts https://github.com/moosmann/matlab.
    """
    return -theta - 0.5 * np.pi


def parse_angles(s):
    """
    Parse the list of angles in degrees separated by commas and/or whitespace,
    e.g. "20,60.001,180" or "20 60.001 180".
    """
    tokens = [t for t in re.split(r"[,\s]+", s.strip()) if t != ""]
    if len(tokens) == 0:
        raise argparse.ArgumentTypeError("empty list of angles")
    try:
        return np.array([float(t) for t in tokens], dtype=np.float64)
    except ValueError as e:
        raise argparse.ArgumentTypeError("invalid list of angles %r: %s" % (s, e))


def common_arguments():
    """Options shared by both subcommands, the subcommand specific ones are added in main."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("outputMatrixFile", help="Output DEN file of FLOAT64 projection matrices.")
    # Defaults from https://confluence.desy.de/display/P5I/Detectors, KIT CMOS, no magnification
    p.add_argument("--projection-sizex", type=int, default=5120, help="PX dimension of detector in pixel count.")
    p.add_argument("--pixel-sizex", type=float, default=0.064, help="X pixel size.")
    p.add_argument("--detector-centerx", type=float, default=0., help="Coordinate x of the center of the detector.")
    p.add_argument("--detector-centery", type=float, default=0., help="Coordinate y of the center of the detector.")
    p.add_argument("--detector-center-offsetvx", type=float, default=0., help="Offset of the center of the detector along VX/|VX| in units of length, added to the center of the detector for each angle.")
    p.add_argument("--rotation-center-offset-pix", type=float, default=0., help="Shift of the image of the rotation axis in pixels, positive towards higher PX, the rotation axis is imaged at (projection_sizex-1)/2 + ROTATION_CENTER_OFFSET_PIX - detector_center_offsetvx/pixel_sizex.")
    p.add_argument("--material-ct-convention", action="store_true", default=False, help="The z axis direction and PY direction will coincide, that is usually not the case in medical CT praxis. See also https://kulvait.github.io/KCT_doc/posts/tomographic-notes-1-geometric-conventions.html.")
    a = p.add_argument_group("angles", "Uniform discretization of the circular scan by default, or explicit angles by --angles or --angles-mat.")
    a.add_argument("--number-of-angles", type=int, default=None, help="Number of views of the circular trajectory. [default %d]" % NUMBER_OF_ANGLES)
    a.add_argument("--theta-zero", type=float, default=None, help="Initial angle theta from Radon transform in degrees. See also https://kulvait.github.io/KCT_doc/posts/tomographic-notes-1-geometric-conventions.html [default %g]" % THETA_ZERO)
    a.add_argument("--theta-angular-range", type=float, default=None, help="Angular range in degrees along which positions are distributed. [default %g]" % THETA_ANGULAR_RANGE)
    a.add_argument("--endpoint", action="store_true", default=False, help="Include thetaZero+thetaAngularRange as an endpoint of the discretization, by default it is not included as it often coincides with the start point.")
    a.add_argument("--angles", type=parse_angles, default=None, metavar="LIST", help="Explicit angles theta in degrees separated by commas, e.g. --angles 20,60.001,180. Use --angles=-20,0,20 when the first angle is negative.")
    a.add_argument("--angles-mat", type=str, default=None, help="Sequence of angles in radians stored as variable angles in a mat file.")
    a.add_argument("--petra-compatibility-transform", action="store_true", help="Interpret the angles as rotation angles of a PETRA dataset and transform them by theta = -angle - pi/2, so that the geometry matches the convention of https://github.com/moosmann/matlab.")
    p.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    p.add_argument("--write-params-file", action="store_true", help="Write parameters of the call to OUTPUTMATRIXFILE.params.")
    p.add_argument('--_json-message', default="Created using pykct script createCameraMatricesForCircularScanTrajectoryPB.py", help=argparse.SUPPRESS)
    return p


def direction_angles(parser, ARG):
    """Angles theta in radians, after the optional PETRA transform."""
    uniformSpecified = ARG.number_of_angles is not None or ARG.theta_zero is not None \
        or ARG.theta_angular_range is not None or ARG.endpoint
    explicit = [o for o, v in (("--angles", ARG.angles), ("--angles-mat", ARG.angles_mat)) if v is not None]
    if len(explicit) > 1:
        parser.error("--angles and --angles-mat are mutually exclusive")
    if len(explicit) == 1 and uniformSpecified:
        parser.error("%s can not be combined with --number-of-angles, --theta-zero, --theta-angular-range or --endpoint" % explicit[0])
    if ARG.angles is not None:
        theta = np.deg2rad(ARG.angles)
    elif ARG.angles_mat is not None:
        import scipy.io
        theta = np.ravel(scipy.io.loadmat(ARG.angles_mat)["angles"]).astype(np.float64)
    else:
        count = NUMBER_OF_ANGLES if ARG.number_of_angles is None else ARG.number_of_angles
        if count < 1:
            parser.error("--number-of-angles must be positive")
        start = np.deg2rad(THETA_ZERO if ARG.theta_zero is None else ARG.theta_zero)
        span = np.deg2rad(THETA_ANGULAR_RANGE if ARG.theta_angular_range is None else ARG.theta_angular_range)
        theta = np.linspace(start, start + span, num=count, endpoint=ARG.endpoint)
    if ARG.petra_compatibility_transform:
        theta = petra_compatibility_transform(theta)
    return theta


def matrices_PB3D(theta, ARG):
    return np.array([
        projection_matrix2D(float(t), ARG.projection_sizex, ARG.projection_sizey,
                            ARG.pixel_sizex, ARG.pixel_sizey,
                            detector_centerx=ARG.detector_centerx,
                            detector_centery=ARG.detector_centery,
                            detector_centerz=ARG.detector_centerz,
                            detector_center_offsetvx=ARG.detector_center_offsetvx,
                            detector_center_offsetvy=ARG.detector_center_offsetvy,
                            rotation_center_offset_pix=ARG.rotation_center_offset_pix,
                            material_ct_convention=ARG.material_ct_convention) for t in theta
    ], dtype=np.float64).reshape((-1, 2, 4))


def matrices_PB2D(theta, ARG):
    return np.array([
        projection_matrix1D(float(t), ARG.projection_sizex, ARG.pixel_sizex,
                            detector_centerx=ARG.detector_centerx,
                            detector_centery=ARG.detector_centery,
                            detector_center_offsetvx=ARG.detector_center_offsetvx,
                            rotation_center_offset_pix=ARG.rotation_center_offset_pix,
                            rotation_center_slope=ARG.rotation_center_slope,
                            material_ct_convention=ARG.material_ct_convention) for t in theta
    ], dtype=np.float64).reshape((-1, 1, 4))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="format", required=True, metavar="{PB3D,PB2D}")
    common = common_arguments()

    p3 = sub.add_parser("PB3D", parents=[common], formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                        help="2x4 matrices [[a, px0], [b, py0]] of the 3D parallel beam tools kct-pb-*.",
                        description="2x4 matrices [[a, px0], [b, py0]] mapping (x, y, z) to (PX, PY), see pykct.PB2D.projection_matrix2D.")
    p3.add_argument("--projection-sizey", type=int, default=3840, help="PY dimension of detector in pixel count.")
    p3.add_argument("--pixel-sizey", type=float, default=0.064, help="Y pixel size.")
    p3.add_argument("--detector-centerz", type=float, default=0., help="Coordinate z of the center of the detector.")
    p3.add_argument("--detector-center-offsetvy", type=float, default=0., help="Offset of the center of the detector along VY/|VY| in units of length, added to the center of the detector for each angle.")
    p3.set_defaults(create=matrices_PB3D, rows=2)

    p2 = sub.add_parser("PB2D", parents=[common], formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                        help="1x4 matrices [[a0, a1, a2, a3]] of the 2D parallel beam tools kct-pb2d-*.",
                        description="1x4 matrices [[a0, a1, a2, a3]] mapping (x, y, r) to PX = a0*x + a1*y + a2*r + a3, where r is the detector row index, see pykct.PB2D.projection_matrix1D. The kct-pb2d-* tools evaluate r = --z-level2d-origin + row index of the processed projections.")
    p2.add_argument("--rotation-center-slope", type=float, default=0., help="Shift of the image of the rotation axis in pixels per detector row r, a2 of the matrix. ROTATION_CENTER_OFFSET_PIX is then the shift in the row r = 0.")
    p2.set_defaults(create=matrices_PB2D, rows=1)

    ARG = parser.parse_args(argv)
    activeParser = p3 if ARG.format == "PB3D" else p2
    theta = direction_angles(activeParser, ARG)

    check_output(ARG.outputMatrixFile, ARG.force)
    paramsFile = "%s.params" % ARG.outputMatrixFile
    if ARG.write_params_file:
        check_output(paramsFile, ARG.force)

    CameraMatrices = ARG.create(theta, ARG)
    DEN.storeNdarrayAsDEN(ARG.outputMatrixFile, CameraMatrices, force=ARG.force)
    if ARG.write_params_file:
        params = {k: v for k, v in vars(ARG).items() if k not in ("create", "rows")}
        if params.get("angles") is not None:
            params["angles"] = [float(x) for x in params["angles"]]
        with open(paramsFile, 'w') as f:
            json.dump(params, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    sys.exit(main())
