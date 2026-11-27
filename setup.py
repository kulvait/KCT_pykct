#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyKCT - Python bindings for KCT C++ Parallel Beam 2D Projector

Setup script for installation.
"""

from setuptools import setup, find_packages

setup(
    name="pykct",
    version="0.2.0",
    author="Vojtěch Kulvait",
    author_email="kulvait@gmail.com",
    description="Python bindings for KCT C++ Parallel Beam 2D Projector",
    long_description=open("README.md").read() if __name__ == "__main__" else "",
    long_description_content_type="text/markdown",
    url="https://github.com/kulvait/KCT_pykct",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.7.0",
        "denpy>=0.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "matplotlib>=3.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "pykct-create-ball-phantom=pykct.scripts.create_ball_phantom:main",
            "pykct-create-shepp-logan=pykct.scripts.create_shepp_logan:main",
            "pykct-create-ball-sinogram=pykct.scripts.create_ball_sinogram:main",
            "pykct-create-shepp-logan-sinogram=pykct.scripts.create_shepp_logan_sinogram:main",
            "pykct-den-centroid=pykct.scripts.den_centroid:main",
            "pykct-create-camera-matrices-parallel-ray3d=pykct.scripts.create_camera_matrices_parallel_ray3d:main",
        ],
    },
)
