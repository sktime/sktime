"""Build script for sktime's compiled (Cython) extensions.

Configuration lives in ``pyproject.toml``; this file exists only to declare
``ext_modules``, which setuptools' pyproject backend cannot yet express.
"""

import sys

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

# fast-math is what makes the kernel beat numba; flags differ MSVC vs gcc/clang.
if sys.platform == "win32":
    _fast = ["/O2", "/fp:fast"]
else:
    _fast = ["-O3", "-ffast-math"]

ext_modules = cythonize(
    [
        Extension(
            "sktime.transformations.rocket._minirocket_multivariate_cython",
            ["sktime/transformations/rocket/_minirocket_multivariate_cython.pyx"],
            include_dirs=[numpy.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=_fast,
        )
    ],
    language_level="3",
)

setup(ext_modules=ext_modules)
