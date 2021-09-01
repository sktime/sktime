# -*- coding: utf-8 -*-
"""Helpers for OpenMP support during the build."""

# adapted fom https://github.com/scikit-learn/scikit-learn/blob/master
# /sklearn/_build_utils/openmp_helpers.py
# This code is adapted for a large part from the astropy openmp helpers, which
# can be found at: https://github.com/astropy/astropy-helpers/blob/master
# /astropy_helpers/openmp_helpers.py  # noqa


import glob
import os
import subprocess
import sys
import tempfile
import textwrap
from distutils.errors import CompileError
from distutils.errors import LinkError
from distutils.sysconfig import customize_compiler

from numpy.distutils.ccompiler import new_compiler

CCODE = textwrap.dedent(
    """\
    #include <omp.h>
    #include <stdio.h>
    int main(void) {
    #pragma omp parallel
    printf("nthreads=%d\\n", omp_get_num_threads());
    return 0;
    }
    """
)


def get_openmp_flag(compiler):
    if hasattr(compiler, "compiler"):
        compiler = compiler.compiler[0]
    else:
        compiler = compiler.__class__.__name__

    if sys.platform == "win32" and ("icc" in compiler or "icl" in compiler):
        return ["/Qopenmp"]
    elif sys.platform == "win32":
        return ["/openmp"]
    elif sys.platform == "darwin" and ("icc" in compiler or "icl" in compiler):
        return ["-openmp"]
    elif sys.platform == "darwin" and "openmp" in os.getenv("CPPFLAGS", ""):
        # -fopenmp can't be passed as compile flag when using Apple-clang.
        # OpenMP support has to be enabled during preprocessing.
        #
        # For example, our macOS wheel build jobs use the following environment
        # variables to build with Apple-clang and the brew installed "libomp":
        #
        # export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
        # export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
        # export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
        # export LDFLAGS="$LDFLAGS -L/usr/local/opt/libomp/lib -lomp"
        # export DYLD_LIBRARY_PATH=/usr/local/opt/libomp/lib
        return []
    # Default flag for GCC and clang:
    return ["-fopenmp"]


def check_openmp_support():
    """Check whether OpenMP test code can be compiled and run"""
    ccompiler = new_compiler()
    customize_compiler(ccompiler)

    if os.getenv("SKTIME_NO_OPENMP"):
        # Build explicitly without OpenMP support
        return False

    start_dir = os.path.abspath(".")

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            os.chdir(tmp_dir)

            # Write test program
            with open("test_openmp.c", "w") as f:
                f.write(CCODE)

            os.mkdir("objects")

            # Compile, test program
            openmp_flags = get_openmp_flag(ccompiler)
            ccompiler.compile(
                ["test_openmp.c"], output_dir="objects", extra_postargs=openmp_flags
            )

            # Link test program
            extra_preargs = os.getenv("LDFLAGS", None)
            if extra_preargs is not None:
                extra_preargs = extra_preargs.split(" ")
            else:
                extra_preargs = []

            objects = glob.glob(os.path.join("objects", "*" + ccompiler.obj_extension))
            ccompiler.link_executable(
                objects,
                "test_openmp",
                extra_preargs=extra_preargs,
                extra_postargs=openmp_flags,
            )

            # Run test program
            output = subprocess.check_output("./test_openmp")
            output = output.decode(sys.stdout.encoding or "utf-8").splitlines()

            # Check test program output
            if "nthreads=" in output[0]:
                nthreads = int(output[0].strip().split("=")[1])
                openmp_supported = len(output) == nthreads
            else:
                openmp_supported = False

        except (CompileError, LinkError, subprocess.CalledProcessError):
            openmp_supported = False

        finally:
            os.chdir(start_dir)

    err_message = textwrap.dedent(
        """
                            ***
        It seems that sktime cannot be built with OpenMP support.

        - If your compiler supports OpenMP but the build still fails, please
          submit a bug report at:
          'https://github.com/alan-turing-institute/sktime/issues'

        - If you want to build sktime without OpenMP support, you can set
          the environment variable SKTIME_NO_OPENMP and rerun the build
          command. Note however that some estimators will run in sequential
          mode and their `n_jobs` parameter will have no effect anymore.

        - See sktime advanced installation instructions for more info:
          'https://https://www.sktime.org/en/latest/installation.html'
                            ***
        """
    )

    if not openmp_supported:
        raise CompileError(err_message)

    return True
