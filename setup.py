#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Install script for sktime."""

# adapted from https://github.com/scikit-learn/scikit-learn/blob/master
# /setup.py

####################

# Helpers for OpenMP support during the build.

# adapted fom https://github.com/scikit-learn/scikit-learn/blob/master
# /sklearn/_build_utils/openmp_helpers.py
# This code is adapted for a large part from the astropy openmp helpers, which
# can be found at: https://github.com/astropy/astropy-helpers/blob/master
# /astropy_helpers/openmp_helpers.py  # noqa


__author__ = ["Markus Löning", "lmmentel"]

import codecs
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from distutils.command.clean import clean as Clean  # noqa
from distutils.errors import CompileError, LinkError
from distutils.extension import Extension
from distutils.sysconfig import customize_compiler

import numpy
import toml
from Cython.Build import cythonize
from numpy.distutils.ccompiler import new_compiler
from setuptools import find_packages

pyproject = toml.load("pyproject.toml")

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
    """Return an compiler and platform specific flag for OpenMP."""
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
    """Check whether OpenMP test code can be compiled and run."""
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
            extra_preargs = (
                extra_preargs.split(" ") if extra_preargs is not None else []
            )
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


def long_description():
    """Read and return README as long description."""
    with codecs.open("README.md", encoding="utf-8-sig") as f:
        return f.read()


# ground truth package metadata is loaded from pyproject.toml
# for context see:
#   - [PEP 621 -- Storing project metadata in pyproject.toml]
#     (https://www.python.org/dev/peps/pep-0621)
pyproject = toml.load("pyproject.toml")

HERE = os.path.abspath(os.path.dirname(__file__))


# Custom clean command to remove build artifacts
class CleanCommand(Clean):
    """Remove build artifacts from the source tree."""

    description = "Remove build artifacts from the source tree"

    def run(self):
        """Run."""
        Clean.run(self)

        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, "PKG-INFO"))
        if remove_c_files:
            print("Will remove generated .c files")  # noqa: T001
        if os.path.exists("build"):
            shutil.rmtree("build")
        for dirpath, dirnames, filenames in os.walk("sktime"):
            for filename in filenames:
                if any(
                    filename.endswith(suffix)
                    for suffix in (".so", ".pyd", ".dll", ".pyc")
                ):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in [".c", ".cpp"]:
                    pyx_file = str.replace(filename, extension, ".pyx")
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == "__pycache__":
                    shutil.rmtree(os.path.join(dirpath, dirname))


cmdclass = {"clean": CleanCommand}


def configuration(parent_package="", top_path=None):
    """Configure."""
    if os.path.exists("MANIFEST"):
        os.remove("MANIFEST")

    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True,
    )
    config.add_subpackage("sktime")

    return config


extensions = [
    Extension(
        "sktime.__check_build._check_build",
        ["sktime/__check_build/_check_build.pyx"],
        extra_compile_args=["-O2"],
    ),
    Extension(
        "sktime.distances.elastic_cython",
        ["sktime/distances/elastic_cython.pyx"],
        extra_compile_args=["/d2FH4-", "-O2"] if sys.platform == "win32" else ["-O2"],
        language="c++",
        include_dirs=[numpy.get_include()],
    ),
]


def setup_package():
    """Set up package."""
    metadata = dict(
        author_email=pyproject["project"]["authors"][0]["email"],
        author=pyproject["project"]["authors"][0]["name"],
        classifiers=pyproject["project"]["classifiers"],
        cmdclass=cmdclass,
        description=pyproject["project"]["description"],
        download_url=pyproject["project"]["urls"]["download"],
        ext_modules=cythonize(extensions, language_level="3"),
        extras_require=pyproject["project"]["optional-dependencies"],
        include_package_data=True,
        install_requires=pyproject["project"]["dependencies"],
        keywords=pyproject["project"]["keywords"],
        license=pyproject["project"]["license"],
        long_description=long_description(),
        maintainer_email=pyproject["project"]["maintainers"][0]["email"],
        maintainer=pyproject["project"]["maintainers"][0]["name"],
        name=pyproject["project"]["name"],
        package_data={
            "sktime": [
                "*.csv",
                "*.csv.gz",
                "*.arff",
                "*.arff.gz",
                "*.txt",
                "*.ts",
                "*.tsv",
            ]
        },
        packages=find_packages(
            where=".",
            exclude=["tests", "tests.*"],
        ),
        project_urls=pyproject["project"]["urls"],
        python_requires=pyproject["project"]["requires-python"],
        setup_requires=pyproject["build-system"]["requires"],
        url=pyproject["project"]["urls"]["repository"],
        version=pyproject["project"]["version"],
        zip_safe=False,
    )

    from setuptools import setup

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
