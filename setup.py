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


__author__ = ["Markus Löning"]

import codecs
import glob
import importlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import traceback
from distutils.command.clean import clean as Clean  # noqa
from distutils.errors import CompileError, LinkError
from distutils.extension import Extension
from distutils.sysconfig import customize_compiler

import toml
from Cython.Build import cythonize
from numpy.distutils.ccompiler import new_compiler
from pkg_resources import parse_version
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


def long_description():
    """Read and return README as long description."""
    with codecs.open("README.md", encoding="utf-8-sig") as f:
        return f.read()


# ground truth package metadata is loaded from pyproject.toml
# for context see:
#   - [PEP 621 -- Storing project metadata in pyproject.toml](https://www.python.org/dev/peps/pep-0621)
pyproject = toml.load("pyproject.toml")

HERE = os.path.abspath(os.path.dirname(__file__))

# Optional setuptools features
# For some commands, use setuptools
SETUPTOOLS_COMMANDS = {
    "install",
    "develop",
    "release",
    "build_ext",
    "bdist_egg",
    "bdist_rpm",
    "bdist_wininst",
    "install_egg_info",
    "build_sphinx",
    "egg_info",
    "easy_install",
    "upload",
    "bdist_wheel",
    "--single-version-externally-managed",
    "sdist",
}
if SETUPTOOLS_COMMANDS.intersection(sys.argv):
    # We need to import setuptools early, if we want setuptools features,
    # (e.g. "bdist_wheel") as it monkey-patches the 'setup' function
    import setuptools  # noqa

    extra_setuptools_args = dict(
        zip_safe=False,  # the package can run out of an .egg file
        include_package_data=True,
    )

else:
    extra_setuptools_args = {}


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

# custom build_ext command to set OpenMP compile flags depending on os and
# compiler
# build_ext has to be imported after setuptools
try:
    from numpy.distutils.command.build_ext import build_ext  # noqa

    class build_ext_subclass(build_ext):
        """Build extension subclass."""

        def build_extensions(self):
            """Build extensions."""
            # from sktime._build_utils.openmp_helpers import get_openmp_flag

            if not os.getenv("SKTIME_NO_OPENMP"):
                openmp_flag = get_openmp_flag(self.compiler)

                for e in self.extensions:
                    e.extra_compile_args += openmp_flag
                    e.extra_link_args += openmp_flag

            build_ext.build_extensions(self)

    cmdclass["build_ext"] = build_ext_subclass

except ImportError:
    # Numpy should not be a dependency just to be able to introspect
    # that python 3.6 is required.
    pass


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
        "sktime.distances.elastic_cython",
        ["sktime/distances/elastic_cython.pyx"],
        extra_compile_args=["-O2", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        language="c++",
    ),
    Extension(
        "sktime.classification.shapelet_based.mrseql.mrseql",
        ["sktime/classification/shapelet_based/mrseql/mrseql.pyx"],
        extra_compile_args=["-O2", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        language="c++",
    ),
]


def setup_package():
    """Set up package."""
    metadata = dict(
        name=pyproject["project"]["name"],
        author=pyproject["project"]["authors"][0]["name"],
        author_email=pyproject["project"]["authors"][0]["email"],
        maintainer=pyproject["project"]["maintainers"][0]["name"],
        maintainer_email=pyproject["project"]["maintainers"][0]["email"],
        description=pyproject["project"]["description"],
        license=pyproject["project"]["license"],
        keywords=pyproject["project"]["keywords"],
        url=pyproject["project"]["urls"]["repository"],
        download_url=pyproject["project"]["urls"]["download"],
        project_urls=pyproject["project"]["urls"],
        version=pyproject["project"]["version"],
        long_description=long_description(),
        classifiers=pyproject["project"]["classifiers"],
        cmdclass=cmdclass,
        python_requires=pyproject["project"]["requires-python"],
        setup_requires=pyproject["build-system"]["requires"],
        install_requires=pyproject["project"]["dependencies"],
        extras_require=pyproject["project"]["optional-dependencies"],
        ext_modules=cythonize(extensions),
        packages=find_packages(
            where=".",
            exclude=["tests", "tests.*"],
        ),
        **extra_setuptools_args,
    )

    # For these actions, NumPy is not required
    # They are required to succeed without Numpy for example when
    # pip is used to install sktime when Numpy is not yet
    # present in the system.
    if len(sys.argv) == 1 or (
        len(sys.argv) >= 2
        and (
            "--help" in sys.argv[1:]
            or sys.argv[1] in ("--help-commands", "egg_info", "--version", "clean")
        )
    ):
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

        metadata["version"] = pyproject["project"]["version"]

    else:
        from numpy.distutils.core import setup

        metadata["configuration"] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
