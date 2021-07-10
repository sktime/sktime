#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Install script for sktime"""

# adapted from https://github.com/scikit-learn/scikit-learn/blob/master
# /setup.py

__author__ = ["Markus Löning"]

import codecs
import importlib
import os
import platform
import re
import shutil
import sys
import traceback
from distutils.command.clean import clean as Clean  # noqa

from pkg_resources import parse_version

MIN_PYTHON_VERSION = "3.6"
MIN_REQUIREMENTS = {
    "numpy": "1.19.0",
    "pandas": "1.1.0",
    "scikit-learn": "0.24.0",
    "statsmodels": "0.12.1",
    "numba": "0.50",
}
EXTRAS_REQUIRE = {
    "all_extras": [
        "cython>=0.29.0",
        "matplotlib>=3.3.2",
        "pmdarima>=1.8.0,!=1.8.1",
        "scikit_posthocs>= 0.6.5",
        "seaborn>=0.11.0",
        "tsfresh>=0.17.0",
        "hcrystalball>=0.1.9",
        "stumpy>=1.5.1",
        "tbats>=1.1.0",
        "fbprophet>=0.7.1",
        "pyod>=0.8.0",
    ],
}

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(HERE, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")


WEBSITE = "https://www.sktime.org"
DISTNAME = "sktime"
DESCRIPTION = "A unified Python toolbox for machine learning with time series"
with codecs.open("README.rst", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "F. Király"
MAINTAINER_EMAIL = "f.kiraly@ucl.ac.uk"
URL = "https://github.com/alan-turing-institute/sktime"
LICENSE = "BSD-3-Clause"
DOWNLOAD_URL = "https://pypi.org/project/sktime/#files"
PROJECT_URLS = {
    "Issue Tracker": "https://github.com/alan-turing-institute/sktime/issues",
    "Documentation": WEBSITE,
    "Source Code": "https://github.com/alan-turing-institute/sktime",
}
VERSION = find_version("sktime", "__init__.py")
INSTALL_REQUIRES = [
    *[
        "{}>={}".format(package, version)
        for package, version in MIN_REQUIREMENTS.items()
    ],
    "wheel",
]
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
]

SETUP_REQUIRES = ["wheel"]

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
    extra_setuptools_args = dict()


# Custom clean command to remove build artifacts
class CleanCommand(Clean):
    description = "Remove build artifacts from the source tree"

    def run(self):
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
        def build_extensions(self):
            from sktime._build_utils.openmp_helpers import get_openmp_flag

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


def check_package_status(package, min_version):
    """
    Returns a dictionary containing a boolean specifying whether given package
    is up-to-date, along with the version string (empty string if
    not installed).
    """
    if package == "scikit-learn":
        package = "sklearn"

    package_status = {}
    try:
        module = importlib.import_module(package)
        package_version = module.__version__
        package_status["up_to_date"] = parse_version(package_version) >= parse_version(
            min_version
        )
        package_status["version"] = package_version
    except ImportError:
        traceback.print_exc()
        package_status["up_to_date"] = False
        package_status["version"] = ""

    req_str = "sktime requires {} >= {}.\n".format(package, min_version)

    instructions = (
        "Installation instructions are available on the "
        "sktime website: "
        "https://www.sktime.org/en/latest"
        "/installation.html\n"
    )

    if package_status["up_to_date"] is False:
        if package_status["version"]:
            raise ImportError(
                "Your installation of {} "
                "{} is out-of-date.\n{}{}".format(
                    package, package_status["version"], req_str, instructions
                )
            )
        else:
            raise ImportError(
                "{} is not " "installed.\n{}{}".format(package, req_str, instructions)
            )


def setup_package():
    metadata = dict(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        download_url=DOWNLOAD_URL,
        project_urls=PROJECT_URLS,
        version=VERSION,
        long_description=LONG_DESCRIPTION,
        classifiers=CLASSIFIERS,
        cmdclass=cmdclass,
        python_requires=">={}".format(MIN_PYTHON_VERSION),
        setup_requires=SETUP_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        **extra_setuptools_args
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

        metadata["version"] = VERSION

    # otherwise check Python and required package versions
    else:
        if sys.version_info < tuple([int(i) for i in MIN_PYTHON_VERSION.split(".")]):
            raise RuntimeError(
                "sktime requires Python %s or later. The current"
                " Python version is %s installed in %s."
                % (MIN_PYTHON_VERSION, platform.python_version(), sys.executable)
            )

        for package, version in MIN_REQUIREMENTS.items():
            check_package_status(package, version)

        from numpy.distutils.core import setup

        metadata["configuration"] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
