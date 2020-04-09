#! /usr/bin/env python
"""Install script for sktime"""

# adapted from https://github.com/scikit-learn/scikit-learn/blob/d2476fb679f05e80c56e8b151ff0f6d7a470e4ae/setup.py#L20

import setuptools # need this due to versioning of setuptools vs distutils, see https://stackoverflow.com/questions/21136266/typeerror-dist-must-be-a-distribution-instance
import codecs
import os
import platform
import re
import shutil
import sys
import traceback
from distutils.command.clean import clean as Clean

from pkg_resources import parse_version

NUMPY_MIN_VERSION = "1.17.0"
SCIPY_MIN_VERSION = "1.2.0"
JOBLIB_MIN_VERSION = "0.13"
PANDAS_MIN_VERSION = "0.24.0"
SKLEARN_MIN_VERSION = "0.21.0"
STATSMODELS_MIN_VERSION = "0.9.0"
SCIKIT_POSTHOCS_MIN_VERSION = "0.5.0"
NUMBA_MIN_VERSION = "0.47"

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(HERE, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")


WEBSITE = 'https://alan-turing-institute.github.io/sktime/'
DISTNAME = 'sktime'
DESCRIPTION = 'scikit-learn compatible Python toolbox for machine learning with time series'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'F. Király'
MAINTAINER_EMAIL = 'fkiraly@turing.ac.uk'
URL = 'https://github.com/alan-turing-institute/sktime'
LICENSE = 'BSD-3-Clause'
DOWNLOAD_URL = 'https://pypi.org/project/sktime/#files'
PROJECT_URLS = {
    'Issue Tracker': 'https://github.com/alan-turing-institute/sktime/issues',
    'Documentation': WEBSITE,
    'Source Code': 'https://github.com/alan-turing-institute/sktime'
}
VERSION = find_version('sktime', '__init__.py')
INSTALL_REQUIRES = (
    'numpy>={}'.format(NUMPY_MIN_VERSION),
    'scipy>={}'.format(SCIPY_MIN_VERSION),
    'scikit-learn>={}'.format(SKLEARN_MIN_VERSION),
    'pandas>={}'.format(PANDAS_MIN_VERSION),
    'joblib>={}'.format(JOBLIB_MIN_VERSION),
    'scikit-posthocs>={}'.format(SCIKIT_POSTHOCS_MIN_VERSION),
    'statsmodels>={}'.format(STATSMODELS_MIN_VERSION),
    'numba>={}'.format(NUMBA_MIN_VERSION)
)
CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'License :: OSI Approved',
    'Programming Language :: Python',
    'Topic :: Software Development',
    'Topic :: Scientific/Engineering',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7'
]
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov',
        'tsfresh'
    ],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

# Optional setuptools features
# We need to import setuptools early, if we want setuptools features,
# as it monkey-patches the 'setup' function
# For some commands, use setuptools
SETUPTOOLS_COMMANDS = {
    'develop', 'release', 'bdist_egg', 'bdist_rpm',
    'bdist_wininst', 'install_egg_info', 'build_sphinx',
    'egg_info', 'easy_install', 'upload', 'bdist_wheel',
    '--single-version-externally-managed',
}
if SETUPTOOLS_COMMANDS.intersection(sys.argv):
    import setuptools

    extra_setuptools_args = dict(
        zip_safe=False,  # the package can run out of an .egg file
        include_package_data=True,
        extras_require={
            'alldeps': EXTRAS_REQUIRE
        },
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
        remove_c_files = not os.path.exists(os.path.join(cwd, 'PKG-INFO'))
        if remove_c_files:
            print('Will remove generated .c files')
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('sktime'):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in ['.c', '.cpp']:
                    pyx_file = str.replace(filename, extension, '.pyx')
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))


cmdclass = {'clean': CleanCommand}

# custom build_ext command to set OpenMP compile flags depending on os and
# compiler
# build_ext has to be imported after setuptools
try:
    from numpy.distutils.command.build_ext import build_ext  # noqa


    class build_ext_subclass(build_ext):

        def build_extensions(self):
            from sktime._build_utils.openmp_helpers import get_openmp_flag

            if not os.getenv('SKTIME_NO_OPENMP'):
                openmp_flag = get_openmp_flag(self.compiler)

                for e in self.extensions:
                    e.extra_compile_args += openmp_flag
                    e.extra_link_args += openmp_flag

            build_ext.build_extensions(self)


    cmdclass['build_ext'] = build_ext_subclass

except ImportError:
    # Numpy should not be a dependency just to be able to introspect
    # that python 3.6 is required.
    pass


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True
    )
    config.add_subpackage('sktime')

    return config


def get_numpy_status():
    """
    Returns a dictionary containing a boolean specifying whether NumPy
    is up-to-date, along with the version string (empty string if
    not installed).
    """
    numpy_status = {}
    try:
        import numpy
        numpy_version = numpy.__version__
        numpy_status['up_to_date'] = parse_version(
            numpy_version) >= parse_version(NUMPY_MIN_VERSION)
        numpy_status['version'] = numpy_version
    except ImportError:
        traceback.print_exc()
        numpy_status['up_to_date'] = False
        numpy_status['version'] = ""
    return numpy_status


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
        python_requires=">=3.6",
        install_requires=INSTALL_REQUIRES,
        **extra_setuptools_args
    )

    # For some actions, NumPy is not required.
    # They are required to succeed without Numpy for example when
    # pip is used to install sktime when Numpy is not yet
    # present in the system.
    if len(sys.argv) == 1 or (
            len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
                                    sys.argv[1] in ('--help-commands',
                                                    'egg_info',
                                                    '--version',
                                                    'clean'))):
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

        metadata['version'] = VERSION

    # otherwise check Python and NumPy version
    else:
        if sys.version_info < (3, 6):
            raise RuntimeError(
                "sktime requires Python 3.6 or later. The current"
                " Python version is %s installed in %s."
                % (platform.python_version(), sys.executable))

        numpy_status = get_numpy_status()
        numpy_req_str = "sktime requires NumPy >= {}.\n".format(
            NUMPY_MIN_VERSION)

        instructions = (f"Installation instructions are available on the "
                        f"sktime website: {WEBSITE}\n")

        if numpy_status['up_to_date'] is False:
            if numpy_status['version']:
                raise ImportError("Your installation of "
                                  "NumPy {} is out-of-date.\n{}{}"
                                  .format(numpy_status['version'],
                                          numpy_req_str, instructions))
            else:
                raise ImportError("NumPy is not installed.\n{}{}"
                                  .format(numpy_req_str, instructions))

        from numpy.distutils.core import setup

        metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
