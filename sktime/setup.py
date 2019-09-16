import os

from sktime._build_utils import maybe_cythonize_extensions
from setuptools import find_packages

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('sktime', parent_package, top_path)

    for package in find_packages('sktime'):
        config.add_subpackage(package)

    # submodules with build utilities
    # config.add_subpackage('__check_build')
    # config.add_subpackage('_build_utils')
    #
    # # submodules which do not have their own setup.py
    # # we must manually add sub-submodules & tests
    # config.add_subpackage('benchmarking')
    # config.add_subpackage('benchmarking/tests')
    # config.add_subpackage('classifiers')
    # config.add_subpackage('contrib')
    # config.add_subpackage('datasets')
    # config.add_subpackage('forecasters')
    # config.add_subpackage('highlevel')
    # config.add_subpackage('metrics')
    # config.add_subpackage('regressors')
    # config.add_subpackage('transformers')
    # config.add_subpackage('utils')
    #
    # # submodules which have their own setup.py
    # config.add_subpackage('distances')
    #
    # # add the test directory
    # config.add_subpackage('tests')

    maybe_cythonize_extensions(top_path, config)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
