 #! /usr/bin/env python
"""Install script for py-hive-cote"""
from __future__ import division, print_function, absolute_import

import sys
import os
import ast

import numpy as np
import codecs

from setuptools import find_packages, setup
from setuptools.extension import Extension



try:
    from Cython.Build import cythonize
except ImportError:
    sys.exit(
        "Cython not found. Cython is needed to build the extension modules.")

# get __version__ from _version.py
ver_file = "_version.py"
with open(ver_file) as f:
    exec(f.read())

build_type = "optimized"
dataexts = (".py", ".pyx", ".pxd", ".c", ".cpp", ".h", ".sh", ".lyx", ".tex",
            ".txt", ".pdf")
datadirs = ()
standard_docs = ["README", "LICENSE", "TODO", "CHANGELOG", "AUTHORS"]
standard_doc_exts = [".md", ".rst", ".txt", "", ".org"]

DISTNAME = 'py_hive_cote'
DESCRIPTION = 'Python repo UEA'
with codecs.open('README.md', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'TonyBagnall'
MAINTAINER_EMAIL = 'Anthony.Bagnall@uea.ac.uk'
URL = 'https://github.com/TonyBagnall/py-hive-cote'
LICENSE = 'undecided'
DOWNLOAD_URL = 'https://github.com/TonyBagnall/py-hive-cote'
VERSION = __version__
INSTALL_REQUIRES = ['numpy', 'scipy', 'scikit-learn', 'pandas', 'xpandas']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Cython',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}
extra_compile_args_math_optimized = [
    '-march=native',
    '-O2',
    '-msse',
    '-msse2',
    '-mfma',
    '-mfpmath=sse',
]
extra_compile_args_math_debug = [
    '-march=native',
    '-O0',
    '-g',
]

extra_link_args_math_optimized = []
extra_link_args_math_debug = []

extra_compile_args_nonmath_optimized = ['-O2']
extra_compile_args_nonmath_debug = ['-O0', '-g']
extra_link_args_nonmath_optimized = []
extra_link_args_nonmath_debug = []

openmp_compile_args = ['-fopenmp']
openmp_link_args = ['-fopenmp']
my_include_dirs = [np.get_include()]

if build_type == 'optimized':
    my_extra_compile_args_math = extra_compile_args_math_optimized
    my_extra_compile_args_nonmath = extra_compile_args_nonmath_optimized
    my_extra_link_args_math = extra_link_args_math_optimized
    my_extra_link_args_nonmath = extra_link_args_nonmath_optimized
    my_debug = False
    print("build configuration selected: optimized")
elif build_type == 'debug':
    my_extra_compile_args_math = extra_compile_args_math_debug
    my_extra_compile_args_nonmath = extra_compile_args_nonmath_debug
    my_extra_link_args_math = extra_link_args_math_debug
    my_extra_link_args_nonmath = extra_link_args_nonmath_debug
    my_debug = True
    print("build configuration selected: debug")
else:
    raise ValueError(
        "Unknown build configuration '%s'; valid: 'optimized', 'debug'" %
        (build_type))


def declare_cython_extension(extName,
                             use_math=False,
                             use_openmp=False,
                             include_dirs=None):
    extPath = extName.replace(".", os.path.sep) + ".pyx"
    if use_math:
        compile_args = list(my_extra_compile_args_math)  # copy
        link_args = list(my_extra_link_args_math)
        libraries = ["m"]
    else:
        compile_args = list(my_extra_compile_args_nonmath)
        link_args = list(my_extra_link_args_nonmath)
        libraries = None

    if use_openmp:
        compile_args.insert(0, openmp_compile_args)
        link_args.insert(0, openmp_link_args)

    return Extension(
        extName, [extPath],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs,
        libraries=libraries)


datafiles = []


def getext(filename):
    os.path.splitext(filename)[1]


for datadir in datadirs:
    datafiles.extend(
        [(root,
          [os.path.join(root, f) for f in files if getext(f) in dataexts])
         for root, dirs, files in os.walk(datadir)])

detected_docs = []
for docname in standard_docs:
    for ext in standard_doc_exts:
        filename = "".join((docname, ext))
        if os.path.isfile(filename):
            detected_docs.append(filename)
datafiles.append(('.', detected_docs))

init_py_path = '_version.py'
version = '0.0.unknown'
try:
    with open(init_py_path) as f:
        for line in f:
            if line.startswith('__version__'):
                version = ast.parse(line).body[0].value.s
                break
        else:
            print(
                "WARNING: Version information not found"
                " in '%s', using placeholder '%s'" % (init_py_path, version),
                file=sys.stderr)
except FileNotFoundError:
    print(
        "WARNING: Could not find file '%s',"
        "using placeholder version information '%s'" % (init_py_path, version),
        file=sys.stderr)
    
ext_module_utils = declare_cython_extension(
    "distances._utils",
    use_math=True,
    use_openmp=False,
    include_dirs=my_include_dirs)

ext_module_distance = declare_cython_extension(
    "distances._distance",
    use_math=True,
    use_openmp=False,
    include_dirs=my_include_dirs)

ext_module_euclidean_distance = declare_cython_extension(
    "distances._euclidean_distance",
    use_math=True,
    use_openmp=False,
    include_dirs=my_include_dirs)

ext_module_dtw_distance = declare_cython_extension(
    "distances._dtw_distance",
    use_math=True,
    use_openmp=False,
    include_dirs=my_include_dirs)

ext_module_impurity = declare_cython_extension(
    "distances._impurity",
    use_math=True,
    use_openmp=False,
    include_dirs=my_include_dirs)

ext_module_tree_builder = declare_cython_extension(
    "distances._tree_builder",
    use_math=True,
    use_openmp=False,
    include_dirs=my_include_dirs)

ext_module_distance_api = declare_cython_extension(
    "distances.distance",
    use_math=False,
    use_openmp=False,
    include_dirs=my_include_dirs)

# this is mainly to allow a manual logical ordering of the declared modules
#
cython_ext_modules = [
    ext_module_utils,
    ext_module_distance,
    ext_module_euclidean_distance,
    ext_module_dtw_distance,
    ext_module_distance_api,
    ext_module_impurity,
    ext_module_tree_builder,
]

my_ext_modules = cythonize(
    cython_ext_modules,
    include_path=my_include_dirs,
    gdb_debug=my_debug,
)


setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
#      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      setup_requires=["cython", "numpy"],
      classifiers=CLASSIFIERS,
      ext_modules=my_ext_modules,
      packages=find_packages(),
      package_data={
        'py_hive_cote': ['*.pxd', '*.pyx'],
   	  },
      install_requires=INSTALL_REQUIRES,
	  extras_require=EXTRAS_REQUIRE,
	  data_files=datafiles)
