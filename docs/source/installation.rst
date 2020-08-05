.. _installation:

Installation
============

sktime is available via PyPI using:

.. code-block:: bash

    pip install sktime

But note that the package is actively being developed and currently not feature stable.


Development version
-------------------
To install the development version of sktime, follow these steps:

1. Download the repository: :code:`git clone https://github.com/alan-turing-institute/sktime.git`
2. Move into the root directory of the repository: :code:`cd sktime`
3. Switch to development branch: :code:`git checkout dev`
4. Make sure your local version is up-to-date: :code:`git pull`
5. Build package from source using: :code:`pip install --editable .`

Please read below for more detailed instrcutions for specific operating
systems.

Building sktime from source also requires

- Cython >= 0.28.5 (available through :code:`pip install cython`)
- OpenMP (see below for instructions)

.. note::

   It is possible to build sktime without OpenMP support by setting the
   ``SKTIME_NO_OPENMP`` environment variable (before cythonization). This is
   not recommended since it will force some estimators to run in sequential
   mode and their ``n_jobs`` parameter will be ignored.

Running tests requires

.. |PytestMinVersion| replace:: 3.3.0

- pytest >=\ |PytestMinVersion|

and a few soft dependencies which are required for running certain modules,
but not necessary for most of sktime's functionality.

Generating the documentation and website requires a few extra dependencies
too.

You can install all extra dependencies for running tests and generating
the documentation by running:

.. code-block:: bash

    pip install --editable .[docs]

Retrieving the latest code
~~~~~~~~~~~~~~~~~~~~~~~~~~

We use `Git <https://git-scm.com/>`_ for version control and
`GitHub <https://github.com/>`_ for hosting our main repository.

You can check out the latest sources with the command:

.. code-block:: bash

    git clone git://github.com/alan-turing-institute/sktime.git

If you want to build a stable version, you can ``git checkout <VERSION>``
to get the code for that particular version, or download an zip archive of
the version from github. To see which versions are available, run ``git tag``.

Building from source
~~~~~~~~~~~~~~~~~~~~

Once you have all the build requirements installed (see below for details),
you can build and install the package in the following way.

If you run the development version, it is cumbersome to reinstall the
package each time you update the sources. Therefore it's recommended that you
install in editable mode, which allows you to edit the code in-place. This
builds the extension in place and creates a link to the development directory
(see `the pip docs <https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`_):

.. code-block:: bash

    pip install --editable .

.. note::

    This is fundamentally similar to using the command ``python setup.py develop``
    (see `the setuptool docs <https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode>`_).
    It is however preferred to use pip.

.. note::

    You will have to re-run:

    .. code-block:: bash

        pip install --editable .

    every time the source code of a compiled extension is changed (for
    instance when switching branches or pulling changes from upstream).
    Compiled extensions are Cython files (ending in `.pyx` or `.pxd`).


Mac OSX
*******

The default C compiler, Apple-clang, on Mac OSX does not directly support
OpenMP. The first solution to build sktime is to install another C
compiler such as gcc or llvm-clang. Another solution is to enable OpenMP
support on the default Apple-clang. In the following we present how to
configure this second option.

You first need to install the OpenMP library:

.. code-block:: bash

    brew install libomp

Then you need to set the following environment variables:

.. code-block:: bash

    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
    export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
    export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
    export LDFLAGS="$LDFLAGS -L/usr/local/opt/libomp/lib -lomp"
    export DYLD_LIBRARY_PATH=/usr/local/opt/libomp/lib

Finally you can build the package using the standard command.

Alternatively, if you have other compilers, such as gcc, installed, the following one-liner will do the job:

.. code-block:: 

   env CC=$(which gcc-9) CXX=$(which g++-9) pip install --editable .

FreeBSD
*******

The clang compiler included in FreeBSD 12.0 and 11.2 base systems does not
include OpenMP support. You need to install the `openmp` library from packages
(or ports):

.. code-block:: bash

    sudo pkg install openmp

This will install header files in ``/usr/local/include`` and libs in
``/usr/local/lib``. Since these directories are not searched by default, you
can set the environment variables to these locations:

.. code-block:: bash

    export CFLAGS="$CFLAGS -I/usr/local/include"
    export CXXFLAGS="$CXXFLAGS -I/usr/local/include"
    export LDFLAGS="$LDFLAGS -L/usr/local/lib -lomp"
    export DYLD_LIBRARY_PATH=/usr/local/lib

Finally you can build the package using the standard command.

For the upcomming FreeBSD 12.1 and 11.3 versions, OpenMP will be included in
the base system and these steps will not be necessary.


Installing build dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Linux
*****

Installing from source without conda requires you to have installed the
sktime runtime dependencies, Python development headers and a working
C/C++ compiler. Under Debian-based operating systems, which include Ubuntu:

.. code-block:: bash

    sudo apt-get install build-essential python3-dev python3-setuptools \
                     python3-pip

and then:

.. code-block:: bash

    pip3 install numpy scipy cython


When precompiled wheels are not avalaible for your architecture, you can
install the system versions:

.. code-block:: bash

    sudo apt-get install cython3 python3-numpy python3-scipy python3-matplotlib

On Red Hat and clones (e.g. CentOS), install the dependencies using:

.. code-block:: bash

    sudo yum -y install gcc gcc-c++ python-devel numpy scipy

.. note::

    To use a high performance BLAS library (e.g. OpenBlas) see
    `scipy installation instructions
    <https://docs.scipy.org/doc/scipy/reference/building/linux.html>`_.

Windows
*******

To build sktime on Windows you need a working C/C++ compiler in
addition to numpy, scipy and setuptools.

The building command depends on the architecture of the Python interpreter,
32-bit or 64-bit. You can check the architecture by running the following in
``cmd`` or ``powershell`` console:

.. code-block:: bash

    python -c "import struct; print(struct.calcsize('P') * 8)"

The above commands assume that you have the Python installation folder in your
PATH environment variable.

You will need `Build Tools for Visual Studio 2017
<https://visualstudio.microsoft.com/downloads/>`_.

.. warning::
	You DO NOT need to install Visual Studio 2019.
	You only need the "Build Tools for Visual Studio 2019",
	under "All downloads" -> "Tools for Visual Studio 2019".

For 64-bit Python, configure the build environment with:

.. code-block:: bash

    SET DISTUTILS_USE_SDK=1
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

Please be aware that the path above might be different from user to user.
The aim is to point to the "vcvarsall.bat" file.

And build sktime from this environment:

.. code-block:: bash

    python setup.py install

Replace ``x64`` by ``x86`` to build for 32-bit Python.


Building binary packages and installers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``.whl`` package and ``.exe`` installers can be built with:

.. code-block:: bash

    pip install wheel
    python setup.py bdist_wheel bdist_wininst

The resulting packages are generated in the ``dist/`` folder.


Using an alternative compiler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to use `MinGW <http://www.mingw.org>`_ (a port of GCC to Windows
OS) as an alternative to MSVC for 32-bit Python. Not that extensions built with
mingw32 can be redistributed as reusable packages as they depend on GCC runtime
libraries typically not installed on end-users environment.

To force the use of a particular compiler, pass the ``--compiler`` flag to the
build step:

.. code-block:: bash

    python setup.py build --compiler=my_compiler install

where ``my_compiler`` should be one of ``mingw32`` or ``msvc``.


References
----------

The installation instruction are adapted from scikit-learn's advanced `installation instructions <https://scikit-learn.org/stable/developers/advanced_installation.html>`_.
