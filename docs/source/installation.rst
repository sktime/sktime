.. _installation:

Installation
============

``sktime`` currently supports:

* environments with python version 3.6, 3.7, or 3.8.
* operating systems Mac OS X, Unix-like OS, Windows 8.1 and higher

See here for a `full list of precompiled wheels available on PyPI <https://pypi.org/simple/sktime/>`_.

We appreciate community contributions towards compatibility with python 3.9, or other operating systems.

Release versions
----------------

For frequent issues with installation, consult the `Release versions - troubleshooting`_ section.

Installing sktime from PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``sktime`` releases are available via PyPI and can be installed via ``pip`` using:

.. code-block:: bash

    pip install sktime

This will install ``sktime`` with core dependencies, excluding soft dependencies.

To install ``sktime`` with maximum dependencies, including soft dependencies, install with the ``all_extras`` modifier:

.. code-block:: bash

    pip install sktime[all_extras]


Installing sktime from conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``sktime`` releases are available via ``conda`` from ``conda-forge``.
They can be installed via ``conda`` using:

.. code-block:: bash

    conda install -c conda-forge sktime

This will install ``sktime`` with core dependencies, excluding soft dependencies.

To install ``sktime`` with maximum dependencies, including soft dependencies, install with the ``all-extras`` recipe:

.. code-block:: bash

    conda install -c conda-forge sktime-all-extras

Note: currently this does not include dependencies ``catch-22``, ``pmdarima``, and ``tbats``.
As these packages are not available on ``conda-forge``, they must be installed via ``pip`` if desired.
Contributions to remedy this situation are appreciated.


Release versions - troubleshooting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Module not found
""""""""""""""""

The most frequent reason for "module not found" errors is installing ``sktime`` with minimum dependencies
and using an estimator which interfaces a packages that has not been installed in the environment.
To resolve this, install the missing package, or install ``sktime`` with maximum dependencies (see above).


Facebook prophet
""""""""""""""""

A frequent issue arises with installation of facebook prophet when installing via ``pip``, especially on Windows systems.

Potential workaround no.1, install manually via ``conda-forge``:

.. code-block:: bash

    conda install -c conda-forge pystan
    conda install -c conda-forge prophet

The remaining packages can be installed via ``pip`` or ``conda``.

Potential workaround no.2, install ``pystan`` with ``no-cache`` parameter:

.. code-block:: bash

    pip install pystan --no-cache

Potential workaround no.3, if on Windows: use WSL (Windows Subsystem for Linux), see end of section `Windows 8.1 and higher`_.


numpy or C related issues
"""""""""""""""""""""""""

``numpy`` and C related errors on Windows based systems are potentially resolved by installing Build Tools for Visual Studio 2019 or 2017.


Development versions
--------------------
To install the the latest development version of ``sktime``, or earlier versions, the sequence of steps is as follows:

Step 1 - ``git`` clone the ``sktime`` repository, the latest version or an earlier version.
Step 2 - ensure build requirements are satisfied
Step 3 - ``pip`` install the package from a ``git`` clone, with the ``editable`` parameter.

Detail instructions for all steps are given below.
For brevity, we discuss steps 1 and 3 first; step 2 is discussed at the end, as it will depend on the operating system.

Step 1 - git cloning the repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``sktime`` repository should be cloned to a local directory, using a graphical user interface, or the command line.

Using the ``git`` command line, the sequence of commands to install the latest version is as follows:

1. Clone the repository: :code:`git clone https://github.com/alan-turing-institute/sktime.git`
2. Move into the root directory of the local clone: :code:`cd sktime`
3. Make sure you are on the main branch: :code:`git checkout main`
4. Make sure your local version is up-to-date: :code:`git pull`

To build a previous version, replace line 3 with:

.. code-block:: bash

    git checkout <VERSION>

This will checkout the code for the version ``<VERSION>``, where ``<VERSION>`` is a valid version string.
Valid version strings are the repository's ``git`` tags, which can be inspected by running ``git tag``.

You can also `download <https://github.com/alan-turing-institute/sktime/releases>`_ a zip archive of the version from GitHub.


Step 2 - satisfying build requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before carrying out step 3, the ``sktime`` build requirements need to be satisfied.
Details for this differ by operating system, and can be found in the `sktime build requirements`_ section below.

Typically, the set-up steps needs to be carried out only once per system.
That is, the steps usually do not need to be followed again on the same system
when installing an ``sktime`` development version for the second or third time.
Similarly, the advanced developer set-up requires the same build requirements,
so typically no additional steps are required if the advanced developer set-up has already been completed.


Step 3 - building sktime from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a static install of ``sktime`` from source, navigate to the local clone's root directory and type:

.. code-block:: bash

    pip install .

Alternatively, the ``.`` may be replaced with a full or relative path to the root directory.

For a developer install that updates the package each time the local source code is changed, install ``sktime`` in editable mode, via:

.. code-block:: bash

    pip install --editable .

This allows editing and extending the code in-place. See also
`the pip reference on editable installs <https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`_).

.. note::

    You will have to re-run:

    .. code-block:: bash

        pip install --editable .

    every time the source code of a compiled extension is changed (for
    instance when switching branches or pulling changes from upstream).
    Compiled extensions are Cython files (ending in `.pyx` or `.pxd`).

Building binary packages and installers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``.whl`` package and ``.exe`` installers can be built with:

.. code-block:: bash

    pip install wheel
    python setup.py bdist_wheel

The resulting packages are generated in the ``dist/`` folder.


sktime build requirements
-------------------------

This section outlines the ``sktime`` build requirements. These are required for:

* installing ``sktime`` from source, e.g., development versions
* the advanced developer set-up

Build requirements summary
~~~~~~~~~~~~~~~~~~~~~~~~~~

The core build requirement for ``sktime`` are:

- Cython >= 0.28.5 (available through :code:`pip install cython`)
- OpenMP and a working C compiler (see below for instructions)

.. note::

   It is possible to build sktime without OpenMP support by setting the
   ``SKTIME_NO_OPENMP`` environment variable (before cythonization). This is
   not recommended since it will force some estimators to run in sequential
   mode and their ``n_jobs`` parameter will be ignored.

For the advanced developer set-up which includes tests and documentation,
see the advanced developer documentation in :ref:`contributing`.

The following sections describe how to satisfy the build requirements, by operating system.
We currently support:

- `FreeBSD`_
- `Mac OSX`_
- `Unix-like OS`_
- `Windows 8.1 and higher`_


FreeBSD
~~~~~~~

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

For the upcoming FreeBSD 12.1 and 11.3 versions, OpenMP will be included in
the base system and these steps will not be necessary.


Mac OSX
~~~~~~~

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

Troubleshooting - Mac OSX build requirements
""""""""""""""""""""""""""""""""""""""""""""

After installing the release version following the installation steps above and running a ``pytest`` command, some contributors received the error message below:

``E   ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject``

A possible solution to the problem is reinstalling your C compiler. If it is gcc, run ``brew reinstall gcc`` and then ``pip install -e .``.
This should be followed by installing the OpenMP library and setting the environment variables using the same commands again as in the section above.

If you found another solution to the problem, please kindly consider contributing to this section.

Unix-like OS
~~~~~~~~~~~~

Installing from source without conda requires you to have installed the
sktime runtime dependencies, Python development headers and a working
C/C++ compiler. Under Debian-based operating systems, which include Ubuntu:

.. code-block:: bash

    sudo apt-get install build-essential python3-dev python3-setuptools \
                     python3-pip

and then:

.. code-block:: bash

    pip3 install numpy scipy cython

When precompiled wheels are not available for your architecture, you can
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


Windows 8.1 and higher
~~~~~~~~~~~~~~~~~~~~~~

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

Some users have experienced issues when installing NumPy, particularly version 1.19.4. Note that a recent Windows update may affect compilation using Visual Studio (see `Windows update issue <https://developercommunity.visualstudio.com/content/problem/1207405/fmod-after-an-update-to-windows-2004-is-causing-a.html>`_).

If you run into a problem installing the development version and are using Anaconda, try:

1. Install Anaconda
2. Create new environment: :code:`conda create -n sktime-dev python=3.8`
3. Activate environment: :code:`conda activate sktime-dev`
4. Install NumPy (pinned to 1.19.3) from pip: :code:`pip install numpy==1.19.3`
5. Install requirements: :code:`pip install -r build_tools/requirements.txt`
6. Follow the instructions above to point to "vcvarsall.bat"
7. Run :code:`pip install --verbose --no-build-isolation --editable .`

In step 5, you may optionally install the packages in build_tools/requirements.txt that are available from Anaconda's default channels or `Conda-Forge <https://anaconda.org/conda-forge>`_ via Conda. Any remaining packages can be added via pip.

.. note::

    It is possible to use `MinGW <http://www.mingw.org>`_ (a port of GCC to Windows
    OS) as an alternative to MSVC for 32-bit Python. Not that extensions built with
    mingw32 can be redistributed as reusable packages as they depend on GCC runtime
    libraries typically not installed on end-users environment.

    To force the use of a particular compiler, pass the ``--compiler`` flag to the
    build step:

    .. code-block:: bash

        python setup.py build --compiler=my_compiler install

    where ``my_compiler`` should be one of ``mingw32`` or ``msvc``.


.. note::

    Another option under Windows is to follow the instructions for `Unix-like OS`_, using the Windows Subsystem for Linux (WSL).
    For installing WSL, follow the instructions `here <https://docs.microsoft.com/en-us/windows/wsl/install-win10#step-2---check-requirements-for-running-wsl-2>`_.


References
----------

The installation instruction are adapted from scikit-learn's advanced `installation instructions <https://scikit-learn.org/stable/developers/advanced_installation.html>`_.
