.. _installation:

Installation
============

``sktime`` currently supports:

* Python versions 3.7, 3.8 and 3.9
* Operating systems Mac OS X, Unix-like OS, Windows 8.1 and higher

See here for a `full list of precompiled wheels available on PyPI <https://pypi.org/simple/sktime/>`_.

We appreciate community contributions towards compatibility with python 3.10, or other operating systems.

.. contents::
   :local:

Release versions
----------------

For frequent issues with installation, consult the `Release versions - troubleshooting`_ section.

Installing sktime from PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``sktime`` releases are available via `PyPI <https://pypi.org/project/sktime/>`_. To install
``sktime`` with core dependencies, excluding soft dependencies, via ``pip`` type:

.. code-block:: bash

    pip install sktime


To install ``sktime`` with maximum dependencies, including soft dependencies, install with the ``all_extras`` modifier:

.. code-block:: bash

    pip install sktime[all_extras]


Installing sktime from conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``sktime`` releases are available via ``conda`` from ``conda-forge``.
To install ``sktime`` with core dependencies, excluding soft dependencies via ``conda`` type:

.. code-block:: bash

    conda install -c conda-forge sktime


To install ``sktime`` with maximum dependencies, including soft dependencies, install with the ``all-extras`` recipe:

.. code-block:: bash

    conda install -c conda-forge sktime-all-extras

Note: currently this does not include the dependency ``catch-22``.
As this package is not available on ``conda-forge``, it must be installed via ``pip`` if desired.
Contributions to remedy this situation are appreciated.


Release versions - troubleshooting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Module not found
""""""""""""""""

The most frequent reason for *module not found* errors is installing ``sktime`` with
minimum dependencies and using an estimator which interfaces a package that has not
been installed in the environment. To resolve this, install the missing package, or
install ``sktime`` with maximum dependencies (see above).

Development versions
--------------------
To install the latest development version of ``sktime``, or earlier versions, the sequence of steps is as follows:

Step 1 - ``git`` clone the ``sktime`` repository, the latest version or an earlier version.
Step 2 - ensure build requirements are satisfied
Step 3 - ``pip`` install the package from a ``git`` clone, with the ``editable`` parameter.

Detail instructions for all steps are given below.
For brevity, we discuss steps 1 and 3 first; step 2 is discussed at the end, as it will depend on the operating system.

Step 1 - clone the git repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``sktime`` repository should be cloned to a local directory, using a graphical user interface, or the command line.

Using the ``git`` command line, the sequence of commands to install the latest version is as follows:

.. code-block:: bash

    git clone https://github.com/alan-turing-institute/sktime.git
    cd sktime
    git checkout main
    git pull


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

Step 3 - building sktime from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build and install ``sktime`` from source, navigate to the local clone's root directory and type:

.. code-block:: bash

    pip install .

Alternatively, the ``.`` may be replaced with a full or relative path to the root directory.

For a developer install that updates the package each time the local source code is changed, install ``sktime`` in editable mode, via:

.. code-block:: bash

    pip install --editable .[dev]

This allows editing and extending the code in-place. See also
`pip reference on editable installs <https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`_).

.. note::

    You will have to re-run:

    .. code-block:: bash

        pip install --editable .

    every time the source code of a compiled extension is changed (for
    instance when switching branches or pulling changes from upstream).

Building binary packages and installers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``.whl`` package and ``.exe`` installers can be built with:

.. code-block:: bash

    pip install build
    python -m build --wheel

The resulting packages are generated in the ``dist/`` folder.


sktime build requirements
-------------------------

This section outlines the ``sktime`` build requirements. These are required for:

* installing ``sktime`` from source, e.g., development versions
* the advanced developer set-up


Setting up a development environment
""""""""""""""""""""""""""""""""""""
You now need to set up a new python virtual environment. Our instructions will go through the commands to set up a ``conda`` environment which is recommended for sktime development.
This relies on an `anaconda installation <https://www.anaconda.com/products/individual#windows>`_. The process will be similar for ``venv`` or other virtual environment managers.

In the ``anaconda prompt`` terminal:

1. Navigate to your local sktime folder :code:`cd sktime`

2. Create new environment with python 3.8: :code:`conda create -n sktime-dev python=3.8`

   .. warning::
       If you already have an environment called "sktime-dev" from a previous attempt you will first need to remove this

3. Activate the environment: :code:`conda activate sktime-dev`

4. Build an editable version of sktime :code:`pip install -e .[all_extras,dev]`

5. If everything has worked you should see message "successfully installed sktime"

Some users have experienced issues when installing NumPy, particularly version 1.19.4.

.. note::

    If step 4. results in a "no matches found" error, it may be due to how your shell handles special characters.

    - Possible solution: use quotation marks:

        .. code-block:: bash

            pip install -e ."[all_extras,dev]"

.. note::

    Another option under Windows is to follow the instructions for `Unix-like OS`_, using the Windows Subsystem for Linux (WSL).
    For installing WSL, follow the instructions `here <https://docs.microsoft.com/en-us/windows/wsl/install-win10#step-2---check-requirements-for-running-wsl-2>`_.


References
----------

The installation instruction are adapted from scikit-learn's advanced `installation instructions <https://scikit-learn.org/stable/developers/advanced_installation.html>`_.
