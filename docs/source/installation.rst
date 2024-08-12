.. _installation:

Installation
============

``sktime`` currently supports:

* Python versions 3.8, 3.9, 3.10, 3.11, and 3.12.
* Operating systems Mac OS X, Unix-like OS, Windows 8.1 and higher

See here for a `full list of precompiled wheels available on PyPI <https://pypi.org/simple/sktime/>`_.

.. contents::
   :local:

For frequent issues with installation, consult the `Release versions - troubleshooting`_ section.

There are three different installation types:

* Installing sktime releases
* Installing the latest sktime development version
* For developers of sktime and 3rd party extensions: Developer setup

Each of these three setups are explained below.

Release versions
----------------

Installing sktime from PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``sktime`` releases are available via `PyPI <https://pypi.org/project/sktime/>`_. To install
``sktime`` with core dependencies, excluding soft dependencies, via ``pip`` type:

.. code-block:: bash

    pip install sktime


To install ``sktime`` with maximum dependencies, including soft dependencies, install with the ``all_extras`` modifier:

.. code-block:: bash

    pip install sktime[all_extras]

``sktime`` also comes with dependency sets specific to learning task, i.e., estimator scitype.
These are curated selections of the most common soft dependencies for the respective learning task.
The available dependency sets are of the same names as the respective modules:
``forecasting``, ``transformations``, ``classification``, ``regression``, ``clustering``, ``param_est``,
``networks``, ``annotation``, ``alignment``.

.. warning::

    Some of the soft dependencies included in ``all_extras`` and the curated soft dependency sets do not work on mac ARM-based processors, such
    as M1, M2, M1Pro, M1Max or M1Ultra. This may cause an error during installation. Mode details can be found in the :ref:`troubleshooting section<Dependency error on mac ARM>` below.

.. warning::
    The soft dependencies with ``all_extras`` are only necessary to have all estimators available, or to run all tests.
    However, this slows down the downloads, and multiples test time.
    For most user or developer scenarios, downloading ``all_extras`` will
    not be necessary. If you are unsure, install ``sktime`` with core dependencies, and install soft dependencies as needed.
    Alternatively, install dependency sets specific to learning task, see above.

Installing sktime from conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``sktime`` releases are available via ``conda`` from ``conda-forge``.
To install ``sktime`` with core dependencies, excluding soft dependencies via ``conda`` type:

.. code-block:: bash

    conda install -c conda-forge sktime


To install ``sktime`` with maximum dependencies, including soft dependencies, install with the ``all-extras`` recipe:

.. code-block:: bash

    conda install -c conda-forge sktime-all-extras

Note: not all soft dependencies of ``sktime`` are also available on ``conda-forge``,
``sktime-all-extras`` includes only the soft dependencies that are available on ``conda-forge``.
The other soft dependencies can be installed via ``pip``, after ``conda install pip``.


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

    git clone https://github.com/sktime/sktime.git
    cd sktime
    git checkout main
    git pull


To build a previous version, replace line 3 with:

.. code-block:: bash

    git checkout <VERSION>

This will checkout the code for the version ``<VERSION>``, where ``<VERSION>`` is a valid version string.
Valid version strings are the repository's ``git`` tags, which can be inspected by running ``git tag``.

You can also `download <https://github.com/sktime/sktime/releases>`_ a zip archive of the version from GitHub.


Step 2 - building sktime from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build and install ``sktime`` from source, navigate to the local clone's root directory and type:

.. code-block:: bash

    pip install .

Alternatively, the ``.`` may be replaced with a full or relative path to the root directory.

For a developer install that updates the package each time the local source code is changed, install ``sktime`` in editable mode, via:

.. code-block:: bash

    pip install --editable '.[dev]'

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

Contributor or 3rd party extension developer setup
--------------------------------------------------

1. Follow the Git workflow: Fork and clone the repository as described in [Git and GitHub workflow](https://www.sktime.net/en/stable/developer_guide/git_workflow.html)

2. Set up a new virtual environment. Our instructions will go through the commands to set up a ``conda`` environment which is recommended for sktime development.
This relies on an `anaconda installation <https://www.anaconda.com/products/individual#windows>`_. The process will be similar for ``venv`` or other virtual environment managers.

In the ``anaconda prompt`` terminal:

3. Navigate to your local sktime folder, :code:`cd sktime` or similar

4. Create a new environment with a supported python version: :code:`conda create -n sktime-dev python=3.8` (or :code:`python=3.11` etc)

   .. warning::
       If you already have an environment called "sktime-dev" from a previous attempt you will first need to remove this.

5. Activate the environment: :code:`conda activate sktime-dev`

6. Build an editable version of sktime.
In order to install only the dev dependencies, :code:`pip install -e .[dev]`
If you also want to install soft dependencies, install them individually, after the above,
or instead use: :code:`pip install -e .[all_extras,dev]` to install all of them.

    .. note::

        If this step results in a "no matches found" error, it may be due to how your shell handles special characters.

        - Possible solution: use quotation marks:

            .. code-block:: bash

                pip install -e ."[dev]"

7. If everything has worked you should see message "successfully installed sktime"

Some users have experienced issues when installing NumPy, particularly version 1.19.4.



.. note::

    Another option under Windows is to follow the instructions for `Unix-like OS`_, using the Windows Subsystem for Linux (WSL).
    For installing WSL, follow the instructions `here <https://docs.microsoft.com/en-us/windows/wsl/install-win10#step-2---check-requirements-for-running-wsl-2>`_.

Troubleshooting
---------------

Module not found
~~~~~~~~~~~~~~~~

The most frequent reason for *module not found* errors is installing ``sktime`` with
minimum dependencies and using an estimator which interfaces a package that has not
been installed in the environment. To resolve this, install the missing package, or
install ``sktime`` with maximum dependencies (see above).

ImportError
~~~~~~~~~~~
Import errors are often caused by an improperly linked virtual environment.  Make sure that
your environment is activated and linked to whatever IDE you are using.  If you are using Jupyter
Notebooks, follow `these instructions <https://janakiev.com/blog/jupyter-virtual-envs/>`_ for
adding your virtual environment as a new kernel for your notebook.

Installing ``all_extras`` on mac with ARM processor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you are using a mac with an ARM processor, you may encounter an error when installing
``sktime[all_extras]``.  This is due to the fact that some libraries included in ``all_extras``
are not compatible with ARM-based processors.

The workaround is not to install some of the packages in ``all_extras`` and install ARM compatible
replacements for others:

* Do not install the following packages:
    * ``esig``
    * ``prophet``
    * ``tsfresh``
    * ``tslearn``
* Replace ``tensorflow`` package with the following packages:
    * ``tensorflow-macos``
    * ``tensorflow-metal`` (optional)

Also, ARM-based processors have issues when installing packages distributed as source distributions
instead of Python wheels. To avoid this issue when installing a package you can try installing it
through conda or use a prior version of the package that was distributed as a wheel.

Other Startup Resources
-----------------------

Virtual environments
~~~~~~~~~~~~~~~~~~~~

Two good options for virtual environment managers are:

* `conda <https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/>`_ (many sktime community members us this)
* `venv <https://realpython.com/python-virtual-environments-a-primer/>`_ (also quite good!).

Be sure to link your new virtual environment as the python kernel in whatever IDE you are using.  You can find the instructions for doing so
in VScode `here <https://code.visualstudio.com/docs/python/environments>`_.

References
----------

The installation instruction are adapted from scikit-learn's advanced `installation instructions <https://scikit-learn.org/stable/developers/advanced_installation.html>`_.
