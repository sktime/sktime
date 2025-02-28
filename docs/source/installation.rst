.. _installation:

Installation
============

``sktime`` currently supports:

* Python versions 3.8, 3.9, 3.10, 3.11, and 3.12.
* Operating systems Mac OS X, Unix-like OS, Windows 8.1 and higher

See here for a `full list of precompiled wheels available on PyPI <https://pypi.org/simple/sktime/>`_.

.. contents::
   :local:

For frequent issues with installation, consult the `Troubleshooting`_ section.

There are three different installation types, depending on your use case:

* Installing stable ``sktime`` releases - for most users, for production environments
* Installing the latest unstable ``sktime`` development version - for pre-release tests
* For developers of ``sktime`` and 3rd party extensions: Developer setup for extensions and contributions

Each of these three setups are explained below.

Installing release versions
---------------------------

For:

* Most users
* Use in production environments

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
``networks``, ``detection``, ``alignment``.

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


Installing latest unstable development version
----------------------------------------------

For:

* pre-release tests, e.g., early testing of new features
* not for reliable production use
* not for contributors or extenders

This type of ``sktime`` installation obtains a latest static snapshot of the repository.
It is intended for developers that wish to build or test code using a version of the library
that contains the all of the latest and current updates.

.. note::
    For an full editible developer setup, please read the section "Full developer setup for contributors and extension developers" below.

To install the latest version of ``sktime`` directly from the repository,
you can use the ``pip`` package manager to install directly from the GitHub repository:

.. code-block:: bash

    pip install git+https://github.com/sktime/sktime.git


To install from a specific branch, use the following command:

.. code-block:: bash

    pip install git+https://github.com/sktime/sktime.git@<branch_name>

Alternatively, a latest version install can be obtained from a local clone of the repository.

For steps on how to obtain a local clone of the repository, please follow the steps described here:
:ref:`Creating a fork and cloning the repository <Creating a fork and cloning the repository - initial one time setup>`


.. code-block:: bash

    pip install .

Alternatively, the ``.`` may be replaced with a full or relative path to the root directory.


Full developer setup for contributors and extension developers
--------------------------------------------------------------

For whom:

* contributors to the ``sktime`` project
* developers of extensions in closed code bases
* developers of 3rd party extensions released as open source

To develop ``sktime`` locally, or to contribute to the project, you need to set up:

* a local clone of the ``sktime`` repository.
* a virtual environment with an editable install of ``sktime`` and its developer dependencies.

The following steps guide you through the process:

1. Follow the Git workflow: :ref:`Creating a fork and cloning the repository <Creating a fork and cloning the repository - initial one time setup>`

2. Set up a new virtual environment. Our instructions will go through the commands to set up a ``conda`` environment, which tends to be beginner friendly.
The process will be similar for ``venv`` or other virtual environment managers.

.. warning::
    Using ``conda`` via one of the commercial distributions such as Anaconda
    is in general not free for commercial use and may incur significant costs or liabilities.
    Consider using free distributions and channels for package management,
    and be aware of applicable terms and conditions.

In the ``conda`` terminal:

3. Navigate to your local sktime folder, :code:`cd sktime` or similar

4. Create a new environment with a supported python version: :code:`conda create -n sktime-dev python=3.11` (or :code:`python=3.12` etc)

   .. warning::
       If you already have an environment called ``sktime-dev`` from a previous attempt you will first need to remove this.

5. Activate the environment: :code:`conda activate sktime-dev`

6. Build an editable version of sktime.
In order to install only the dev dependencies, :code:`pip install -e ".[dev]"`
If you also want to install soft dependencies, install them individually, after the above,
or instead use: :code:`pip install -e ".[all_extras,dev]"` to install all of them.

7. If everything has worked, you should see message "successfully installed sktime"

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

* `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ (beginner friendly, but may incur license fees for commercial use if using a commercial distribution).
* `venv <https://docs.python.org/3/library/venv.html>`_ (also quite good!).

Be sure to link your new virtual environment as the python kernel in whatever IDE you are using.  You can find the instructions for doing so
in VScode `here <https://code.visualstudio.com/docs/python/environments>`_.

References
----------

The installation instruction are adapted from scikit-learn's advanced `installation instructions <https://scikit-learn.org/stable/developers/advanced_installation.html>`_.
