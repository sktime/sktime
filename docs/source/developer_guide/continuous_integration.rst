.. _continuous_integration:

Testing and continuous integration
==================================

This page gives a summary of:

* testing for contributors - code style and local testing
* testing for maintainers - continuous integration

If you are a contributor or developer, ensure that you have set
up your developer environment, and installed a
:doc:`development version </installation>`
of ``sktime``.

``sktime`` use continuous integration (CI) services on GitHub to automatically check
if new pull requests do not break anything and meet code quality
standards such as a common `coding style <#Coding-style>`__.

.. contents::
   :local:

Local Testing
-------------

If you contribute to ``sktime``, the below gives you a guide on how to
test your code locally before you make a pull request.

We recommend:

* set up code quality checks in your local dev IDE
* learn how to use the ``check_estimator`` utility for estimators and ``sktime`` objects
* advanced contributions: ensure you can run the full ``pytest`` test suite locally, via your dev IDE, console, or docker


Prerequisite: local python environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Local testing requires a developer setup
of ``sktime``. If you do not have this already, we recommend you to create a new
virtual environment and install an editable development version of ``sktime``.

For a full guide on how to obtain a developer setup, read the following link `Full developer setup for contributors and extension developers
<https://www.sktime.net/en/latest/installation.html#full-developer-setup-for-contributors-and-extension-developers>`__ .

If you have an environment setup already, ensure you have an editable development version of sktime with developer dependencies.
To install, if not already installed:

   .. code:: bash

      pip install -e .[dev]

   This installs an editable `development
   version <https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`__
   of sktime which will include the changes you make.

.. note::

   For trouble shooting on different operating systems, please see our detailed
   :doc:`installation instructions </installation>`.

Code quality checks
~~~~~~~~~~~~~~~~~~~

.. _pre-commit: https://pre-commit.com

We use `pre-commit`_ for code quality checks (a process we also refer to as "linting" checks).

We recommend that you also set this up locally as it will ensure that you never run into code quality errors when you make your first PR!
These checks run automatically before you make a new commit.
To setup, simply navigate to the sktime folder and install our pre-commit configuration:

   .. code:: bash

      pre-commit install

pre-commit should now automatically run anything you make a commit! Please let us know if you encounter any issues getting this setup.

For a detailed guide on code quality and linting for developers, see :ref:`coding_standards`.

Testing objects via ``check_estimator``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For contributions that are localized to estimators or objects, the ``check_estimator``
utility can be used.

For this, follow the instructions in the
:doc:`estimator development guide </developer_guide/add_estimators>`

Full test suite runs via ``pytest``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The full test suite can be run locally via `pytest <https://docs.pytest.org/en/latest/>`__,
which ``sktime`` uses for its testing framework.

To run all tests via the console via `make <https://www.gnu.org/software/make/>`_ (only unix based OS):

   .. code:: bash

      make test

or, from a console with ``pytest`` in the path, from the repository root:

   .. code:: bash

      pytest ./sktime

Further, developer IDEs such as pycharm or vs code will automatically recognize
the tests via ``pytest``, refer to the documentation of the IDEs for testing
via the embedded graphical user interface.

Running docstring examples via ``doctest``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``sktime``'s Python modules are equipped with docstrings that include examples
demonstrating the usage of specific classes within each module.

Docstring examples can be executed in bulk using ``doctest``,
to ensure that this is indeed the case.

To run doctest on all the files with ``pytest``,
navigate to the root directory and execute the following command:

  .. code:: bash

      pytest --doctest-modules

To run doctest on all the files without ``pytest``,
navigate to the root directory and execute the following command:

(for ``UNIX`` based OS)
  .. code:: bash

      find . -name "*.py" -print0 | xargs -0 python -m doctest -v -o=ELLIPSIS

(for windows)
  .. code:: bash

      for /r %G in (*.py) do python -m doctest -v "%G" -o=ELLIPSIS

To run doctest on a specific module, navigate to the directory where the
module is located and execute the following command:

   .. code:: bash

      python -m doctest -v -o=ELLIPSIS {filename}


Executing this command will display the test results for all the docstrings
contained within the module.

Alternative: dockerized testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We also provide an option to execute the test suite via ``docker`` containers.
This requires a local docker installation.
To install, follow the instructions `here <https://docs.docker.com/desktop/>`_.

The docker images for the tests are in the folder ``build_tools/docker``,
with the image of name ``PYTHON_VERSION`` based on the following python versions:

+----------------+----------------+
| Python version | PYTHON_VERSION |
+================+================+
|     3.8     |      py38         |
+----------------+----------------+
|     3.9     |      py39         |
+----------------+----------------+
|     3.10    |      py310        |
+----------------+----------------+
|     3.11    |      py311        |
+----------------+----------------+
|     3.12    |      py312        |
+----------------+----------------+

The dockerized tests can be also executed via `make <https://www.gnu.org/software/make/>`_,
via the command ``make dockertest PYTHON_VERSION=<python version>``.
The ``PYTHON_VERSION`` argument specifies the python version and is the same string as in the table above.
For example, to execute the tests in the Python version ``3.8``,
use ``make dockertest PYTHON_VERSION=py38``.


Continuous integration
----------------------

Infrastructure overview
~~~~~~~~~~~~~~~~~~~~~~~

This section gives an overview of the infrastructure and continuous
integration services we use.

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Platforms
     - Operation
     - Configuration
   * - `GitHub Action <https://docs.github.com/en/free-pro-team@latest/actions>`__
     - Build/test/distribute on Linux, MacOS and Windows, run code quality checks
     - `.github/workflows/ <https://github.com/sktime/sktime/blob/main/.github/workflows/>`__
   * - `Read the Docs <https://readthedocs.org>`__
     - Build/deploy documentation
     - `.readthedocs.yml <https://github.com/alan-turing-institute/sktime/blob/main/.github/workflows/code-quality.yml>`__
   * - `Codecov <https://codecov.io>`__
     - Test coverage
     - `.codecov.yml <https://github.com/sktime/sktime/blob/main/.codecov.yml>`__, `.coveragerc <https://github.com/alan-turing-institute/sktime/blob/main/.coveragerc>`__

Additional scripts used for building, unit testing and distribution can
be found in
`build_tools/ <https://github.com/sktime/sktime/tree/main/build_tools>`__.

Test coverage
~~~~~~~~~~~~~

.. _codecov: https://codecov.io
.. _coverage: https://coverage.readthedocs.io/
.. _pytest-cov: https://github.com/pytest-dev/pytest-cov

We use `coverage`_, the `pytest-cov`_ plugin, and `codecov`_ for test coverage.
