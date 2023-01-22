.. _continuous_integration:

Continuous integration
======================

We use continuous integration services on GitHub to automatically check
if new pull requests do not break anything and meet code quality
standards such as a common `coding style <#Coding-style>`__.
Before setting up Continuous Integration, be sure that you have set
up your developer environment, and installed a
`developement version <https://www.sktime.org/en/stable/installation.html>`__
 of sktime.

.. contents::
   :local:

Code quality checks
-------------------

.. _pre-commit: https://pre-commit.com

We use `pre-commit`_ for code quality checks (a process we also refer to as "linting" checks).

We recommend that you also set this up locally as it will ensure that you never run into code quality errors when you make your first PR!
These checks run automatically before you make a new commit.
To setup, simply navigate to the sktime folder and install our pre-commit configuration:

::
   pre-commit install

pre-commit should now automatically run anything you make a commit! Please let us know if you encounter any issues getting this setup.

For a detailed guide on code quality and linting for developers, see :ref:`coding_standards`.

Unit testing
~~~~~~~~~~~~

We use `pytest <https://docs.pytest.org/en/latest/>`__ for unit testing.

To check if your code passes all tests locally, you need to install the
development version of sktime and all extra dependencies.

1. Install the development version of sktime with developer dependencies:

   .. code:: bash

      pip install -e .[dev]

   This installs an editable `development
   version <https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`__
   of sktime which will include the changes you make.

.. note::

   For trouble shooting on different operating systems, please see our detailed
   `installation instructions <https://www.sktime.org/en/latest/installation.html>`__.

2. To run all unit tests, run:

   .. code:: bash

      make test

or if you don't have `make <https://www.gnu.org/software/make/>`_ installed:

   .. code:: bash

      pytest ./sktime

Test coverage
-------------

.. _codecov: https://codecov.io
.. _coverage: https://coverage.readthedocs.io/
.. _pytest-cov: https://github.com/pytest-dev/pytest-cov

We use `coverage`_, the `pytest-cov`_ plugin, and `codecov`_ for test coverage.

Infrastructure
--------------

This section gives an overview of the infrastructure and continuous
integration services we use.

+---------------+-----------------------+-------------------------------------+
| Platform      | Operation             | Configuration                       |
+===============+=======================+=====================================+
| `GitHub       | Build/test/           | `.github/workflows/ <https://gi     |
| Actions       | distribute            | thub.com/sktime/skti |
| <https:/      | on Linux, MacOS and   | me/blob/main/.github/workflows/>`__ |
| /docs.github. | Windows,              |                                     |
| com/en/free-p | run code quality      |                                     |
| ro-team@lates | checks                |                                     |
| t/actions>`__ |                       |                                     |
+---------------+-----------------------+-------------------------------------+
| `Read the     | Build/deploy          | `.readthedocs.yml                   |
| Docs <h       | documentation         | <https://github.com/alan-tu         |
| ttps://readth |                       | ring-institute/sktime/blob/main/.gi |
| edocs.org>`__ |                       | thub/workflows/code-quality.yml>`__ |
+---------------+-----------------------+-------------------------------------+
| `Codecov      | Test coverage         | `.codecov.yml <https                |
| <https://c    |                       | ://github.com/sktime |
| odecov.io>`__ |                       | /sktime/blob/main/.codecov.yml>`__, |
|               |                       | `.coveragerc <htt                   |
|               |                       | ps://github.com/alan-turing-institu |
|               |                       | te/sktime/blob/main/.coveragerc>`__ |
+---------------+-----------------------+-------------------------------------+

Additional scripts used for building, unit testing and distribution can
be found in
`build_tools/ <https://github.com/sktime/sktime/tree/main/build_tools>`__.
