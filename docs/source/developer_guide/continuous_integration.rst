.. _continuous_integration:

Continuous integration
======================

We use continuous integration services on GitHub to automatically check
if new pull requests do not break anything and meet code quality
standards such as a common `coding style <#Coding-style>`__.

.. contents::
   :local:

Code quality checks
-------------------

.. _precommit: https://pre-commit.com

We use `pre-commit <precommit>`_ for code quality checks.
These checks run automatically before you make a new commit.

To set up pre-commit, follow these steps:

1. Install pre-commit:

.. code:: bash

   pip install pre-commit

2. Set up pre-commit:

.. code:: bash

   pre-commit install

Once installed, pre-commit will automatically run our code quality
checks on the files you changed whenever you make a new commit.

You can find our pre-commit configuration in
`.pre-commit-config.yaml <https://github.com/alan-turing-institute/sktime/blob/main/.pre-commit-config.yaml>`_.
Additional configurations can be found in
`setup.cfg <https://github.com/alan-turing-institute/sktime/blob/main/setup.cfg>`_.

.. note::
   If you want to exclude some line of code from being checked, you can add a ``# noqa`` (no quality assurance) comment at the end of that line.


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
.. _pytestcov: https://github.com/pytest-dev/pytest-cov

We use `coverage`_, the `pytest-cov <pytestcov>`_ plugin, and `codecov`_ for test coverage.

Infrastructure
--------------

This section gives an overview of the infrastructure and continuous
integration services we use.

+---------------+-----------------------+-------------------------------------+
| Platform      | Operation             | Configuration                       |
+===============+=======================+=====================================+
| `GitHub       | Build/test/           | `.github/workflows/ <https://gi     |
| Actions       | distribute            | thub.com/alan-turing-institute/skti |
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
| <https://c    |                       | ://github.com/alan-turing-institute |
| odecov.io>`__ |                       | /sktime/blob/main/.codecov.yml>`__, |
|               |                       | `.coveragerc <htt                   |
|               |                       | ps://github.com/alan-turing-institu |
|               |                       | te/sktime/blob/main/.coveragerc>`__ |
+---------------+-----------------------+-------------------------------------+

Additional scripts used for building, unit testing and distribution can
be found in
`build_tools/ <https://github.com/alan-turing-institute/sktime/tree/main/build_tools>`__.
