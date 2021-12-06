.. _continious_integration:

Continious integration
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
------------

We use `pytest <https://docs.pytest.org/en/latest/>`__ for unit testing.
To check if your code passes all tests locally, you need to install the
development version of sktime and all extra dependencies.

1. Install all extra requirements from the root directory of sktime:

   .. code:: bash

      pip install -r build_tools/requirements.txt

2. Install the development version of sktime:

   .. code:: bash

      pip install -e .

   This installs an editable `development
   version <https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`__
   of sktime which will include the changes you make.

.. note::

   For trouble shooting on different operating systems, please see our detailed
   `installation instructions <https://www.sktime.org/en/latest/installation.html>`__.

3. To run all unit tests, run:

   .. code:: bash

      pytest sktime/

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

+---------------+----------------+-------------------------------------+
| Platform      | Operation      | Configuration                       |
+===============+================+=====================================+
| `Appveyor     | Build/t        | `.appveyor.yml <https               |
|  <https://ci. | est/distribute | ://github.com/alan-turing-institute |
| appveyor.com/ | on Windows     | /sktime/blob/main/.appveyor.yml>`__ |
| project/mloni |                |                                     |
| ng/sktime>`__ |                |                                     |
+---------------+----------------+-------------------------------------+
| `Azure        | Build/t        | `azure-pipelines.yml <https://git   |
| Pipelines <h  | est/distribute | hub.com/alan-turing-institute/sktim |
| ttps://dev.az | on Linux       | e/blob/main/azure-pipelines.yml>`__ |
| ure.com/mloni | (`manylin      |                                     |
| ng/sktime>`__ | ux <https://gi |                                     |
|               | thub.com/pypa/ |                                     |
|               | manylinux>`__) |                                     |
+---------------+----------------+-------------------------------------+
| `GitHub       | Build/t        | `.github/workflows/ <https://gi     |
| Act           | est/distribute | thub.com/alan-turing-institute/skti |
| ions <https:/ | on MacOS; Code | me/blob/main/.github/workflows/>`__ |
| /docs.github. | quality checks |                                     |
| com/en/free-p |                |                                     |
| ro-team@lates |                |                                     |
| t/actions>`__ |                |                                     |
+---------------+----------------+-------------------------------------+
| `Read the     | Build/deploy   | `.readthed                          |
| Docs <h       | documentation  | ocs.yml <https://github.com/alan-tu |
| ttps://readth |                | ring-institute/sktime/blob/main/.gi |
| edocs.org>`__ |                | thub/workflows/code-quality.yml>`__ |
+---------------+----------------+-------------------------------------+
| `Codec        | Test coverage  | `.codecov.yml <https                |
| ov <https://c |                | ://github.com/alan-turing-institute |
| odecov.io>`__ |                | /sktime/blob/main/.codecov.yml>`__, |
|               |                | `.coveragerc <htt                   |
|               |                | ps://github.com/alan-turing-institu |
|               |                | te/sktime/blob/main/.coveragerc>`__ |
+---------------+----------------+-------------------------------------+

Additional scripts used for building, unit testing and distribution can
be found in
`build_tools/ <https://github.com/alan-turing-institute/sktime/tree/main/build_tools>`__.
