.. _contributing:

=================
How to contribute
=================

.. toctree::
   :maxdepth: 1
   :hidden:

   developer_guide
   installation
   contributing/enhancement_proposals
   contributing/reporting_bugs
   contributing/reviewer_guide

Welcome to sktime's contributing guide!

We value all kinds of contributions - not just code.
For a detailed overview of current and future work, check out our :ref:`roadmap`.

We are particularly motivated to support new contributors
and people who are looking to learn and develop their skills.

-  **Good-first issues.** Check out our `good-first
   issues <https://github.com/alan-turing-institute/sktime/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22>`_.
   If you are interested in one of them, simply comment on the issue!
-  **Mentoring.** New to sktime? Apply to sktime’s own
   :ref:`mentoring program <mentoring>`!
-  **The Turing Way**. Check out their
   `Guide for Reproducible
   Research <https://the-turing-way.netlify.app/reproducible-research/reproducible-research.html>`_ to get started and learn more about open-source collaboration.

.. _discord: https://discord.com/invite/gqSab2K
.. _gitter: https://gitter.im/sktime/community

.. note::

   If you get stuck, chat with us on `Gitter`_ or `Discord`_.

.. panels::
    :card: + intro-card text-center

    ---

    Developer Guide
    ^^^^^^^^^^^^^^^

    ``sktime`` developer guide.

    +++

    .. link-button:: developer_guide
            :type: ref
            :text: Developer Guide
            :classes: btn-block btn-secondary stretched-link

    ---

    Installation
    ^^^^^^^^^^^^

    ``sktime`` developer installation guide.

    +++

    .. link-button:: installation
            :type: ref
            :text: Developer Installation
            :classes: btn-block btn-secondary stretched-link

    ---

    Enhancement Proposals
    ^^^^^^^^^^^^^^^^^^^^^

    ``sktime`` enhancement proposals.

    +++

    .. link-button:: enhancement_proposals
            :type: ref
            :text: Enhancement Proposals
            :classes: btn-block btn-secondary stretched-link

    ---

    Reporting Bugs
    ^^^^^^^^^^^^^^

    ``sktime`` reporting bugs.

    +++

    .. link-button:: reporting_bugs
            :type: ref
            :text: Reporting Bugs
            :classes: btn-block btn-secondary stretched-link

    ---
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

      pytest sktime/

Test coverage
~~~~~~~~~~~~~

.. _codecov: https://codecov.io
.. _coverage: https://coverage.readthedocs.io/
.. _pytestcov: https://github.com/pytest-dev/pytest-cov

We use `coverage`_, the `pytest-cov <pytestcov>`_ plugin, and `codecov`_ for test coverage.

API design
----------

The general design approach of sktime is described in the
paper `“Designing Machine Learning Toolboxes: Concepts, Principles and
Patterns” <https://arxiv.org/abs/2101.04938>`__.

.. note::

   This is a first draft of the paper.
   Feedback and improvement suggestions are very welcome!

Documentation
-------------

.. _sphinx: https://www.sphinx-doc.org/
.. _readthedocs: https://readthedocs.org/projects/sktime/

We use `sphinx`_ to build our documentation and `readthedocs`_ to host it.
You can find our latest documentation `here <https://www.sktime.org/en/latest/>`_.

The source files can be found
in `docs/source/ <https://github.com/alan-turing-institute/sktime/tree/main/docs/source>`_.
The main configuration file for sphinx is
`conf.py <https://github.com/alan-turing-institute/sktime/blob/main/docs/source/conf.py>`__
and the main page is
`index.rst <https://github.com/alan-turing-institute/sktime/blob/main/docs/source/index.rst>`__.
To add new pages, you need to add a new ``.rst`` file and include it in
the ``index.rst`` file.

To build the documentation locally, you need to install a few extra
dependencies listed in
`pyproject.toml <https://github.com/alan-turing-institute/sktime/blob/main/pyproject.toml>`__.

1. To install extra dependencies from the root directory, run:

   .. code:: bash

      pip install .[docs]

2. To build the website locally, run:

   .. code:: bash

      make docs

You can find the generated files in the ``sktime/docs/_build/`` folder.
To view the website, open ``sktime/docs/_build/html/index.html`` with
your preferred web browser.

Dependencies
------------

We try to keep the number of core dependencies to a minimum and rely on
other packages as soft dependencies when feasible.

.. note::

   A soft dependency is a dependency that is only required to import
   certain modules, but not necessary to use most functionality. A soft
   dependency is not installed automatically when the package is
   installed. Instead, users need to install it manually if they want to
   use a module that requires a soft dependency.

If you add a new dependency or change the version of an existing one,
you need to update the following file:

-  `pyproject.toml <https://github.com/alan-turing-institute/sktime/blob/main/pyproject.toml>`__
   following the `PEP 621 <https://www.python.org/dev/peps/pep-0621/>`_ convention all dependencies
   including build time dependencies and optional dependencies are specified in this file.

If a user is missing a soft dependency, we raise a user-friendly error message.
This is handled through our ``_check_soft_dependencies`` defined
`here <https://github.com/alan-turing-institute/sktime/blob/main/sktime/utils/validation/_dependencies.py>`__.

We use continuous integration tests to check if all soft
dependencies are properly isolated to specific modules.
If you add a new soft dependency, make sure to add it
`here <https://github.com/alan-turing-institute/sktime/blob/main/build_tools/azure/check_soft_dependencies.py>`__
together with the module that depends on it.

Coding style
------------

We follow the `PEP8 <https://www.python.org/dev/peps/pep-0008/>`__
coding guidelines. A good example can be found
`here <https://gist.github.com/nateGeorge/5455d2c57fb33c1ae04706f2dc4fee01>`__.

We use the `pre-commit <#Code-quality-checks>`_ workflow together with
`black <https://black.readthedocs.io/en/stable/>`__ and
`flake8 <https://flake8.pycqa.org/en/latest/>`__ to automatically apply
consistent formatting and check whether your contribution complies with
the PEP8 style.

For docstrings, we use the `numpy docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_, along with sktime specific conventions described in our :ref:`developer_guide`'s :ref:`documentation section <developer_guide_documentation>`.

In addition, we add the following guidelines:

-  Please check out our :ref:`glossary`.
-  Use underscores to separate words in non-class names: ``n_instances``
   rather than ``ninstances``.
-  Avoid multiple statements on one line. Prefer a line return after a
   control flow statement (``if``/``for``).
-  Use absolute imports for references inside sktime.
-  Please don’t use ``import *`` in the source code. It is considered
   harmful by the official Python recommendations. It makes the code
   harder to read as the origin of symbols is no longer explicitly
   referenced, but most important, it prevents using a static analysis
   tool like pyflakes to automatically find bugs.

.. _infrastructure::

Infrastructure
--------------

This section gives an overview of the infrastructure and continuous
integration services we use.

+---------------+-----------------------+-------------------------------------+
| Platform      | Operation             | Configuration                       |
+===============+=======================+=====================================+
| `GitHub       | Build/test/           | `.github/workflows/ <https://gi     |
| Act           | distribute            | thub.com/alan-turing-institute/skti |
| ions <https:/ | on Linux, MacOS and   | me/blob/main/.github/workflows/>`__ |
| /docs.github. | Windows,              |                                     |
| com/en/free-p | run code quality      |                                     |
| ro-team@lates | checks                |                                     |
| t/actions>`__ |                       |                                     |
+---------------+-----------------------+-------------------------------------+
| `Read the     | Build/deploy          | `.readthed                          |
| Docs <h       | documentation         | ocs.yml <https://github.com/alan-tu |
| ttps://readth |                       | ring-institute/sktime/blob/main/.gi |
| edocs.org>`__ |                       | thub/workflows/code-quality.yml>`__ |
+---------------+-----------------------+-------------------------------------+
| `Codec        | Test coverage         | `.codecov.yml <https                |
| ov <https://c |                       | ://github.com/alan-turing-institute |
| odecov.io>`__ |                       | /sktime/blob/main/.codecov.yml>`__, |
|               |                       | `.coveragerc <htt                   |
|               |                       | ps://github.com/alan-turing-institu |
|               |                       | te/sktime/blob/main/.coveragerc>`__ |
+---------------+-----------------------+-------------------------------------+

Additional scripts used for building, unit testing and distribution can
be found in
`build_tools/ <https://github.com/alan-turing-institute/sktime/tree/main/build_tools>`__.

Releases
--------

This section is for core developers. To make a new release, you need
push-to-write access on our main branch.

sktime is not a pure Python package and depends on some non-Python code
including Cython and C. We distribute compiled files, called wheels, for
different operating systems and Python versions.

.. note::

    Reviewer Guide
    ^^^^^^^^^^^^^^

    ``sktime`` reviewer guide.

    +++

    .. link-button:: reviewer_guide
            :type: ref
            :text: Reporting Bugs
            :classes: btn-block btn-secondary stretched-link




Acknowledging contributions
---------------------------

We follow the `all-contributors
specification <https://allcontributors.org>`_ and recognize various
types of contributions.
Take a look at our past and current
`contributors <https://github.com/alan-turing-institute/sktime/blob/main/CONTRIBUTORS.md>`_!

If you are a new contributor, make sure we add you to our list of
contributors.
All contributions are recorded in
`.all-contributorsrc <https://github.com/alan-turing-institute/sktime/blob/main/.all-contributorsrc>`_.

.. note::

   If we have missed anything, please `raise an issue <https://github.com/alan-turing-institute/sktime/issues/new/choose>`_ or chat with us on `Gitter`_ or `Discord`_.
