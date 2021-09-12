.. _contributing:

=================
How to contribute
=================

Welcome to sktime's contributing guide!

sktime is a community-driven project and your help is extremely welcome.

.. contents::
   :local:

How to get started
------------------

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

Reporting bugs
--------------

We use GitHub issues to track all bugs and feature requests; feel free
to open an issue if you have found a bug or wish to see a feature
implemented.

It is recommended to check that your issue complies with the following
rules before submitting:

-  Verify that your issue is not being currently addressed by other
   `issues <https://github.com/alan-turing-institute/sktime/issues>`__
   or `pull
   requests <https://github.com/alan-turing-institute/sktime/pulls>`__.
-  Please ensure all code snippets and error messages are formatted in
   appropriate code blocks. See `Creating and highlighting code
   blocks <https://help.github.com/articles/creating-and-highlighting-code-blocks>`__.
-  Please be specific about what estimators and/or functions are
   involved and the shape of the data, as appropriate; please include a
   `reproducible <https://stackoverflow.com/help/mcve>`__ code snippet
   or link to a `gist <https://gist.github.com>`__. If an exception is
   raised, please provide the traceback.

.. note::

   To find out more about how to take part in sktime’s community, check out our `governance
   guidelines <https://www.sktime.org/en/latest/governance.html>`__.

Where to contribute
-------------------

We value all kinds of contributions - not just code.
For a detailed overview of current and future work, check out our :ref:`roadmap`.

Installation
------------

Please visit our detailed `installation
instructions <https://www.sktime.org/en/latest/installation.html>`__ to
resolve any package issues and dependency errors before they occur in
the following steps. OS specific instruction is available at the prior
link.

Git and GitHub workflow
-----------------------

The preferred workflow for contributing to sktime’s repository is to
fork the `main
repository <https://github.com/alan-turing-institute/sktime/>`__ on
GitHub, clone, and develop on a new branch.

1.  Fork the `project
    repository <https://github.com/alan-turing-institute/sktime>`__ by
    clicking on the 'Fork' button near the top right of the page. This
    creates a copy of the code under your GitHub user account. For more
    details on how to fork a repository see `this
    guide <https://help.github.com/articles/fork-a-repo/>`__.

2.  `Clone <https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository>`__
    your fork of the sktime repo from your GitHub account to your local
    disk:

    .. code:: bash

       git clone git@github.com:<username>/sktime.git
       cd sktime

    where :code:`<username>` is your GitHub username.

3.  Configure and link the remote for your fork to the upstream
    repository:

    .. code:: bash

       git remote -v
       git remote add upstream https://github.com/alan-turing-institute/sktime.git

4.  Verify the new upstream repository you've specified for your fork:

    .. code:: bash

       git remote -v
       > origin    https://github.com/<username>/sktime.git (fetch)
       > origin    https://github.com/<username>/sktime.git (push)
       > upstream  https://github.com/alan-turing-institute/sktime.git (fetch)
       > upstream  https://github.com/alan-turing-institute/sktime.git (push)

5.  `Sync <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork>`_
    the ``main`` branch of your fork with the upstream repository:

    .. code:: bash

       git fetch upstream
       git checkout main
       git merge upstream/main

6.  Create a new ``feature`` branch from the ``main`` branch to hold
    your changes:

    .. code:: bash

       git checkout main
       git checkout -b <feature-branch>

    Always use a ``feature`` branch. It's good practice to never work on
    the ``main`` branch! Name the ``feature`` branch after your
    contribution.

7.  Develop your contribution on your feature branch. Add changed files
    using ``git add`` and then ``git commit`` files to record your
    changes in Git:

    .. code:: bash

       git add <modified_files>
       git commit

8.  When finished, push the changes to your GitHub account with:

    .. code:: bash

       git push --set-upstream origin my-feature-branch

9.  Follow `these
    instructions <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`__
    to create a pull request from your fork. If your work is still work
    in progress, open a draft pull request.

.. note::

    We recommend to open a pull request early, so that other contributors become aware of
    your work and can give you feedback early on.

10. To add more changes, simply repeat steps 7 - 8. Pull requests are
    updated automatically if you push new changes to the same branch.

.. note::

   If any of the above seems like magic to you, look up the `Git documentation <https://git scm.com/documentation>`_.
   If you get stuck, chat with us on `Gitter`_ or `Discord`_.

.. _ci::

Continuous integration
----------------------

We use continuous integration services on GitHub to automatically check
if new pull requests do not break anything and meet code quality
standards such as a common `coding style <#Coding-style>`__.

Code quality checks
~~~~~~~~~~~~~~~~~~~

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
`docs/requirements.txt <https://github.com/alan-turing-institute/sktime/blob/main/docs/requirements.txt>`__.

1. To install extra requirements from the root directory, run:

   .. code:: bash

      pip install -r docs/requirements.txt

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
you need to update the following files:

-  `sktime/setup.py <https://github.com/alan-turing-institute/sktime/blob/main/setup.py>`__
   for package installation and minimum version requirements,
-  `build_tools/requirements.txt <https://github.com/alan-turing-institute/sktime/blob/main/build_tools/requirements.txt>`__
   for continuous integration and distribution,
-  `docs/requirements.txt <https://github.com/alan-turing-institute/sktime/blob/main/docs/requirements.txt>`__
   for building the documentation,
-  `.binder/requirements.txt <https://github.com/alan-turing-institute/sktime/blob/main/.binder/requirements.txt>`__
   for launching notebooks on Binder.

If a user is missing a soft dependency, we raise a user-friendly error message.
This is handled through our ``_check_soft_dependencies`` defined
`here <https://github.com/alan-turing-institute/sktime/blob/main/sktime/utils/validation/_dependencies.py>`__.

We use contiunous integration tests to check if all soft
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

Releases
--------

This section is for core developers. To make a new release, you need
push-to-write access on our main branch.

sktime is not a pure Python package and depends on some non-Python code
including Cython and C. We distribute compiled files, called wheels, for
different operating systems and Python versions.

.. note::

   For more details, see the `Python guide for packaging <https://packaging.python.org/guides/>`__ and the `Cython guide on compilation/distribution <https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html>`_.

We use :ref:`continuous integration <infrastructure>` services to automate the building of wheels on different platforms.
The release process is triggered by pushing a non-annotated `tagged
commit <https://git-scm.com/book/en/v2/Git-Basics-Tagging>`__ using
`semantic versioning <https://semver.org>`__.
Pushing a new tag will build the wheels for different platforms and upload them to PyPI.

You can see all available wheels `here <https://pypi.org/simple/sktime/>`__.

To make the release process easier, we have an interactive script that
you can follow. Simply run:

.. code:: bash

   make release

This calls
`build_tools/make_release.py <https://github.com/alan-turing-institute/sktime/blob/main/build_tools/make_release.py>`__
and will guide you through the release process.

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
