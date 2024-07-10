.. _coding_standards:

================
Coding standards
================

.. contents::
   :local:

Coding style
============

In coding, we follow:

*  the `PEP8 <https://www.python.org/dev/peps/pep-0008/>`__ coding guidelines. A good example can be found `here <https://gist.github.com/nateGeorge/5455d2c57fb33c1ae04706f2dc4fee01>`__.

* code formatting according to ``ruff``

Code formatting and linting
---------------------------

We adhere to the following code formatting standards:

* `ruff <https://docs.astral.sh/ruff/>`__ with a ``max_line_length=88`` more configurations can be found in ``pyproject.toml``

* ``numpydoc`` to enforce numpy `docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_ , along with sktime specific conventions described in our :ref:`developer_guide`'s :ref:`documentation section <developer_guide_documentation>`.

This is enforced through our CI/CD workflows via `pre-commit <https://pre-commit.com/>`_.

The full pre-commit configuration can be found in
`.pre-commit-config.yaml <https://github.com/sktime/sktime/blob/main/.pre-commit-config.yaml>`_.
Additional configurations can be found in
`pyproject.toml <https://github.com/sktime/sktime/blob/main/pyproject.toml>`_.

``sktime`` specific code formatting conventions
-----------------------------------------------

-  Use underscores to separate words in non-class names: ``n_instances``
   rather than ``ninstances``.
-  exceptionally, capital letters ``X``, ``Y``, ``Z``, are permissible as variable names
   or part of variable names such as ``X_train`` if referring to data sets, in accordance
   with the PEP8 convention that such variable names are permissible if in prior use in an area
   (here, this is the ``scikit-learn`` adjacent ecosystem)
-  Avoid multiple statements on one line. Prefer a line return after a
   control flow statement (``if``/``for``).
-  Use absolute imports for references inside sktime.
-  Don't use ``import *`` in the source code. It is considered
   harmful by the official Python recommendations. It makes the code
   harder to read as the origin of symbols is no longer explicitly
   referenced, but most important, it prevents using a static analysis
   tool like pyflakes to automatically find bugs.

Setting up local code quality checks
------------------------------------

There are two options to set up local code quality checks:

* using ``pre-commit`` for automated code formatting
* setting up ``ruff`` and/or ``numpydoc`` manually in a local dev IDE

Using pre-commit
^^^^^^^^^^^^^^^^

To set up pre-commit, follow these steps in a python environment
with the ``sktime`` ``dev`` dependencies installed.

Type the below in your python environment, and in the root of your local repository clone:

1. If not already done, ensure ``sktime`` with ``dev`` dependencies is installed, this includes ``pre-commit``:

.. code:: bash

   pip install -e .[dev]

2. Set up pre-commit:

.. code:: bash

   pre-commit install

Once installed, pre-commit will automatically run all ``sktime`` code quality
checks on the files you changed whenever you make a new commit.

You can find our pre-commit configuration in
`.pre-commit-config.yaml <https://github.com/sktime/sktime/blob/main/.pre-commit-config.yaml>`_.
Additional configurations can be found in
`pyproject.toml <https://github.com/sktime/sktime/blob/main/pyproject.toml>`_.

.. note::
   If you want to exclude some line of code from being checked, you can add a ``# noqa: rule`` (no quality assurance) comment at the end of that line.
   The set of rules that can be excluded can be found here `ruff rules <https://docs.astral.sh/ruff/rules>`_.

Integrating with your local developer IDE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Local developer IDEs will usually integrate with common code quality checks, but need setting them up in IDE specific ways.

For Visual Studio Code, ``ruff`` and/or ``numpydoc`` will need to be activated individually in the preferences you can install ``ruff`` vscode extension from the marketplace and can be found here `ruff extension <https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff>`_.
The Ruff VS Code extension will respect any Ruff configuration as defined in your project's ``pyproject.toml``, ``ruff.toml``, or ``.ruff.toml file`` this means that after installing the extension you can start using it right away.
The packages ``ruff`` etc will need to be installed in the python environment used by the IDE,
this can be achieved by an install of ``sktime`` with ``dev`` dependencies.

In Visual Studio Code, we also recommend to add ``"editor.ruler": 88`` to your local ``settings.json`` to display the max line length.

API design
============

The general design approach of sktime is described in the
paper `"Designing Machine Learning Toolboxes: Concepts, Principles and
Patterns" <https://arxiv.org/abs/2101.04938>`__.

.. note::

   Feedback and improvement suggestions are very welcome!
