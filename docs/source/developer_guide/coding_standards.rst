.. _coding_standards:

Coding standards
================

.. contents::
   :local:

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

Soft dependencies in sktime should usually be restricted to estimators.
To add an estimator with a soft dependency, ensure the following:

*   imports of the soft dependency only happen inside the estimator,
    e.g., in ``_fit`` or ``__init__`` methods of the estimator.
*   Errors and warnings, with informative instructions on how to install the soft dependency,
    are raised through ``_check_soft_dependencies``
    `here <https://github.com/alan-turing-institute/sktime/blob/main/sktime/utils/validation/_dependencies.py>`__.
    In the python module containing the estimator, the function should be called twice:
    at the top of the module, with ``severity="warning"``. This will warn the user whenever
    they import the file and the soft dependency is not installed; and, at the beginning
    of ``__init__``, with ``severity="error"``. This will raise an exception whenever
    the user attempts to instantiate the estimator, and the soft dependency is not installed.
*   ensure the module containing the estimator is registered
    `here <https://github.com/alan-turing-institute/sktime/blob/main/build_tools/azure/check_soft_dependencies.py>`__.
    This allows continuous integration tests to check if all soft dependencies are properly isolated to specific modules.

In addition, if the soft dependency introduced is new to ``sktime``,
or whenever changing the version of an existing one, you need to update the following files:

*   `pyproject.toml <https://github.com/alan-turing-institute/sktime/blob/main/pyproject.toml>`__,
    following the `PEP 621 <https://www.python.org/dev/peps/pep-0621/>`_ convention all dependencies
    including build time dependencies and optional dependencies are specified in this file.
*   Ensure new soft dependencies are added
    `here <https://github.com/alan-turing-institute/sktime/blob/main/build_tools/azure/check_soft_dependencies.py>`__
    together with the module that depends on it.

API design
----------

The general design approach of sktime is described in the
paper `“Designing Machine Learning Toolboxes: Concepts, Principles and
Patterns” <https://arxiv.org/abs/2101.04938>`__.

.. note::

   Feedback and improvement suggestions are very welcome!
