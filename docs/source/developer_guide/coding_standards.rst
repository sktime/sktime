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

We use the `pre-commit <https://pre-commit.com/>`_ workflow together with
`black <https://black.readthedocs.io/en/stable/>`__ and
`flake8 <https://flake8.pycqa.org/en/latest/>`__ to apply
consistent formatting and check if your contribution complies with
the PEP8 style.

For docstrings, we use the `numpy docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_, along with sktime specific conventions described in our :ref:`developer_guide`'s :ref:`documentation section <developer_guide_documentation>`.

In addition, ensure that you:

-  Check out our :ref:`glossary`.
-  Use underscores to separate words in non-class names: ``n_instances``
   rather than ``ninstances``.
-  Avoid multiple statements on one line. Prefer a line return after a
   control flow statement (``if``/``for``).
-  Use absolute imports for references inside sktime.
-  Don’t use ``import *`` in the source code. It is considered
   harmful by the official Python recommendations. It makes the code
   harder to read as the origin of symbols is no longer explicitly
   referenced, but most important, it prevents using a static analysis
   tool like pyflakes to automatically find bugs.

.. _infrastructure::

API design
----------

The general design approach of sktime is described in the
paper `“Designing Machine Learning Toolboxes: Concepts, Principles and
Patterns” <https://arxiv.org/abs/2101.04938>`__.

.. note::

   Feedback and improvement suggestions are very welcome!
