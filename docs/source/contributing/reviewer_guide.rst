.. _reviewer_guide:

==============
Reviewer Guide
==============

.. warning::

    The reviewer guide is under development.


Triage
======

* Assign relevant labels
* Assign to relevant project board
* Title: Is it using the 3-letter codes? Is it understandable?
* Description: Is it understandable? Any related issues/PRs?
* CI checks: approval for first-time contributors, any help needed with
  code/doc quality checks?
* Merge conflicts

Code
====

* Unit testing: Are the code changes tested? Are the tests understandable? Are all changes covered by tests? We usually aim for a test coverage of at least 90%. Code coverage will be reported as part of the automated CI checks on GitHub and on the `Codecov website <https://app.codecov.io/gh/sktime/sktime>`_.
* Test changes locally: Does everything work as expected?
* Deprecation warnings: Has the public API changed? Have deprecation warnings been added before making the changes?

.. _reviewer_guide_doc:

Documenation
============

* Are the docstrings complete and understandable to users?
* Do they follow the NumPy format and sktime conventions?
* If the same parameter, attribute, return object or error is included elsewhere in sktime are the docstring descriptions
  as similar as possible
* Does the online documentation render correctly with the changes?
* Do the docstrings contain links to the relevant topics in the :ref:`glossary` or :ref:`user_guide`?

.. warning::

    If a Pull Request does not meet sktime's :ref:`documentation guide <developer_guide_documentation>`
    a reviewer should require the documentation be updated prior to approving the Pull Request.
