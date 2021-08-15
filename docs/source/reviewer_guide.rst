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

* Unit testing: Are the code changes tested? Are the unit tests understandable?
* Test code locally: Does everything work as expected?
* Deprecation warnings: Has the public API changed? Have deprecation
warnings been added before making the changes?

Documenation
============

* Are the docstrings complete? Are they understandable?
* Do they follow the NumPy format and sktime conventions?
* Does the online documentation render correctly with the changes?
* Could we add links to relevant topics in the :ref:`glossary` or
:ref:`user_guide`?
