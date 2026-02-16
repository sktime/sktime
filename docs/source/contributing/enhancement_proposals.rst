.. _enhancement_proposals:
.. _steps:

=====================
Enhancement Proposals
=====================

Description
===========

An sktime enhancement proposal (STEP) is a software design document providing information to the sktime community.
The proposal should provide a rationale and concise technical specification of proposed design.

We collect and discuss proposals in sktime's STEP `repository`_.

.. _repository: https://github.com/sktime/enhancement-proposals

We intend STEPs to be the primary mechanisms for proposing major changes, for collecting community input on an issue, and for documenting the design decisions that have gone into sktime.
Smaller changes can be discussed and implemented directly on issues and pull requests.

For the general design principles and patterns followed in sktime, we refer to our paper: `Designing ML Toolboxes: Concepts, Principles and Patterns <https://arxiv.org/abs/2101.04938>`_.

Submitting a STEP
=================

To create a new STEP, please copy and use `template`_ and open a pull request on the `repository`_.

.. _template: https://github.com/sktime/enhancement-proposals/blob/main/TEMPLATE.md

It is highly recommended that a single STEP contains a single key proposal or new idea.
The more focused the proposal, the more successful it tends to be.
If in doubt, split your STEP into several well-focused ones.

A STEP should be a consolidated document, including:

* a concise problem statement,
* a clear description of the proposed solution,
* a comparison with alternative solutions.
