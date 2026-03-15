.. _estimator_types_ref:

===============
Estimator types
===============

This section lists the various estimator types (scitypes) available in ``sktime``.

Each estimator type corresponds to a specific unified interface and base class.
Every object in ``sktime`` has one or more scitypes, which can be inspected
via its ``"object_type"`` tag.

.. currentmodule:: sktime.registry._base_classes

General types
-------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst
    :nosignatures:

    object
    estimator

Specific estimator types
------------------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst
    :nosignatures:

    aligner
    classifier
    clusterer
    detector
    forecaster
    param_est
    regressor
    transformer

Other object types
------------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst
    :nosignatures:

    metric
    splitter
    dataset
    catalogue
