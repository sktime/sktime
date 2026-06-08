.. _estimator_types_ref:

===============
Estimator types
===============

This section lists the various object types (scitypes) available in ``sktime``.

``sktime`` supports unified interfaces for different types of algorithms and objects.
These are internally referred to as *scitypes*, with strict string identifiers
such as ``"forecaster"``, ``"classifier"``, ``"detector"``, etc.

Each object type corresponds to a specific unified interface and base class.
Every object in ``sktime`` has one or more scitypes, which can be inspected
via its ``"object_type"`` tag.

For a list of all tags, see :ref:`tags_ref`.

.. currentmodule:: sktime.registry._base_classes

AI algorithms
-------------

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
    transformer_pairwise
    transformer_pairwise_panel

Data handling
-------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst
    :nosignatures:

    dataset
    dataset_classification
    dataset_forecasting
    dataset_regression
    splitter

Evaluation and metrics
----------------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst
    :nosignatures:

    metric
    metric_detection
    metric_forecasting
    metric_forecasting_proba

Catalogues and collections
--------------------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst
    :nosignatures:

    catalogue

General types
-------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst
    :nosignatures:

    object
    estimator

Retrieving estimator types programmatically
-------------------------------------------

.. currentmodule:: sktime.registry

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    get_obj_scitype_list
    get_base_class_register
    get_base_class_lookup
