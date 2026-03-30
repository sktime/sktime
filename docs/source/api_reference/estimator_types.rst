.. _estimator_types_ref:

Estimator and object types
==========================

``sktime`` supports unified interfaces for different types of algorithms and objects.
These are internally referred to as *scitypes*, with strict string identifiers
such as ``"forecaster"``, ``"classifier"``, ``"detector"``, etc.

Each scitype has a corresponding base class that defines its interface.
Users can retrieve estimators of a given type using :func:`sktime.registry.all_estimators`,
and can inspect or filter by type using the ``object_type`` tag.

For a list of all tags, see :ref:`tags_ref`.

Retrieving estimator types programmatically
-------------------------------------------

.. currentmodule:: sktime.registry

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    get_obj_scitype_list
    get_base_class_register
    get_base_class_lookup

List of estimator and object types
-----------------------------------

.. currentmodule:: sktime.registry._base_classes
.. autosummary::
    :toctree: auto_generated/
    :template: class.rst
    :nosignatures:

    object
    estimator
    forecaster
    global_forecaster
    classifier
    early_classifier
    regressor
    clusterer
    transformer
    transformer_pairwise
    transformer_pairwise_panel
    detector
    metric
    metric_forecasting
    metric_forecasting_proba
    metric_detection
    param_est
    aligner
    splitter
    network
    dataset
    dataset_classification
    dataset_forecasting
    dataset_regression
    catalogue
    reconciler