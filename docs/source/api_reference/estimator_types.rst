.. _estimator_types_ref:

===============
Estimator Types
===============

``sktime`` supports unified interfaces for different types of algorithms and objects, e.g., ``forecaster``, ``classifier``, ``detector``, etc. These are internally used throughout as strict types with defined strings.

This page lists all the different estimator types available in ``sktime``, with a shorthand explanation and links to further details for each type.

.. currentmodule:: sktime.registry._base_classes

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst
    :nosignatures:

    object
    estimator
    aligner
    classifier
    clusterer
    early_classifier
    forecaster
    global_forecaster
    metric
    metric_detection
    metric_forecasting
    metric_forecasting_proba
    network
    param_est
    regressor
    detector
    splitter
    transformer
    transformer_pairwise
    transformer_pairwise_panel
    dataset
    dataset_classification
    dataset_forecasting
    dataset_regression
    catalogue
    reconciler
