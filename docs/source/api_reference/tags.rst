.. _tags_ref:

Object and estimator tags
=========================

Every first-class object in ``sktime``
is tagged with a set of tags that describe its properties and capabilities,
or control its behavior.

Tags are key-value pairs, where the key is a string with the name of the tag.
The value of the tag can have arbitrary type, and describes a property, capability,
or controls behaviour of the object, depending on the tag.

For instance, a forecaster has the tag ``"capability:pred_int": True`` if it can
make probabilistic predictions.
Users can find all forecasters that can make probabilistic predictions by filtering
for this tag.

This API reference lists all tags available in ``sktime``, and key utilities
for their usage.

To search estimators by tags on the ``sktime`` webpage, use the
:doc:`Estimator Search Page </estimator_overview>`

To search estimators by tags in a python environment, use the
``sktime.registry.all_estimators`` utility.


Inspecting tags, retrieving by tags
-----------------------------------

Tags can be inspected at runtime using the following utilities:

* to get the tags of an object, use the ``get_tags`` method.
  An object's tags can depend on its hyper-parameters.
* to get the tags of a class, use the ``get_class_tags`` method of the class.
  A class's tags are static and do not depend on its hyper-parameters.
  By default, class tags that may vary for instances take the most "capable" value,
  in the case of capabilities.
* to programmatically retrieve all tags available in ``sktime``
  or for a particular type of object, at runtime, use the ``registry.all_tags`` utility
* to programmatically retrieve all objects or estimators in ``sktime``,
  filtered for values of tags, use the ``registry.all_estimators`` utility


.. currentmodule:: sktime.registry

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    all_tags
    all_estimators


.. _packaging_tags:

General tags, packaging
-----------------------

This section lists tags that are general and apply to all objects in ``sktime``.
These tags are typically used for typing, packaging and documentation purposes.

.. currentmodule:: sktime.registry._tags

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst
    :nosignatures:

    object_type
    maintainers
    authors
    python_version
    python_dependencies
    env_marker
    requires_cython


.. _forecaster_tags:

Forecaster tags
---------------

This section lists tags applying to forecasters (``"forecaster"`` type).
These tags are used to describe capabilities, properties, and behavior of forecasters.

.. currentmodule:: sktime.registry._tags

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst
    :nosignatures:

    capability__exogeneous
    capability__insample
    capability__pred_int
    capability__pred_int__insample
    capability__missing_values
    capability__categorical_in_X
    capability__random_state
    requires_fh_in_fit
    fit_is_empty
    property__randomness


.. _panel_tags:

Tags for classifiers, regressors, clustering
--------------------------------------------

This section lists tags applying to time series classifiers, regressors,
and clusterers  (``"classifier"``, ``"regressor"``, ``"clusterer"`` types).
These tags are used to describe capabilities, properties, and behavior of
these types of objects.

.. currentmodule:: sktime.registry._tags

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst
    :nosignatures:

    capability__multivariate
    capability__multioutput
    capability__missing_values
    capability__unequal_length
    capability__predict_proba
    capability__feature_importance
    capability__contractable
    capability__train_estimate
    capability__random_state
    property__randomness


.. _transformer_tags:

Tags for ordinary transformers
------------------------------

This section lists tags applying to ordinary transformers, i.e., objects that
transform a single time series object (``"transformer"`` type).

.. currentmodule:: sktime.registry._tags

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst
    :nosignatures:

    scitype__transform_input
    scitype__transform_output
    scitype__transform_labels
    requires_x
    requires_y
    capability__missing_values
    capability__unequal_length
    capability__unequal_length__adds
    capability__unequal_length__removes
    capability__random_state
    capability__inverse_transform
    capability__inverse_transform__exact
    capability__inverse_transform__range
    capability__bootstrap_index
    fit_is_empty
    transform_returns_same_time_index
    property__randomness


.. _detector_tags:

Tags for detectors
------------------

This section lists tags applying to time series detectors (``"detector"`` types).
These tags are used to describe capabilities, properties, and behavior of
detectors.

.. currentmodule:: sktime.registry._tags

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst
    :nosignatures:

    task
    learning_type
    capability__update
    capability__multivariate
    capability__missing_values
    capability__random_state
    property__randomness


.. _metric_tags:

Tags for metrics
----------------

This section lists tags applying to time series metrics (``"metric"`` type).

.. currentmodule:: sktime.registry._tags

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst
    :nosignatures:

    lower_is_better
    capability__sample_weight
    scitype__y_pred
    requires_y_true
    requires_y_pred_benchmark
    requires_y_train
    inner_implements_multilevel


Tags for time series aligners
-----------------------------

This section lists tags applying to time series aligners (``"aligner"`` type).

.. currentmodule:: sktime.registry._tags

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst
    :nosignatures:

.. _dev_common_tags:

    capability__multiple_alignment
    capability__distance
    capability__distance_matrix
    property__alignment_type


Common developer tags
---------------------

This section lists tags that are used to control internal behaviour of objects,
e.g., the boilerplate layer.

These are primarily useful for power users using the extension
templates to create ``sktime`` compatible objects.

The tags below have limited use in retrieval or inspection of objects.

.. currentmodule:: sktime.registry._tags

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst
    :nosignatures:

    x_inner_mtype
    y_inner_mtype
    visual_block_kind

.. _dev_testing_tags:

Testing and CI tags
-------------------

These tags control behaviour of estimators in the ``sktime`` continuous integration
tests.

They are primarily useful for developers managing CI behaviour of individual objects.

.. currentmodule:: sktime.registry._tags

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst
    :nosignatures:

    tests__core
    tests__vm
    tests__skip_all
    tests__skip_by_name
