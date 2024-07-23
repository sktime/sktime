.. _tags_ref:

Object and estimator tags
=========================

Every first-class object in ``sktime``
is tagged with a set of tags that describe its properties and capabilities,
or control its behavior.

Tags are key-value pairs, where the key is a string with the name of the tag.
The value of the tag can have arbitrary type, and describes a property, capability,
or controls behaviour of the object, depending on the value.

For instance, a forecaster may have the tag ``"capability:pred_int": True`` if it can
make probabilistic predictions.
Users can find all forecasters that can make probabilistic predictions by filtering
for this tag.

This API reference lists all tags available in ``sktime``, and key utilities
for their usage.

To search estimators by tags on the ``sktime`` webpage, use the
:doc:`Estimator Search Page </estimator_overview>`


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
    python_dependencies_alias
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
    requires_fh_in_fit
    fit_is_empty


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
    capability__inverse_transform
    capability__inverse_transform__exact
    capability__inverse_transform__range
    fit_is_empty
    transform_returns_same_time_index


.. _dev_common_tags:

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
