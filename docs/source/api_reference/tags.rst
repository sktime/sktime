.. _tags_ref:

Object and estimator tags
=========================

Every first-class object in ``sktime``
is tagged with a set of tags that describe its properties and capabilities,
or control its behavior.

Tags are key-value pairs, where the key is a string with the name of the tag.
The value of the tag can have arbitrary type, and describes a property, capability,
or controls behaviour of the object, depending on the value.

For instance, a forecaster may have the tag ``capability:pred_int: True`` if it can
make probabilistic predictions.
Users can find all forecasters that can make probabilistic predictions by filtering
for this tag.

This API reference lists all tags available in ``sktime``, and key utilities
for their usage.


Inspecting tags, retrieving by tags
-----------------------------------

* to get the tags of an object, use the ``get_tags`` method.
  An object's tags can depend on its hyper-parameters.
* to get the tags of a class, use the ``get_tags`` method of the class.
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

This section lists tags applying to forecasters.
These tags are used to describe capabilities, properties, and behavior of forecasters.

The list also includes some developer facing tags that are used to
control internal behavior of the forecaster.

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


.. _panel_tags:

Tags for classifiers, regressors, clustering
--------------------------------------------

.. currentmodule:: sktime.registry._tags

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst
    :nosignatures:

    capability__multivariate
    capability__missing_values
    capability__unequal_length
    capability__feature_importance
    capability__contractable
    capability__train_estimate
