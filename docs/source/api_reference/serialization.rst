.. _serialization_ref:

Estimator serialization
=======================

All :class:`sktime.base.BaseObject` descendants can be persisted with ``save``
and restored with :func:`sktime.base.load`. This includes fitted estimators.
The public workflow is the same whether an estimator contains ordinary Python
state or models from deep-learning frameworks:

.. code-block:: python

    from sktime.base import load
    from sktime.forecasting.naive import NaiveForecaster

    forecaster = NaiveForecaster().fit(y, fh=[1, 2, 3])
    forecaster.save("forecaster")

    restored = load("forecaster")
    y_pred = restored.predict()

``save("forecaster")`` creates ``forecaster.zip``. The string passed to
``load`` is the original save location, without the automatically appended
``.zip`` suffix. A :class:`pathlib.Path` passed to ``load`` must point to the
ZIP file itself.

.. warning::

   Estimator archives contain pickle data. Only load archives from trusted
   sources. Loading an archive may execute arbitrary Python code.


Archive contents
----------------

An ordinary estimator archive contains two files:

.. code-block:: text

    forecaster.zip
    |-- _metadata
    `-- _obj

``_metadata`` contains the estimator class and ``_obj`` contains its serialized
Python state. The ``serialization_format`` argument to ``save`` controls how
these two entries are written. Supported values are ``"pickle"`` (the default)
and ``"cloudpickle"``.

Some estimators contain fitted models that should be stored with their
framework's native persistence API. For these estimators, the archive also has
an ``_artifacts`` directory:

.. code-block:: text

    forecaster.zip
    |-- _metadata
    |-- _obj
    `-- _artifacts
        |-- index.json
        `-- model_
            `-- ... framework-specific files ...

The directory name for each artifact is the estimator attribute name, such as
``model_`` or ``network``. ``index.json`` maps these names to the backend,
Python class, and relative path needed during loading. Native attributes are not
duplicated inside ``_obj``. If no non-``None`` native artifact is present, the
``_artifacts`` directory is omitted.

For example, the index entry for an attribute named ``model_`` can be:

.. code-block:: json

    {
      "model_": {
        "backend": "keras",
        "class": "keras.src.models.sequential.Sequential",
        "path": "model_"
      }
    }

The exact files below the attribute directory depend on the selected backend.

.. list-table:: Native artifact layouts
   :header-rows: 1
   :widths: 22 28 50

   * - Backend in ``index.json``
     - Typical artifact files
     - Save and load mechanism
   * - ``pretrained``
     - ``config.json`` and model weight files, or PEFT adapter configuration and
       weight files
     - ``save_pretrained`` and ``from_pretrained``. Exact filenames are defined
       by the installed library and may include sharded weight indexes.
   * - ``keras``
     - ``model.keras``
     - ``keras.Model.save`` and ``keras.models.load_model``
   * - ``lightning_checkpoint``
     - ``model.ckpt``
     - A Lightning checkpoint and ``load_from_checkpoint``
   * - ``torch_state_dict``
     - ``state_dict.pt``
     - A CPU state dictionary. The estimator reconstructs the module
       architecture before the state is loaded.

For example, a fitted estimator with a Keras ``model_`` attribute has this
layout:

.. code-block:: text

    estimator.zip
    |-- _metadata
    |-- _obj
    `-- _artifacts
        |-- index.json
        `-- model_
            `-- model.keras

A Transformers model using the attribute ``model`` may instead look like:

.. code-block:: text

    estimator.zip
    |-- _metadata
    |-- _obj
    `-- _artifacts
        |-- index.json
        `-- model
            |-- config.json
            `-- model.safetensors

The Transformers library chooses the precise weight filenames. Large models can
therefore contain multiple weight shards and a ``*.index.json`` file instead.


In-memory serialization
-----------------------

Calling ``save()`` without a path returns a two-element tuple:

.. code-block:: python

    cls, serialized = estimator.save()
    restored = load((cls, serialized))

For an estimator without native artifacts, ``serialized`` is the ordinary
pickle or cloudpickle byte stream. When native artifacts are present, it is an
in-memory ZIP archive with the same layout described above. Callers should treat
the bytes as opaque and pass the complete tuple to :func:`sktime.base.load`.


Loading requirements
--------------------

Loading restores ``_obj`` first and then each entry in ``_artifacts/index.json``.
The estimator's class and all libraries needed by its native artifacts must be
importable in the loading environment. In practice, use compatible versions of
``sktime``, Python, and the relevant frameworks when moving an archive between
environments.

Artifacts tagged as skipped caches are absent from the archive. The estimator
is responsible for recreating any skipped attribute that it needs. For example,
a zero-shot estimator can reload a cached model from its configured model
identifier when prediction is requested.


Developer interface
-------------------

Serialization tags are private, framework-facing tags. Estimators opt in by
listing fitted attributes in one or both tags:

.. code-block:: python

    from sktime.base import BaseEstimator

    class NativeModelEstimator(BaseEstimator):
        _tags = {
            "serialization:native_artifacts": ("model_",),
            "serialization:skip": ("trainer_",),
        }

``serialization:native_artifacts``
    Attributes persisted outside ``_obj`` using a supported native backend.
    Backend selection is based on the fitted object's type or native protocol.

``serialization:skip``
    Cache or wrapper attributes omitted from persistence. The estimator must be
    able to reconstruct these attributes from its serialized state before use.

Tag values are tuples or lists of attribute names. Missing attributes and
attributes whose value is ``None`` do not produce artifact entries. The source
estimator retains all attributes after ``save`` returns.

Trainable estimators should use native artifacts for fitted weights. A zero-shot
estimator may instead skip a cached model when it can reproduce that model from
serialized constructor or fitted state. If this choice depends on estimator
parameters or the fit strategy, set the tags dynamically on the instance.

Torch modules require the estimator to implement
``_create_torch_artifact(name)``. It must construct and return a compatible,
uninitialized ``torch.nn.Module``; the backend then loads ``state_dict.pt`` into
it. Pretrained-style artifacts can optionally implement
``_get_native_artifact_load_kwargs(name)`` to supply keyword arguments to
``from_pretrained``. Keras estimators with custom objects can expose
``get_custom_objects()`` for ``keras.models.load_model``.

Only attributes of the object on which ``save`` is called are selected by these
tags. Native-artifact discovery does not recursively inspect arbitrary nested
objects.
