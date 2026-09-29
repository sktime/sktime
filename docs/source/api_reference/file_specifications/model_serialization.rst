.. _model_serialization_format:

Estimator Serialization Format
==============================

``sktime`` estimators can be serialized with :meth:`sktime.base.BaseObject.save`
and restored with :func:`sktime.base.load`. This page describes the on-disk and
in-memory containers produced by the base implementation. Fitted framework
models are stored as optional native artifact members when an estimator opts in
through the serialization tags described under Extension points.


On-disk container
-----------------

Calling ``estimator.save(path)`` creates a ZIP archive at ``path`` with a
``.zip`` suffix. For example, ``estimator.save("model")`` creates
``model.zip`` in the current working directory. The temporary directory used
while creating the archive is removed before ``save`` returns.

The base implementation writes the following members to the archive:

.. list-table::
    :widths: 20 80
    :header-rows: 1

    * - Member
      - Contents
    * - ``_metadata``
      - The type of the estimator object, i.e., ``type(self)``. Used by
        :func:`sktime.base.load` to select the loading implementation.
    * - ``_obj``
      - The serialized estimator instance, including its fitted state when the
        estimator was fitted before saving. Attributes selected as native
        artifacts or skipped caches are omitted from this member.
    * - ``_artifacts/``
      - Optional directory of framework-native model files. Present only when
        at least one selected native artifact is not ``None``.

``_metadata`` and ``_obj`` are written with the same serializer. The
``serialization_format`` argument of ``save`` selects ``"pickle"``
(the default) or ``"cloudpickle"``. ``cloudpickle`` is an optional dependency.
Native artifact files always use the framework format of the selected backend,
independent of ``serialization_format``.

An archive without native artifacts is flat, with the two members at its root:

.. code-block:: text

    model.zip
    ├── _metadata
    └── _obj

When native artifacts are present, each artifact is a directory named after the
estimator attribute, such as ``model_`` or ``network``. ``index.json`` maps
those names to the backend, Python class, and relative path used during
loading. Native attributes are not duplicated inside ``_obj``.

.. code-block:: text

    model.zip
    ├── _metadata
    ├── _obj
    └── _artifacts
        ├── index.json
        └── model_
            └── ... framework-specific files ...

An index entry for an attribute named ``model_`` can look as follows:

.. code-block:: json

    {
      "model_": {
        "backend": "keras",
        "class": "keras.src.models.sequential.Sequential",
        "path": "model_"
      }
    }

The files below each attribute directory depend on the backend recorded in
``index.json``:

.. list-table:: Native artifact layouts
    :widths: 22 28 50
    :header-rows: 1

    * - Backend in ``index.json``
      - Typical artifact files
      - Save and load mechanism
    * - ``pretrained``
      - ``config.json`` and model weight files, or PEFT adapter
        configuration and weight files
      - ``save_pretrained`` and ``from_pretrained``. Exact filenames are
        defined by the installed library and may include sharded weight
        indexes.
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

    model.zip
    ├── _metadata
    ├── _obj
    └── _artifacts
        ├── index.json
        └── model_
            └── model.keras

A Transformers model stored on an attribute named ``model`` may instead look
as follows. The Transformers library chooses the weight filenames. Large models
can contain multiple weight shards and a ``*.index.json`` file.

.. code-block:: text

    model.zip
    ├── _metadata
    ├── _obj
    └── _artifacts
        ├── index.json
        └── model
            ├── config.json
            └── model.safetensors

To restore an archive, pass either the original string path without the
``.zip`` suffix or a :class:`pathlib.Path` pointing to the archive to
:func:`sktime.base.load`. The loader reads ``_metadata`` first and delegates
the remainder of the work to the class method ``load_from_path`` of the stored
estimator class. That method restores ``_obj`` and then each entry in
``_artifacts/index.json``. The estimator class and the libraries required by
its native artifacts must be importable in the loading environment. Use
compatible versions of ``sktime``, Python, and the relevant frameworks when
moving an archive between environments.

The following example saves a fitted forecaster, inspects the archive, and
loads it back:

.. code-block:: python

    from pathlib import Path
    from zipfile import ZipFile

    from sktime.base import load
    from sktime.datasets import load_airline
    from sktime.forecasting.naive import NaiveForecaster

    y = load_airline()
    forecaster = NaiveForecaster(strategy="mean")
    forecaster.fit(y, fh=[1, 2, 3])

    # creates model.zip in the current working directory
    forecaster.save("model")

    ZipFile("model.zip").namelist()
    # ['_metadata', '_obj']

    # both are equivalent
    restored_from_string = load("model")
    restored_from_path = load(Path("model.zip"))

    restored_from_path.predict()

In-memory container
-------------------

Calling ``estimator.save()`` without a path returns a two-element tuple:

1. the type of the estimator object, i.e., ``type(self)``;
2. a bytes object containing the serialized estimator.

When the estimator has no native artifacts, the second element is the pickle or
cloudpickle byte stream of the estimator. When native artifacts are present, it
is an in-memory ZIP archive with the same member layout as the on-disk
container. Callers should treat those bytes as opaque and pass the complete
tuple to :func:`sktime.base.load`. The loader delegates to the class method
``load_from_serial`` of the class in the first tuple element, passing it the
second element:

.. code-block:: python

    from sktime.base import load
    from sktime.datasets import load_airline
    from sktime.forecasting.naive import NaiveForecaster

    y = load_airline()
    forecaster = NaiveForecaster(strategy="mean")
    forecaster.fit(y, fh=[1, 2, 3])

    serial = forecaster.save()
    # (<class 'sktime.forecasting.naive._naive.NaiveForecaster'>, b'\x80\x05...')

    restored = load(serial)
    restored.predict()

Extension points
----------------

The base format covers estimators whose state can be stored in ``_obj``.
Estimators with fitted framework models or reconstructable caches extend that
format through serialization tags. These tags are private and framework-facing.
Tag values are tuples or lists of attribute names. Missing attributes and
attributes whose value is ``None`` do not produce archive entries. The source
estimator retains all attributes after ``save`` returns.

Only attributes of the object on which ``save`` is called are selected.
Native-artifact discovery does not inspect nested objects.

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
    Cache or wrapper attributes omitted from the archive. The estimator must
    reconstruct these attributes from its serialized state before use. A
    zero-shot estimator can, for example, reload a cached model from its
    configured model identifier when prediction is requested.

Trainable estimators should use native artifacts for fitted weights. A
zero-shot estimator may instead skip a cached model when it can reproduce that
model from serialized constructor or fitted state. If this choice depends on
estimator parameters or the fit strategy, set the tags dynamically on the
instance.

Torch modules require the estimator to implement
``_create_torch_artifact(name)``. It must construct and return a compatible,
uninitialized ``torch.nn.Module``. The backend then loads ``state_dict.pt``
into that module. Pretrained-style artifacts can optionally implement
``_get_native_artifact_load_kwargs(name)`` to supply keyword arguments to
``from_pretrained``. Keras estimators with custom objects can expose
``get_custom_objects()`` for ``keras.models.load_model``.

Security considerations
-----------------------

Both supported serialization formats can execute arbitrary code during
deserialization. Only load archives or in-memory containers from trusted
sources.
