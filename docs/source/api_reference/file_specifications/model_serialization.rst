.. _model_serialization_format:

Estimator Serialization Format
==============================

``sktime`` estimators can be serialized with :meth:`sktime.base.BaseObject.save`
and restored with :func:`sktime.base.load`. This page describes the on-disk and
in-memory containers produced by the base implementation. Estimators with
additional external state may override the loading hooks and add files to the
on-disk container.

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
        estimator was fitted before saving.

Both members are written with the same serializer. The
``serialization_format`` argument of ``save`` selects ``"pickle"``
(the default) or ``"cloudpickle"``. ``cloudpickle`` is an optional dependency.

The archive is flat, with the two members at its root:

.. code-block:: text

    model.zip
    ├── _metadata
    └── _obj

To restore an archive, pass either the original string path without the
``.zip`` suffix or a :class:`pathlib.Path` pointing to the archive to
:func:`sktime.base.load`. The loader reads ``_metadata`` first and delegates
the remainder of the work to the class method ``load_from_path`` of the stored
estimator class.

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
2. a bytes object containing the serialized estimator instance.

Pass this tuple directly to :func:`sktime.base.load` to restore the estimator.
The loader delegates to the class method ``load_from_serial`` of the class in
the first tuple element, passing it the second element:

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

The base format covers estimators whose state can be stored in one serialized
object. Estimators with additional resources can override ``save``,
``load_from_serial``, or ``load_from_path``. Such estimators may add archive
members while retaining ``_metadata`` and ``_obj`` so that the generic loader
can identify the estimator class and dispatch to its loading hook.

For example, the deep learning estimators store the ``keras`` model and the
training history alongside the base members, so their archives look as follows:

.. code-block:: text

    model.zip
    ├── _metadata
    ├── _obj
    ├── history
    └── keras/
        └── model.keras

Their in-memory container remains a two-element tuple, with the additional
state nested inside the second element. Consumers of serialized estimators
should therefore use the public :meth:`~sktime.base.BaseObject.save` and
:func:`sktime.base.load` interfaces rather than depending on archive members
other than those documented above.

Security considerations
-----------------------

Both supported serialization formats can execute arbitrary code during
deserialization. Only load archives or in-memory containers from trusted
sources.
