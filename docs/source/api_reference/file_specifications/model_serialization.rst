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
    :widths: 20 30 50
    :header-rows: 1

    * - Member
      - Serialization
      - Contents
    * - ``_metadata``
      - ``pickle`` or ``cloudpickle``
      - The estimator class, used by :func:`sktime.base.load` to select its
        loading implementation.
    * - ``_obj``
      - ``pickle`` or ``cloudpickle``
      - The serialized estimator instance, including its fitted state when the
        estimator was fitted before saving.

The ``serialization_format`` argument of ``save`` selects ``"pickle"``
(the default) or ``"cloudpickle"``. ``cloudpickle`` is an optional dependency.
The selected format is used for both members of archives written by the base
implementation.

To restore an archive, pass either the original string path without the
``.zip`` suffix or a :class:`pathlib.Path` pointing to the archive to
:func:`sktime.base.load`:

.. code-block:: python

    from pathlib import Path

    from sktime.base import load

    restored_from_string = load("model")
    restored_from_path = load(Path("model.zip"))

The loader reads ``_metadata`` first and delegates the remainder of the work to
the class method ``load_from_path`` of the stored estimator class.

In-memory container
-------------------

Calling ``estimator.save()`` without a path returns a two-element tuple:

1. the estimator class;
2. a bytes object containing the serialized estimator instance.

Pass this tuple directly to :func:`sktime.base.load` to restore the estimator.
The loader delegates to the class method ``load_from_serial`` of the class in
the first tuple element.

Extension points
----------------

The base format covers estimators whose state can be stored in one serialized
object. Estimators with additional resources can override ``save``,
``load_from_serial``, or ``load_from_path``. Such estimators may add archive
members while retaining ``_metadata`` and ``_obj`` so that the generic loader
can identify the estimator class and dispatch to its loading hook.

For example, deep learning estimators may store model weights or training
history alongside the base members. Consumers of serialized estimators should
therefore use the public :meth:`~sktime.base.BaseObject.save` and
:func:`sktime.base.load` interfaces rather than depending on archive members
other than those documented above.

Security considerations
-----------------------

Both supported serialization formats can execute arbitrary code during
deserialization. Only load archives or in-memory containers from trusted
sources.
