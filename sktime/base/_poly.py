#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements polymorphic base class."""

__author__ = ["fkiraly", "benheid", "miraep8"]
__all__ = ["BasePolymorph"]

from sktime.base import BaseObject


class _Polymorph(BaseObject):
    """Polymorphic class mixin.

    Example
    -------
    from sktime.base._poly import _Polymorph
    from sktime.datasets import load_airline
    from sktime.forecasting.naive import NaiveForecaster

    f = _Polymorph(_cls=NaiveForecaster)

    y = load_airline()
    f = f.fit(y, fh=[1])
    """

    @classmethod
    def _infer_class(cls, *args, **kwargs):
        """Class inference method, can be overridden by descendants.

        The polymorph will behave as cls(*args, *kwargs),
        where cls is the a class inheriting from the output of _infer_class,
        and args/kwargs are the output of _infer_params.
        """
        cls = kwargs.pop("_cls")
        return (cls,)

    @classmethod
    def _infer_params(cls, *args, **kwargs):
        """Params inference method, can be overridden by descendants.

        The polymorph will behave as cls(*args, *kwargs),
        where cls is the a class inheriting from the output of _infer_class,
        and args/kwargs are the output of _infer_params.
        """
        return args, kwargs

    def __new__(cls, *args, **kwargs):
        """Polymorphic constructor."""
        dyn_cls_parents = cls._infer_class(*args, **kwargs)
        dyn_args, dyn_kwargs = cls._infer_params(*args, **kwargs)

        dyn_cls = type(
            cls.__name__,
            dyn_cls_parents,
            {"__init__": cls.__init__},
        )

        return dyn_cls(*dyn_args, **dyn_kwargs)

    def __init__(self, _cls=None):

        if _cls is not None:
            self._cls = _cls
            _cls.__init__(self)
        else:
            BaseObject.__init__(self)


class BasePolymorph(BaseObject):
    """Handles parameter management for estimators composed of named estimators.

    Partly adapted from sklearn utils.metaestimator.py.
    """

    @classmethod
    def _infer_estimator_type(cls, *args, **kwargs):
        """Estimator type inference method, can be overridden by children."""
        estimator_type = kwargs.get("estimator_type", "base")
        return estimator_type

    def __new__(cls, *args, **kwargs):
        """Polymorphic dispatcher to all sktime base classes."""
        from sktime.registry._lookup import _check_estimator_types

        estimator_type = cls._infer_estimator_type(*args, **kwargs)
        baseclass = _check_estimator_types(estimator_type)[0]
        if "estimator_type" in kwargs.keys():
            kwargs.pop("estimator_type")
        obj = baseclass(*args, **kwargs)

        return obj

    def __init__(self, estimator_type="base"):

        self.estimator_type = estimator_type
        super().__init__()


class _Delegator(BasePolymorph):
    """Polymorphic delegator class.

    Example
    -------
    from sktime.base._poly import _Delegator
    from sktime.datasets import load_airline
    from sktime.forecasting.naive import NaiveForecaster

    d = _Delegator(estimator=NaiveForecaster())

    y = load_airline()
    d = d.fit(y, fh=[1])
    """

    def __new__(cls, estimator):
        """Polymorphic dispatcher to all sktime delegator classes."""
        from sktime.classification._delegate import _DelegatedClassifier
        from sktime.forecasting.base._delegate import _DelegatedForecaster
        from sktime.registry import BASE_CLASS_LOOKUP, scitype
        from sktime.transformations._delegate import _DelegatedTransformer

        delegator_dict = {
            "classifier": _DelegatedClassifier,
            "forecaster": _DelegatedForecaster,
            "transformer": _DelegatedTransformer,
        }

        base = BASE_CLASS_LOOKUP[scitype(estimator)]
        delegator = delegator_dict[scitype(estimator)]

        def del_init(self, estimator):
            self.estimator = estimator
            self.estimator_ = estimator
            base.__init__(self)

        dyn_cls = type(
            cls.__name__,
            (delegator, base),
            {"__init__": del_init},
        )

        return dyn_cls(estimator)


class Pipeline(_Delegator):
    """Polymorphic pipeline class, via delegation.

    Example
    -------
    from sktime.base._poly import Pipeline
    from sktime.datasets import load_airline
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.transformations.series.exponent import ExponentTransformer

    Pipeline([ExponentTransformer(), NaiveForecaster()])

    y = load_airline()
    d = d.fit(y, fh=[1])
    """

    def __new__(cls, steps):

        from sktime.pipeline import make_pipeline

        pipe_delegate = make_pipeline(*steps)
        inst = super().__new__(cls, estimator=pipe_delegate)
        return inst

    def __init__(self, steps):
        """Polymorphic dispatcher to all sktime pipeline classes."""
        from sktime.pipeline import make_pipeline

        self.steps = steps

        pipe_delegate = make_pipeline(*steps)
        self.estimator_ = pipe_delegate


class Pipeline2(BasePolymorph):
    """Polymorphic pipeline class, via construction."""

    def __new__(cls, steps):
        """Polymorphic dispatcher to all sktime pipeline classes."""
        from sktime.pipeline import make_pipeline

        pipe_delegate = make_pipeline(*steps)

        obj = pipe_delegate

        return obj


class Pipeline3(BasePolymorph):
    """Polymorphic pipeline class, via delegation."""

    @classmethod
    def _infer_estimator_type(cls, *args, **kwargs):
        """Estimator type inference method, can be overridden by children."""
        from sktime.pipeline import make_pipeline
        from sktime.registry import scitype

        steps = kwargs.get("steps")
        return scitype(make_pipeline(*steps))

    def __init__(self, steps):
        self.steps = steps
        super().__init__()
