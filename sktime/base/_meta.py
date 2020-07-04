#!/usr/bin/env python3 -u
# coding: utf-8
<<<<<<< HEAD

__author__ = ["Markus Löning"]
__all__ = ["BaseComposition"]


from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator


class BaseComposition(BaseEstimator, metaclass=ABCMeta):
    """Handles parameter management for estimtators composed of named estimators.
=======
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = [
    "MetaEstimatorMixin",
    "BaseHeterogenousMetaEstimator"
]

from abc import ABCMeta

from sktime.base import BaseEstimator


class MetaEstimatorMixin:
    _required_parameters = []


class BaseHeterogenousMetaEstimator(MetaEstimatorMixin, BaseEstimator,
                                    metaclass=ABCMeta):
    """Handles parameter management for estimtators composed of named
    estimators.
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed

    from sklearn utils.metaestimator.py
    """

<<<<<<< HEAD
    @abstractmethod
    def __init__(self):
        pass
=======
    def get_params(self, deep=True):
        raise NotImplementedError("abstract method")

    def set_params(self, **params):
        raise NotImplementedError("abstract method")
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed

    def _get_params(self, attr, deep=True):
        out = super().get_params(deep=deep)
        if not deep:
            return out
        estimators = getattr(self, attr)
        out.update(estimators)
        for name, estimator in estimators:
            if hasattr(estimator, 'get_params'):
                for key, value in estimator.get_params(deep=True).items():
                    out['%s__%s' % (name, key)] = value
        return out

    def _set_params(self, attr, **params):
        # Ensure strict ordering of parameter setting:
        # 1. All steps
        if attr in params:
            setattr(self, attr, params.pop(attr))
        # 2. Step replacement
        items = getattr(self, attr)
        names = []
        if items:
            names, _ = zip(*items)
        for name in list(params.keys()):
            if '__' not in name and name in names:
                self._replace_estimator(attr, name, params.pop(name))
        # 3. Step parameters and other initialisation arguments
        super().set_params(**params)
        return self

    def _replace_estimator(self, attr, name, new_val):
        # assumes `name` is a valid estimator name
        new_estimators = list(getattr(self, attr))
        for i, (estimator_name, _) in enumerate(new_estimators):
            if estimator_name == name:
                new_estimators[i] = (name, new_val)
                break
        setattr(self, attr, new_estimators)

    def _check_names(self, names):
        if len(set(names)) != len(names):
            raise ValueError('Names provided are not unique: '
                             '{0!r}'.format(list(names)))
        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError('Estimator names conflict with constructor '
                             'arguments: {0!r}'.format(sorted(invalid_names)))
        invalid_names = [name for name in names if '__' in name]
        if invalid_names:
            raise ValueError('Estimator names must not contain __: got '
                             '{0!r}'.format(invalid_names))
