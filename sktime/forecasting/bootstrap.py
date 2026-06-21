#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements bootstrapping for time series forecasting."""

__author__ = ["Shubh Garg"]
__all__ = ["BaseBootstrap"]

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from sktime.base import BaseObject


class BaseBootstrap(BaseObject):
	"""Base class for bootstrapping time series data.

	This class provides the base interface for bootstrapping algorithms in sktime,
	with built-in capability to return indices of bootstrapped samples.

	Parameters
	----------
	random_state : int, RandomState instance or None, default=None
		Controls the randomness of the estimator.
		If int, random_state is the seed used by the random number generator;
		If RandomState instance, random_state is the random number generator;
		If None, the random number generator is the RandomState instance used
		by `np.random`.

	Attributes
	----------
	random_state_ : RandomState
		Random number generator instance.

	Examples
	--------
	>>> from sktime.forecasting.bootstrap import BaseBootstrap
	>>> import numpy as np
	>>> X = np.array([1, 2, 3, 4, 5])
	>>> bootstrap = BaseBootstrap(random_state=42)
	>>> bootstrap.fit(X)
	>>> X_boot, indices = bootstrap.generate(X, n_samples=2)
	"""

	def __init__(self, random_state=None):
		self.random_state = random_state
		super(BaseBootstrap, self).__init__()

	def _fit(self, X):
		"""Fit the bootstrap to the data.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			Training data.

		Returns
		-------
		self : object
			Returns self.
		"""
		self.random_state_ = check_random_state(self.random_state)
		return self

	def _generate(self, X, n_samples=1):
		"""Generate bootstrap samples.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			Training data.
		n_samples : int, default=1
			Number of bootstrap samples to generate.

		Returns
		-------
		Xt : array-like of shape (n_samples, n_features)
			Generated bootstrap samples.
		indices : array-like of shape (n_samples,)
			Indices of the samples selected in the bootstrap.
		"""
		raise NotImplementedError("abstract method")

	def fit(self, X):
		"""Fit the bootstrap to the data.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			Training data.

		Returns
		-------
		self : object
			Returns self.
		"""
		self._fit(X)
		return self

	def generate(self, X, n_samples=1):
		"""Generate bootstrap samples.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			Training data.
		n_samples : int, default=1
			Number of bootstrap samples to generate.

		Returns
		-------
		Xt : array-like of shape (n_samples, n_features)
			Generated bootstrap samples.
		indices : array-like of shape (n_samples,)
			Indices of the samples selected in the bootstrap.
		"""
		return self._generate(X, n_samples)