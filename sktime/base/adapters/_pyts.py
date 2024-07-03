# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for pyts models."""

__all__ = ["_PytsAdapter"]
__author__ = ["fkiraly"]

from inspect import signature

from sktime.utils.adapters.forward import _clone_fitted_params


class _PytsAdapter:
    """Mixin adapter class for pyts models."""

    _tags = {
        "X_inner_mtype": "numpyflat",
        "python_dependencies": ["pyts"],
    }

    # defines the name of the attribute containing the pyts estimator
    _estimator_attr = "_estimator"

    def _get_pyts_class(self):
        """Abstract method to get pyts class.

        should import and return pyts class
        """
        # from pyts import PytsClass
        #
        # return Pyts
        raise NotImplementedError("abstract method")

    def _get_pyts_object(self):
        """Abstract method to initialize pyts object.

        The default initializes result of _get_pyts_class
        with self.get_params.
        """
        cls = self._get_pyts_class()
        return cls(**self.get_params())

    def _init_pyts_object(self):
        """Abstract method to initialize pyts object and set to _estimator_attr.

        The default writes the return of _get_pyts_object to
        the attribute of self with name _estimator_attr
        """
        cls = self._get_pyts_object()
        setattr(self, self._estimator_attr, cls)
        return getattr(self, self._estimator_attr)

    def _call_with_y_optional(self, method, X, y=None):
        """Call method with or without y parameter.

        The y parameter is passed only if the method has it.

        Parameters
        ----------
        method: callable
            Method to call
        X, y: any objects
        """
        # check if method has y parameter
        # if yes, call with y, otherwise without
        method_has_y = "y" in signature(method).parameters

        if method_has_y:
            return method(X, y)
        else:
            return method(X)

    def _fit(self, X, y=None):
        """Fit estimator training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_dimensions, series_length)
            Training features, passed only for classifiers or regressors
        y: None or 1D np.ndarray of shape (n_instances,)
            Training labels, passed only for classifiers or regressors

        Returns
        -------
        self: sktime estimator
            Fitted estimator.
        """
        pyts_est = self._init_pyts_object()

        self._call_with_y_optional(pyts_est.fit, X, y)

        # write fitted params to self
        _clone_fitted_params(to_obj=self, from_obj=pyts_est)

        return self

    def _transform(self, X, y=None):
        """Transform method adapter.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances, n_dimensions, series_length))
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        np.ndarray (1d array of shape (n_instances,))
            Index of the cluster each time series in X belongs to.
        """
        pyts_est = getattr(self, self._estimator_attr)

        return self._call_with_y_optional(pyts_est.transform, X, y)

    def _predict(self, X, y=None):
        """Predict method adapter.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances, n_dimensions, series_length))
        y: passed to pyts predict method if it has y parameter

        Returns
        -------
        np.ndarray (1d array of shape (n_instances,))
            Index of the cluster each time series in X belongs to.
        """
        pyts_est = getattr(self, self._estimator_attr)

        return self._call_with_y_optional(pyts_est.predict, X, y)

    def _predict_proba(self, X, y=None):
        """Predict_proba method adapter.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances, n_dimensions, series_length))
            Time series instances to predict their cluster indexes.
        y: passed to pyts predict method if it has y parameter

        Returns
        -------
        np.ndarray (1d array of shape (n_instances,))
            Index of the cluster each time series in X belongs to.
        """
        pyts_est = getattr(self, self._estimator_attr)

        # check if pyts_est has predict_proba
        # if not, call super's _predict_proba
        pyts_has_predict_proba = hasattr(pyts_est, "predict_proba")
        if not pyts_has_predict_proba:
            method = super()._predict_proba
        else:
            method = pyts_est.predict_proba

        return self._call_with_y_optional(method, X, y)
