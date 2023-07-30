# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for tslearn models."""

__all__ = ["_TslearnAdapter"]
__author__ = ["fkiraly"]

from inspect import signature


class _TslearnAdapter:
    """Mixin adapter class for tslearn models."""

    _tags = {
        "X_inner_mtype": "numpy3D",
        "python_dependencies": ["tslearn"],
    }

    # defines the name of the attribute containing the tslearn estimator
    _estimator_attr = "_estimator"

    def _get_tslearn_class(self):
        """Abstract method to get tslearn class.

        should import and return tslearn class
        """
        # from tslearn import TslearnClass
        #
        # return TslearnClass
        raise NotImplementedError("abstract method")

    def _get_tslearn_object(self):
        """Abstract method to initialize tslearn object.

        The default initializes result of _get_tslearn_class
        with self.get_params.
        """
        cls = self._get_tslearn_class()
        return cls(**self.get_params())

    def _init_tslearn_object(self):
        """Abstract method to initialize tslearn object and set to _estimator_attr.

        The default writes the return of _get_tslearn_object to
        the attribute of self with name _estimator_attr
        """
        cls = self._get_tslearn_object()
        setattr(self, self._estimator_attr, cls)
        return getattr(self, self._estimator_attr)

    def _fit(self, X, y=None):
        """Fit estimator training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_dimensions, series_length)
            Training time series instances to cluster
        y: None or 1D np.ndarray of shape (n_instances,)
            Training labels, passed only for classifiers or regressors

        Returns
        -------
        self: sktime estimator
            Fitted estimator.
        """
        tslearn_est = self._init_tslearn_object()

        # check if tslearn_est fit has y parameter
        # if yes, call with y, otherwise without
        tslearn_has_y = "y" in signature(tslearn_est.fit).parameters

        if tslearn_has_y:
            tslearn_est.fit(X, y)
        else:
            tslearn_est.fit(X)

        # write fitted params to self
        tslearn_fitted_params = self._get_fitted_params_default(tslearn_est)
        for k, v in tslearn_fitted_params.items():
            setattr(self, f"{k}_", v)

        return self

    def _predict(self, X, y=None):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances, n_dimensions, series_length))
            Time series instances to predict their cluster indexes.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        np.ndarray (1d array of shape (n_instances,))
            Index of the cluster each time series in X belongs to.
        """
        tslearn_est = getattr(self, self._estimator_attr)

        # check if tslearn_est fit has y parameter
        # if yes, call with y, otherwise without
        tslearn_has_y = "y" in signature(tslearn_est.predict).parameters

        if tslearn_has_y:
            return tslearn_est.predict(X, y)
        else:
            return tslearn_est.predict(X)
