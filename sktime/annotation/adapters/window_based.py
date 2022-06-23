# -*- coding: utf-8 -*-
"""Window Based change point detector.

Change point detection algorithm outputting change points.
Adapter class for external library ruptures :
ruptures.detection.window.Window
"""


__author__ = ["lielleravid", "NoaBenAmi"]
__all__ = ["WindowBasedChangePoint"]

from sktime.annotation.base import BaseSeriesAnnotator
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("ruptures", severity="warning")


class WindowBasedChangePoint(BaseSeriesAnnotator):
    """Window based change point detection algorithm.

    Window-based change point detection is used to perform fast segmentation.
    The algorithm uses two windows which slide along the data stream.
    The statistical properties of the data within each window are compared with
    a discrepancy measure. The discrepancy measure is calculated using a
    cost function.

    Parameters
    ----------
    width : int, optional (default = 100)
        sliding window length
    model : string, optional (default = "l2")
        Segmentation cost function.
        Must be one of the following options ["l1", "l2", "rbf"].
        where l1 - Least absolute deviation
        l2 -Least squared deviation
        rbf - radial basis function kernel
        This param will not be used if 'custom_cost' param is not None
    custom_cost : ruptures.BaseCost, optional (default = None)
        custom cost function, if the 'model' param options are not relevant
        must be an object that inherits from ruptures.BaseCost class
    min_size : int, optional (default = 2)
        minimum segment length
    jump : int, optional (default = 5)
        subsample, search for change points within every *jump* points
        The higher 'jump' is, the faster the prediction is achieved. This can be
        at the expense of precision.
    params : dict, optional (default = None)
        a dictionary of parameters for the custom cost instance
    n_changepoints : int, optional (default = None)
        number of breakpoints to find before stopping the algorithm
        for predict method to compute one of 'n_changepoints',
        'penalty', 'epsilon' must be set.
    penalty : float, optional (default = None)
        stopping rule, threshold for sum of costs of change points before and after
        last change point detected. if that difference is bigger than penalty, stop
        computing change points
        must be > 0
        for predict method to compute one of 'n_changepoints',
        'penalty', 'epsilon' must be set.
    epsilon : float, optional (default = None)
        stopping rule, threshold for sum of costs of change points detected
        if sum of costs of change points is bigger than 'epsilon', stop computing
        must be > 0
        for predict method to compute one of 'n_changepoints',
        'penalty', 'epsilon' must be set.

    Components
    ----------
    estimator_prefit : ruptures.detection.window.Window,
        ruptures.BaseEstimator descendant
        estimator for window based changepoint detection, used before fit method
    estimator_postfit_ : ruptures.detection.window.Window,
        ruptures.BaseEstimator descendant
        estimator for window based changepoint detection, used after fit method
    """

    def __init__(
        self,
        width=100,
        model="l2",
        custom_cost=None,
        min_size=2,
        jump=5,
        params=None,
        n_changepoints=None,
        penalty=None,
        epsilon=None,
    ):
        # estimators should precede parameters
        #  if estimators have default values, set None and initalize below
        _check_soft_dependencies("ruptures", severity="error", object=self)
        # self.estimator_prefit = None
        # self.estimator_postfit_ = None
        self.estimator = None
        self.width = width
        self.model = model
        self.custom_cost = custom_cost
        self.min_size = min_size
        self.jump = jump
        self.params = params
        self.n_changepoints = n_changepoints
        self.penalty = penalty
        self.epsilon = epsilon

        # important: no checking or other logic should happen here
        super(WindowBasedChangePoint, self).__init__(fmt="sparse", labels="score")

    def _fit(self, X, Y=None):
        """
        Calculate parameters for change point detection.

        Parameters
        ----------
        X : array of shape (n_samples,) or (n_samples, n_features).
            data to find change points within
        Y : ignored argument for interface compatibility

        Returns
        -------
        self : reference to self
        """
        # from ruptures import Window
        # if self.estimator_prefit is None:
        #    self.estimator_prefit = Window(width, model, custom_cost,
        #    min_size, jump, params)
        # self.estimator_postfit_ = self.estimator_prefit.fit(X)
        return self

    def _predict(self, X):
        """
        Find and return the change points detected.

        Must be called after the fit method, the change points are found within
        the data given at fit method.
        The stopping rule depends on the parameters given in constructor
        'n_changepoints', 'penalty', 'epsilon'.

        Parameters
        ----------
        X : ignored argument for interface compatibility

        Raises
        ------
        AssertionError : if the parameters 'n_changepoints',
                'penalty', 'epsilon' given in constructor
                are all None
        BadSegmentationParameters: in case of impossible segmentation
                configuration

        Returns
        -------
        Y : sorted list of change points detected
        """
        # return self.estimator_postfit_.predict(self.n_changepoints,
        # self.penalty, self.epsilon)
        from ruptures import Window

        self.estimator = Window(
            self.width,
            self.model,
            self.custom_cost,
            self.min_size,
            self.jump,
            self.params,
        )
        try:
            return self.estimator.fit_predict(
                X, self.n_changepoints, self.penalty, self.epsilon
            )
        except AssertionError:
            raise Exception(
                "No stopping rule given: 'n_changepoints', "
                "'penalty' or 'epsilon' must not be None"
            )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for annotators.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = {"n_changepoints": 10}
        return params

    def _predict_scores(self, X):
        pass
