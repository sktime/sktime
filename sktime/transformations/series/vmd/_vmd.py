"""Variational Mode Decomposition transformer."""

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.transformations.series.vmd._vmdpy import VMD

__author__ = ["DaneLyttinen", "vcarvo"]


class VmdTransformer(BaseTransformer):
    """Variational Mode Decomposition transformer.

    An implementation of the Variational Mode Decomposition method (2014) [1]_,
    based on the ``vmdpy`` package [3]_ by ``vcarvo``, which in turn is based
    on the original MATLAB implementation by Dragomiretskiy and Zosso [1]_.

    This transformer is the official continuation of the ``vmdpy`` package,
    maintained in ``sktime``.

    VMD is an decomposition (series-to-series) transformer which uses the Variational
    Mode Decomposition method to decompose an original time series into
    multiple Intrinsic Mode Functions.
    The number of Intrinsic Mode Functions created depend on the K parameter and should
    be optimally defined if known.
    If the K parameter is unknown, this transformer will attempt to find
    a good estimate of it by comparing the
    original time series against the reconstruction of the signals using
    the energy loss coefficient, default of 0.01.

    This is useful if you have a complex series and want to decompose it
    into easier-to-learn IMF's,
    which when summed together make up an estimate of the original time series with
    some loss of information.

    References
    ----------
    .. [1] K. Dragomiretskiy and D. Zosso, - Variational Mode Decomposition:
        IEEE Transactions on Signal Processing, vol. 62, no. 3, pp. 531-544, Feb.1,
        2014, doi: 10.1109/TSP.2013.2288675.
    .. [2] Vinícius R. Carvalho, Márcio F.D. Moraes, Antônio P. Braga,
        Eduardo M.A.M. Mendes -
        Evaluating five different adaptive decomposition methods for EEG signal
        seizure detection and classification,
        Biomedical Signal Processing and Control, Volume 62, 2020,
        102073, ISSN 1746-8094
        https://doi.org/10.1016/j.bspc.2020.102073.
    .. [3] https://github.com/vrcarva/vmdpy

    Parameters
    ----------
    K : int, optional (default='None')
        the number of Intrinsic Mode Functions to decompose original series to.
        If None, will decompose the series iteratively until kMax is reached or the sum
        of the decomposed modes against the original series is less than
        the ``energy_loss_coefficient`` parameter.
    kMax : int, optional (default=30)
        the limit on the number of Intrinsic Mode Functions to
        decompose the original series to
        if the ``energy_loss_coefficient`` hasn't been reached.
        Only used if ``K`` is ``None``, ignored otherwise.
    energy_loss_coefficient : int, optional (default=0.01)
        decides the acceptable loss of information from the
        original series when the decomposed modes are summed together
        as calculated by the energy loss coefficient.
        Only used if ``K`` is ``None``, ignored otherwise.
    alpha : int, optional (default=2000)
        bandwidth constraint for the generated Intrinsic Mode Functions,
        balancing parameter of the data-fidelity constraint
    tau : int, optional (default=0.)
        noise tolerance of the generated modes, time step of dual ascent
    DC : int, optional (default=0)
        Imposed DC parts
    init : int, optional (default=1)
        parameter for omegas, default of one will initialize the omegas uniformly,
        1 = all omegas initialized uniformly
        0 = all omegas start at 0,
        2 = all omegas are initialized at random
    tol : int, optional (default=1e-7)
        convergence tolerance criterion

    Examples
    --------
    >>> from sktime.transformations.series.vmd import VmdTransformer  # doctest: +SKIP
    >>> from sktime.datasets import load_solar  # doctest: +SKIP
    >>> y = load_solar()  # doctest: +SKIP
    >>> transformer = VmdTransformer()  # doctest: +SKIP
    >>> modes = transformer.fit_transform(y)  # doctest: +SKIP

    VmdTransformer can be used in a forecasting pipeline,
    to decompose, forecast individual components, then recompose:
    >>> from sktime.forecasting.trend import TrendForecaster  # doctest: +SKIP
    >>> pipe = VmdTransformer() * TrendForecaster()  # doctest: +SKIP
    >>> pipe.fit(y, fh=[1, 2, 3])  # doctest: +SKIP
    >>> y_pred = pipe.predict()  # doctest: +SKIP
    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "scitype:transform-labels": "None",
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "None",
        "univariate-only": False,
        "requires_y": False,
        "remember_data": False,
        "fit_is_empty": False,
        "X-y-must-have-same-index": False,
        "enforce_index_type": None,
        "transform-returns-same-time-index": False,
        "capability:inverse_transform": True,
        "capability:inverse_transform:exact": False,
        "skip-inverse-transform": False,
        "capability:unequal_length": False,
        "capability:unequal_length:removes": False,
        "handles-missing-data": False,
        "capability:missing_values:removes": False,
    }

    def __init__(
        self,
        K=None,
        kMax=30,
        alpha=2000,
        tau=0.0,
        DC=0,
        init=1,
        tol=1e-7,
        energy_loss_coefficient=0.01,
    ):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.DC = DC
        self.init = init
        self.tol = tol
        self.kMax = kMax
        self.energy_loss_coefficient = energy_loss_coefficient
        self.fit_column_names = None

    def _inverse_transform(self, X, y=None):
        row_sums = X.sum(axis=1)
        row_sums.columns = self.fit_column_names
        return row_sums

    def _fit(self, X, y=None):
        if self.K is None:
            self._K = self.__runVMDUntilCoefficientThreshold(X.to_numpy())
        else:
            self._K = self.K
        self.fit_column_names = X.columns
        return self

    def _transform(self, X, y=None):
        # Package truncates last if odd, so make even
        # through duplication then remove duplicate
        values = X.values
        if len(values) % 2 == 1:
            values = np.append(values, values[-1])
        u, u_hat, omega = VMD(
            values, self.alpha, self.tau, self._K, self.DC, self.init, self.tol
        )
        transposed = u.T
        if len(transposed) != len(X.values):
            transposed = transposed[:-1]
        Y = pd.DataFrame(transposed)
        return Y

    def __runVMDUntilCoefficientThreshold(self, data):
        K = 1
        data = data.flatten()
        if len(data) % 2 == 1:
            data = np.append(data, data[-1])
        while K < self.kMax:
            u, u_hat, omega = VMD(
                data, self.alpha, self.tau, K, self.DC, self.init, self.tol
            )
            reconstruct = sum(u)
            energy_loss_coef = np.linalg.norm(
                (data - reconstruct), 2
            ) ** 2 / np.linalg.norm(data, 2)
            if energy_loss_coef > self.energy_loss_coefficient:
                K += 1
                continue
            else:
                break
        return K
