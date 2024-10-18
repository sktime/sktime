"""Variational Mode Decomposition transformer."""

import numpy as np
import pandas as pd

from sktime.libs.vmdpy import VMD
from sktime.transformations.base import BaseTransformer

__author__ = ["DaneLyttinen", "vrcarva"]


class VmdTransformer(BaseTransformer):
    """Variational Mode Decomposition transformer.

    An implementation of the Variational Mode Decomposition method (2014) [1]_,
    based on the ``vmdpy`` package [3]_ by ``vrcarva``, which in turn is based
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
        If None, will decompose the series iteratively with increasing K,
        until kMax is reached or the sum
        of the decomposed modes against the original series is less than
        the ``energy_loss_coefficient`` parameter (whichever occurs earlier).
        In this case, the lowest K to satisfy one of the condition
        is used in ``transform``.
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
    returned_decomp : bool, optional (default="u")
        which decomposition object is returned by ``transform``

        * ``"u"``: the decomposed modes
        * ``"u_hat"``: the mode spectra (absolute values)
        * ``"u_both"``: both the decomposed modes and the mode spectra,
          these will be returned column concatenated, first the modes then the spectra

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
        # packaging info
        # --------------
        "authors": ["DaneLyttinen", "vrcarva"],
        "maintainers": ["DaneLyttinen", "vrcarva"],
        # estimator type
        # --------------
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
        returned_decomp="u",
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
        self.returned_decomp = returned_decomp
        self.fit_column_names = None

    def _inverse_transform(self, X, y=None):
        row_sums = X.sum(axis=1)
        row_sums.columns = self.fit_column_names
        return row_sums

    def _fit(self, X, y=None):
        if self.K is None:
            K = self.__runVMDUntilCoefficientThreshold(X.to_numpy())
        else:
            K = self.K
        self.fit_column_names = X.columns
        self.K_ = K
        return self

    def _transform(self, X, y=None):
        return_dec = self.returned_decomp
        # Package truncates last if odd, so make even
        # through duplication then remove duplicate
        values = X.values
        if len(values) % 2 == 1:
            values = np.append(values, values[-1])
        u, u_hat, omega = VMD(
            values, self.alpha, self.tau, self.K_, self.DC, self.init, self.tol
        )
        if return_dec in ["u", "u_both"]:
            transposed = u.T
            if len(transposed) != len(X.values):
                transposed = transposed[:-1]
            u_return = pd.DataFrame(transposed)
        if return_dec in ["u_hat", "u_both"]:
            u_hat_return = pd.DataFrame(np.abs(u_hat))
        if return_dec == "omega":
            omega_return = pd.DataFrame(omega)

        if return_dec == "u":
            return u_return
        elif return_dec == "u_hat":
            return u_hat_return
        elif return_dec == "omega":
            return omega_return
        elif return_dec == "u_both":
            u_returns = pd.concat([u_return, u_hat_return], axis=1)
            u_returns.columns = pd.RangeIndex(len(u_returns.columns))
            return u_returns
        else:
            raise ValueError(
                "Error in VmdTransformer: "
                f"Unknown return_decomp parameter: {return_dec}"
            )

    def __runVMDUntilCoefficientThreshold(self, data):
        K = 1
        data = data.flatten()
        if len(data) % 2 == 1:
            data = np.append(data, data[-1])
        while K < self.kMax:
            u, _, _ = VMD(data, self.alpha, self.tau, K, self.DC, self.init, self.tol)
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

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params0 = {"kMax": 4}
        params1 = {"K": 3, "returned_decomp": "u_hat"}
        params2 = {"kMax": 3, "energy_loss_coefficient": 0.1}
        params3 = {"K": 3, "returned_decomp": "u_both", "alpha": 1000}

        return [params0, params1, params2, params3]
