import numpy as np
import pandas as pd
from vmdpy import VMD
from sktime.transformations.base import BaseTransformer


class VmdTransformer(BaseTransformer):
    """Variational Mode Decomposition transformer.

    An implementation of the Variational Mode Decomposition method.
    Uses the vmdpy package as the implementation of the transformer as described in the works below
    Vinícius R. Carvalho, Márcio F.D. Moraes, Antônio P. Braga, Eduardo M.A.M. Mendes, Evaluating five different adaptive decomposition methods for EEG signal seizure detection and classification, Biomedical Signal Processing and Control, Volume 62, 2020, 102073, ISSN 1746-8094, https://doi.org/10.1016/j.bspc.2020.102073.

    This transformer will take a single series as input, and decompose the series into multiple Intrinsic Mode Function, the number of Instrinsic Mode Functions created depend on the K parameter.
    This is useful if you have a complex series and want to decompose it into easier-to-learn IMF's, which when summed together make up an estimate of the original time series with some loss of information.

    References
    ----------
    .. [1] K. Dragomiretskiy and D. Zosso, - Variational Mode Decomposition:
        IEEE Transactions on Signal Processing, vol. 62, no. 3, pp. 531-544, Feb.1, 2014, doi: 10.1109/TSP.2013.2288675.
    .. [2] Vinícius R. Carvalho, Márcio F.D. Moraes, Antônio P. Braga, Eduardo M.A.M. Mendes -
        Evaluating five different adaptive decomposition methods for EEG signal seizure detection and classification, Biomedical Signal Processing and Control, Volume 62, 2020, 102073, ISSN 1746-8094
        https://doi.org/10.1016/j.bspc.2020.102073.
    .. [3] https://github.com/vrcarva/vmdpy

    Parameters
    ----------
    K : int, optional (default='None')
        the number of Intrinsic Mode Functions to decompose original series to. If None, will decompose the series iteratively until kMax is reached or the sum of the decomposed modes against the original series is less than the energy_loss_coefficient parameter.
    kMax : int, optional (default=30)
        the limit on the number of Intrinsic Mode Functions to decompose the original series to if the energy_loss_coefficient hasn't been reached. Only used if K is None.
    energy_loss_coefficient : int, optional (default=0.01)
        decides the acceptable loss of information from the original series when the decomposed modes are summed together as calculated by the energy loss coefficient. Only used if K is None
    alpha : int, optional (default=2000)
        bandwidth constraint for the generated Intrinsic Mode Functions.
    tau : int, optional (default=0.)
        noise tolerance of the generated modes
    DC : int, optional (default=0)
        Imposed DC parts
    init : int, optional (default=1)
        parameter for omegas, default of one will initialize the omegas uniformly
    tol : int, optional (default=1e-7)
        parameter for omegas, default of one will initialize the omegas uniformly

    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
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
        "skip-inverse-transform": True,
        "capability:unequal_length": False,
        "capability:unequal_length:removes": False,
        "handles-missing-data": False,
        "capability:missing_values:removes": False,
        "python_version": None,
        "python_dependencies": ['vmdpy', 'numpy', 'pandas']
    }

    def __init__(self, K=None, kMax=30, alpha=2000, tau=0., DC=0, init=1, tol=1e-7, energy_loss_coefficient=0.01):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.tau = tau  # noise-tolerance (no strict fidelity enforcement)
        self.DC = DC  # no DC part imposed
        self.init = init  # initialize omegas uniformly
        self.tol = tol
        self.kMax = kMax
        self.energy_loss_coefficient = energy_loss_coefficient

    def _inverse_transform(self, X, y=None):
        row_sums = X.sum(axis=1)
        return row_sums

    def _fit(self, X, y=None):
        if (self.K == None):
            self.K = self.__runVMDUntilCoefficientThreshold(X.to_numpy())
        return self

    def _transform(self, X, y=None):
        u, u_hat, omega = VMD(X.values, self.alpha, self.tau, self.K, self.DC, self.init, self.tol)
        transposed = u.T
        if (len(u.T[0]) != len(X.values)):
            last_row = transposed[-1, :]
            transposed = np.vstack([transposed, last_row])
        Y = pd.DataFrame(transposed)
        return Y

    def __runVMDUntilCoefficientThreshold(self, data):
        K = 1
        data = data.flatten()
        while K < self.kMax:
            u, u_hat, omega = VMD(data, self.alpha, self.tau, K, self.DC, self.init, self.tol)
            reconstruct = sum(u)
            originalData = data
            ## The package truncates the last odd data point in series
            if (len(data) % 2 == 1):
                originalData = data[:-1]
            energy_loss_coef = (np.linalg.norm((originalData - reconstruct), 2) ** 2 / np.linalg.norm(originalData, 2))
            if (energy_loss_coef > self.energy_loss_coefficient):
                K += 1
                continue
            else:
                break
        return K