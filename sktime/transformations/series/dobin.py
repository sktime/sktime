"""Preprocessing algorithm DOBIN (Distance based Outlier BasIs using Neighbors)."""

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from sktime.transformations.base import BaseTransformer

__author__ = ["KatieBuc"]
__all__ = ["DOBIN"]


class DOBIN(BaseTransformer):
    """Distance based Outlier BasIs using Neighbors (DOBIN).

    DOBIN is a pre-processing algorithm that constructs a set of basis
    vectors tailored for outlier detection as described by _[1]. DOBIN
    has a simple mathematical foundation and can be used as a dimension
    reduction tool for outlier detection tasks.

    Method assumes normalized data, the original R code implementation uses:
    ``from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler``
    This prevents variables with large variances having disproportional
    influence on Euclidean distances. The original implelemtation _[1] uses
    ``MinMaxScaler`` normalization, and removes NA values before normalization.

    We emphasize that DOBIN is not an outlier detection method; rather it is
    a pre-processing step that can be used by any outlier detection method.

    Parameters
    ----------
    frac : float (default=0.95)
        The cut-off quantile for Y space
        (parameter q in _[1]).
    k : int (default=None)
        Number of nearest neighbours considered
        (parameter k_2 on page 9 in _[1])

    Attributes
    ----------
    _basis : pd.DataFrame
        The basis vectors suitable for outlier detection
        (denoted as Theta in _[1]).
    _coords : pd.DataFrame
        The transformed coordinates of the data
        (denoted as tilde{X}, see equation 8 in _[1])

    References
    ----------
    .. [1] Kandanaarachchi, Sevvandi, and Rob J. Hyndman. "Dimension reduction
    for outlier detection using DOBIN." Journal of Computational and Graphical
    Statistics 30.1 (2021): 204-219.

    Examples
    --------
    >>> from sktime.transformations.series.dobin import DOBIN  # doctest: +SKIP
    >>> from sklearn.preprocessing import MinMaxScaler  # doctest: +SKIP
    >>> import numpy as np  # doctest: +SKIP
    >>> import pandas as pd  # doctest: +SKIP
    >>> from sktime.datasets import load_uschange  # doctest: +SKIP
    >>> _, X = load_uschange()  # doctest: +SKIP
    >>> scaler = MinMaxScaler()  # doctest: +SKIP
    >>> X = scaler.fit_transform(X)  # doctest: +SKIP
    >>> model = DOBIN()  # doctest: +SKIP
    >>> X_outlier = model.fit_transform(pd.DataFrame(X))  # doctest: +SKIP
    >>> X_outlier.head()  # doctest: +SKIP
            DB0       DB1       DB2       DB3
    0  1.151965  0.116488  0.286064  0.288140
    1  1.191976  0.100772  0.050835  0.225985
    2  1.221158  0.078031  0.034030  0.249676
    3  1.042420  0.188494  0.218460  0.205251
    4  1.224701  0.020028 -0.294705  0.199827
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "KatieBuc",
        "python_dependencies": "scipy",
        # estimator type
        # --------------
        "X_inner_mtype": "pd.DataFrame",
        "fit_is_empty": False,
        "skip-inverse-transform": True,
    }

    def __init__(
        self,
        frac=0.95,
        k=None,
    ):
        self.frac = frac
        self.k = k
        super().__init__()

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Series of mtype X_inner_mtype
            Data to be transformed
        y : Series of mtype y_inner_mtype, default=None
            Not required for this unsupervised transform.

        Returns
        -------
        self: reference to self
        """
        from scipy.linalg import null_space

        self._X = X

        assert all(X.apply(is_numeric_dtype, axis=0))

        n_obs, n_dim = X.shape

        if n_dim == 1:
            warnings.warn(
                "Warning: Input data X is univariate. For dimensionality reduction, "
                "please provide multivariate input.",
                stacklevel=2,
            )
            self._coords = X
            return self

        # if more dimensions than observations, change of basis to subspace
        if n_obs < n_dim:
            pca = PCA(n_components=n_obs)
            X = pca.fit_transform(X)
            self._X_pca = X
            _, n_dim = X.shape

        self.k_ = min(20, max(n_obs // 20, 2)) if self.k is None else self.k

        X_copy = X.copy()
        B = np.identity(n_dim)
        basis = pd.DataFrame()

        for _ in range(n_dim):
            # Compute Y space
            y_space = close_distance_matrix(X_copy, self.k_, self.frac)

            # Find eta
            w = y_space.apply(sum, axis=0)
            eta = np.array([w / np.sqrt(sum(w**2))])

            # If issues finding Y space (e.g. no variance in column)
            # get null space of basis
            if np.isnan(eta).any():
                basis_col = pd.DataFrame(null_space(basis.T))
                basis = pd.concat([basis, basis_col], axis=1)
                break

            # Update basis
            basis_col = pd.DataFrame(np.dot(B, eta.T))
            basis = pd.concat([basis, basis_col], axis=1)

            # Find xperp
            xperp = X_copy - np.dot(np.dot(np.array(X_copy), eta.T), eta)

            # Find a basis B for xperp
            B1 = null_space(eta)

            # Change xperp coordinates to B basis
            X_copy = np.dot(xperp, B1)

            # Update B with B1, each time 1 dimension is reduced
            B = np.dot(B, B1)

        # new coordinates
        coords = pd.DataFrame(
            X.dot(np.array(basis))
        )  # convert np.array, error if both pd.DataFrames and rownames != colnames

        basis.columns = ["".join(["DB", str(i)]) for i in range(len(basis.columns))]
        self._basis = basis
        coords.columns = ["".join(["DB", str(i)]) for i in range(len(coords.columns))]
        self._coords = coords

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series of mtype X_inner_mtype
            Data to be transformed
        y : Series of mtype y_inner_mtype, default=None
            Not required for this unsupervised transform.

        Returns
        -------
        transformed version of X, representing the original data on a new set of
        coordinates, obtained by multiplying input data by the basis vectors.
        """
        # fit again if indices not seen, but don't store anything
        if not X.index.equals(self._X.index):
            X_full = X.combine_first(self._X)
            new_dobin = DOBIN(
                frac=self.frac,
                k=self.k,
            ).fit(X_full)
            warnings.warn(
                "Warning: Input data X differs from that given to fit(). "
                "Refitting with new input data, not storing updated public class "
                "attributes. For this, explicitly use fit(X) or fit_transform(X).",
                stacklevel=2,
            )
            return new_dobin._coords

        return self._coords


def close_distance_matrix(X: npt.ArrayLike, k: int, frac: float):
    """Calculate distance between close pairs.

    Parameters
    ----------
    X : np.ArrayLike
        Data to be transformed
    k : int
        Number of nearest neighbours considered. If k = None, it is empirically
        derived as ``min(0.05 * number of observations, 20)``

    Returns
    -------
    pd.DataFrame of pairs of close neighbour indices
    """
    X = pd.DataFrame(X)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    _, indices = nbrs.kneighbors(X)

    dist = pd.DataFrame(
        [
            ((X.iloc[i,] - X.iloc[j,]) ** 2).tolist()
            for (i, j) in zip(
                np.repeat(indices[:, 0], repeats=k), indices[:, 1:].flatten()
            )
        ]
    )

    row_sums = dist.apply(sum, axis=1)
    q_frac = np.quantile(row_sums, q=frac)

    mask = row_sums > q_frac

    return dist.loc[mask,]
