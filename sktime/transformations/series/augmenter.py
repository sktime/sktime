# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Series transformers for time series augmentation."""

__author__ = ["MrPr3ntice", "MFehsenfeld", "iljamaurer"]
__all__ = [
    "WhiteNoiseAugmenter",
    "ReverseAugmenter",
    "InvertAugmenter",
    "RandomSamplesAugmenter",
]


import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from sktime.transformations.base import BaseTransformer
from sktime.utils.validation.series import check_series


class _AugmenterTags:
    _tags = {
        # packaging info
        # ----------------
        "authors": ["MrPr3ntice", "MFehsenfeld", "iljamaurer"],
        "maintainers": ["MrPr3ntice", "MFehsenfeld", "iljamaurer"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,
        "capability:missing_values": False,
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "X-y-must-have-same-index": False,
        "fit_is_empty": True,
        "transform-returns-same-time-index": False,
        "capability:inverse_transform": False,
    }


class WhiteNoiseAugmenter(BaseTransformer):
    """
    Augmenter that adds Gaussian (white) noise to a univariate or multivariate time series.

    Internally, this always works on a NumPy array. If the user passes a pd.DataFrame,
    we convert it to np.ndarray, add noise, then wrap it back into a DataFrame with the
    same index and columns.

    Parameters
    ----------
    scale : float, default=1.0
        Standard deviation of the Gaussian noise to add.
    random_state : int or np.random.RandomState or None, default=42
        If int or RandomState, use it for drawing the random variates.
        If None, use a new RandomState internally.
    """

    _tags = {
        "scitype:transform-output": "Series",  # outputs same mtype as input
        "capability:inverse_transform": False,
    }

    def __init__(self, scale: float = 1.0, random_state: int | np.random.RandomState | None = 42):
        super().__init__()
        self.scale = scale
        self.random_state = random_state
        self._is_fitted = False
        self._rng = None

    def fit(self, X, y=None):
        """
        Fit does nothing except initialize the random number generator and
        check that X is a valid univariate/multivariate series.

        Parameters
        ----------
        X : np.ndarray or pd.Series or pd.DataFrame
            Input series (univariate or multivariate). If DataFrame, columns are vars.
        y : ignored
        """
        # Validate input series (accepts np.ndarray, pd.Series, or pd.DataFrame)
        _ = check_series(X, enforce_univariate=False)

        # Initialize RNG
        if isinstance(self.random_state, (int, np.integer)):
            self._rng = np.random.RandomState(self.random_state)
        elif isinstance(self.random_state, np.random.RandomState):
            self._rng = self.random_state
        else:
            # None or anything else → new RandomState
            self._rng = np.random.RandomState()
        self._is_fitted = True
        return self

    def _check_fitted(self):
        if not self._is_fitted:
            raise ValueError(f"{self.__class__.__name__} has not been fitted yet. Call `fit` first.")

    def transform(self, X, y=None):
        """
        Add Gaussian noise to X. Works for np.ndarray, pd.Series, or pd.DataFrame.

        Parameters
        ----------
        X : np.ndarray or pd.Series or pd.DataFrame
            Time series to augment.
        y : ignored

        Returns
        -------
        X_noisy : same type as X (np.ndarray, pd.Series, or pd.DataFrame)
            The input series plus white noise ~ N(0, scale^2).
        """
        self._check_fitted()

        # Validate input
        _ = check_series(X, enforce_univariate=False)

        # Detect if it was a DataFrame (to re‐wrap later)
        was_dataframe = isinstance(X, pd.DataFrame)
        was_series = isinstance(X, pd.Series)

        # Obtain raw NumPy array
        if was_dataframe:
            X_vals = X.values
        elif was_series:
            # make it a 2D array to handle univariate series uniformly
            idx = X.index
            X_vals = X.to_numpy().reshape(-1, 1)
        else:
            # Already an ndarray (1D or 2D)
            X_vals = np.asarray(X)
            if X_vals.ndim == 1:
                X_vals = X_vals.reshape(-1, 1)

        # Generate Gaussian noise of same shape
        noise = self._rng.normal(loc=0.0, scale=self.scale, size=X_vals.shape)

        # Add noise
        X_noisy_vals = X_vals + noise

        # Convert back to original type
        if was_dataframe:
            return pd.DataFrame(
                X_noisy_vals,
                index=X.index,
                columns=X.columns,
            )
        elif was_series:
            # Collapse back to a Series
            return pd.Series(X_noisy_vals.ravel(), index=idx, name=X.name)
        else:
            # Original was np.ndarray: return same shape
            return X_noisy_vals if X_noisy_vals.ndim > 1 else X_noisy_vals.ravel()

    def fit_transform(self, X, y=None):
        """
        Just calls fit and then transform on X.
        """
        return self.fit(X, y).transform(X)


class ReverseAugmenter(_AugmenterTags, BaseTransformer):
    r"""Augmenter reversing the time series.

    If ``transform`` is given a time series :math:`X={x_1, x_2, ... , x_n}`, then
    returns :math:`X_t={x_n, x_{n-1}, ..., x_2, x_1}`.
    Time series augmentation by reversing has been discussed e.g. in [1].

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.transformations.series.augmenter import ReverseAugmenter
    >>> X = pd.Series([1,2,3,4,5])
    >>> augmenter = ReverseAugmenter()
    >>> Xt = augmenter.fit_transform(X)
    >>> Xt
    0    5
    1    4
    2    3
    3    2
    4    1
    dtype: int64

    References and Footnotes
    ----------

        [1]: IWANA, Brian Kenji; UCHIDA, Seiichi. An empirical survey of data
        augmentation for time series classification with neural networks. Plos
        one, 2021, 16. Jg., Nr. 7, S. e0254841.
    """

    def __init__(self):
        super().__init__()

    def _transform(self, X, y=None):
        return X.loc[::-1].reset_index(drop=True, inplace=False)


class InvertAugmenter(_AugmenterTags, BaseTransformer):
    r"""Augmenter inverting the time series by multiplying it by -1.

    If ``transform`` is given a time series :math:`X={x_1, x_2, ... , x_n}`, then
    returns :math:`X_t={-x_1, -x_2, ... , -x_n}`.

    Examples
    --------
    >>> from sktime.transformations.series.augmenter import InvertAugmenter
    >>> import pandas as pd
    >>> X = pd.Series([1,2,3,4,5])
    >>> augmenter = InvertAugmenter()
    >>> Xt = augmenter.fit_transform(X)
    >>> Xt
    0   -1
    1   -2
    2   -3
    3   -4
    4   -5
    dtype: int64
    """

    def __init__(self):
        super().__init__()

    def _transform(self, X, y=None):
        return X.mul(-1)


class RandomSamplesAugmenter(_AugmenterTags, BaseTransformer):
    r"""Draw random samples from time series.

    ``transform`` takes a time series :math:`X={x_1, x_2, ... , x_m}` with :math:`m`
    elements and returns :math:`X_t={x_i, x_{i+1}, ... , x_n}`, where
    :math:`{x_i, x_{i+1}, ... , x_n}` are :math:`n`=``n`` random samples drawn
    from :math:`X` (with or ``without_replacement``).

    Parameters
    ----------
    n: int or float, optional (default = 1.0)
            To specify an exact number of samples to draw, set `n` to an int value.
            Number of samples to draw.
            To specify the returned samples as a proportion of the given times series
            set `n` to a float value :math:`n \in [0, 1]`.
            By default, the same number of samples is returned as given by the input
            time series.
    without_replacement: bool, optional (default = True)
            Whether to draw without replacement. If True, every sample of the input
            times series `X` will appear at most once in ``Xt``.
    random_state: None or int or ``np.random.RandomState`` instance, optional
            "If int or RandomState, use it for drawing the random variates.
            If None, rely on ``self.random_state``.
            Default is None." [1]

    References and Footnotes
    ----------

        [1]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.random_state.html
    """  # noqa: E501

    def __init__(
        self,
        n=1.0,
        without_replacement=True,
        random_state=42,
    ):
        if isinstance(n, float):
            if n <= 0.0 or not np.isfinite(n):
                raise ValueError("n must be a positive, finite number.")
        elif isinstance(n, int):
            if n < 1 or not np.isfinite(n):
                raise ValueError("n must be a finite number >= 1.")
        else:
            raise ValueError("n must be int or float, not " + str(type(n))) + "."
        self.n = n
        self.without_replacement = without_replacement
        self.random_state = random_state
        super().__init__()

    def _transform(self, X, y=None):
        if isinstance(self.n, float):
            n = int(np.ceil(self.n * len(X)))
        else:
            n = self.n
        rng = check_random_state(self.random_state)
        values = np.concatenate(X.values)
        if self.without_replacement:
            replace = False
        else:
            replace = True
        Xt = rng.choice(values, n, replace)
        return pd.DataFrame(Xt)
