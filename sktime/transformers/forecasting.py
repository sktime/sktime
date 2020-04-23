import numpy as np
import pandas as pd
from sktime.utils.validation import check_is_fitted
from statsmodels.tsa.seasonal import seasonal_decompose

from sktime.transformers.base import BaseTransformer
from sktime.transformers.compose import Tabulariser
from sktime.utils.data_container import get_time_index
from sktime.utils.time_series import fit_trend, remove_trend, add_trend
from sktime.utils.validation.forecasting import validate_sp, check_is_fitted_in_transform
from sktime.utils.validation.supervised import validate_X, check_X_is_univariate


class Deseasonaliser(BaseTransformer):
    """A transformer that removes a seasonal component from time series/panel data

    Parameters
    ----------
    sp : int, optional (default=1)
        Seasonal periodicity
    model : str {'additive', 'multiplicative'}, optional (default='additive')
        Model to use for estimating seasonal component
    check_input : bool, optional (default=True)
        When set to ``True``, inputs will be validated, otherwise inputs are assumed to be valid
        and no checks are performed. Use with caution.
    """

    def __init__(self, sp=1, model='additive', check_input=True):
        self.sp = validate_sp(sp)
        allowed_models = ('additive', 'multiplicative')
        if model in allowed_models:
            self.model = model
        else:
            raise ValueError(f"Allowed models are {allowed_models}, but found: {model}")
        self.check_input = check_input

        self._time_index = None
        self._input_shape = None

    def transform(self, X, y=None):
        """Transform X.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_samples, n_features]
            Nested dataframe with time-series in cells.

        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame with same number of rows and one column for each generated interval.
        """
        if self.check_input:
            validate_X(X)
            if X.shape[1] > 1:
                raise NotImplementedError(f"Currently does not work on multiple columns, make use of ColumnTransformer "
                                          f"instead")

        self._input_shape = X.shape

        # when seasonal periodicity is equal to 1 return X unchanged
        if self.sp == 1:
            return X

        # keep time index as transform/inverse transform depends on it, e.g. to carry forward trend in inverse_transform
        self._time_index = get_time_index(X)

        # convert into tabular format
        tabulariser = Tabulariser()
        Xs = tabulariser.transform(X.iloc[:, :1])

        check_is_fitted(self, 'is_fitted_')
        validate_X(X)

        # fit seasonal decomposition model
        seasonal_components = self._fit_seasonal_decomposition_model(Xs)

        # remove seasonal components from data
        if self.model == 'additive':
            Xt = Xs - seasonal_components
        else:
            Xt = Xs / seasonal_components

        # keep fitted seasonal components for inverse transform, they are repeated after the first seasonal
        # period so we only keep the components for the first seasonal period
        self.seasonal_components_ = seasonal_components[:, :self.sp]

        # convert back into nested format
        Xt = tabulariser.inverse_transform(pd.DataFrame(Xt))
        Xt.columns = X.columns
        return Xt

    def inverse_transform(self, X, y=None):
        """Inverse transform X

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_samples, n_features]
            Nested dataframe with time-series in cells.

        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame with same number of rows and one column for each generated interval.
        """

        if self.check_input:
            validate_X(X)
            check_X_is_univariate(X)

        # check that number of samples are the same, inverse transform depends on parameters fitted in transform and
        # hence only works on data with the same (number of) rows
        if not X.shape[0] == self._input_shape[0]:
            raise ValueError(f"Inverse transform only works on data with the same number samples "
                             f"as seen during transform, but found: {X.shape[0]} samples "
                             f"!= {self._input_shape[0]} samples (seen during transform)")

        # if the seasonal periodicity is 1, return unchanged X
        sp = self.sp
        if sp == 1:
            return X

        # check if seasonal decomposition model has been fitted in transform
        check_is_fitted_in_transform(self, 'seasonal_components_')

        # check if time index is aligned with time index seen during transform
        time_index = get_time_index(X)

        # align seasonal components with index of X
        if self._time_index.equals(time_index):
            # if time index is the same as used for fitting seasonal components, simply expand it to the size of X
            seasonal_components = self.seasonal_components_

        else:
            # if time index is not aligned, make sure to align fitted seasonal components to new index
            seasonal_components = self._align_seasonal_components_to_index(time_index)

        # expand or shorten aligned seasonal components to same size as X
        n_obs = len(time_index)
        if n_obs > sp:
            n_tiles = np.int(np.ceil(n_obs / sp))
            seasonal_components = np.tile(seasonal_components, n_tiles)
        seasonal_components = seasonal_components[:, :n_obs]

        # convert into tabular format
        tabulariser = Tabulariser()
        Xs = tabulariser.transform(X.iloc[:, :1])

        # inverse transform data
        if self.model == 'additive':
            Xit = Xs + seasonal_components
        else:
            Xit = Xs * seasonal_components

        # convert back into nested format
        Xit = tabulariser.inverse_transform(pd.DataFrame(Xit))
        Xit.columns = X.columns
        return Xit

    def _align_seasonal_components_to_index(self, time_index):
        """Helper function to align seasonal components with new time series index"""
        # find out by how much we have to shift seasonal_components to align with new index
        shift = -time_index[0] % self.sp

        # align seasonal components with new starting point of new time_index
        return np.roll(self.seasonal_components_, shift=shift, axis=1)

    def _fit_seasonal_decomposition_model(self, X):
        """Fit seasonal decopmosition model and return fitted seasonal components"""
        # statsmodels `seasonal_decompose` expects time series to be in columns, rather than rows, we therefore need to
        # transpose X here
        res = seasonal_decompose(X.values.T, model=self.model, period=self.sp, filt=None, two_sided=True,
                                 extrapolate_trend=0)
        seasonal_components = res.seasonal.T
        return np.atleast_2d(seasonal_components)


class Detrender(BaseTransformer):
    """A transformer that removes trend of given polynomial order from time series/panel data

    Parameters
    ----------
    order : int
        Polynomial order, zero: mean, one: linear, two: quadratic, etc
    check_input : bool, optional (default=True)
        When set to ``True``, inputs will be validated, otherwise inputs are assumed to be valid
        and no checks are performed. Use with caution.
    """

    def __init__(self, order=0, check_input=True):

        if not (isinstance(order, int) and (order >= 0)):
            raise ValueError(f"order must be a positive integer, but found: {type(order)}")
        self.order = order
        self.check_input = check_input
        self._time_index = None
        self._input_shape = None

    def transform(self, X, y=None):
        """Transform X.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_samples, n_features]
            Nested dataframe with time-series in cells.

        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame with same number of rows and one column for each generated interval.
        """

        if self.check_input:
            if not isinstance(X, pd.DataFrame):
                raise ValueError(f"Input must be pandas DataFrame, but found: {type(X)}")

        if X.shape[1] > 1:
            raise NotImplementedError(f"Currently does not work on multiple columns")

        self._input_shape = X.shape

        # keep time index as trend depends on it, e.g. to carry forward trend in inverse_transform
        self._time_index = get_time_index(X)

        # convert into tabular format
        tabulariser = Tabulariser()
        Xs = tabulariser.transform(X.iloc[:, :1])

        # fit polynomial trend
        self.coefs_ = fit_trend(Xs, order=self.order)

        # remove trend
        Xt = remove_trend(Xs, coefs=self.coefs_, time_index=self._time_index)

        # convert back into nested format
        Xt = tabulariser.inverse_transform(pd.DataFrame(Xt))
        Xt.columns = X.columns
        return Xt

    def inverse_transform(self, X, y=None):
        """Inverse transform X

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_samples, n_features]
            Nested dataframe with time-series in cells.

        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame with same number of rows and one column for each generated interval.
        """

        check_is_fitted_in_transform(self, 'coefs_')

        if self.check_input:
            if not isinstance(X, pd.DataFrame):
                raise ValueError(f"Input must be pandas DataFrame, but found: {type(X)}")

        if X.shape[1] > 1:
            raise NotImplementedError(f"Currently does not work on multiple columns, make use of ColumnTransformer "
                                      f"instead")

        if not X.shape[0] == self._input_shape[0]:
            raise ValueError(f"Inverse transform only works on data with the same number samples "
                             f"as seen during transform, but found: {X.shape[0]} samples "
                             f"!= {self._input_shape[0]} samples (seen during transform)")

        time_index = get_time_index(X)

        # convert into tabular format
        tabulariser = Tabulariser()
        Xs = tabulariser.transform(X.iloc[:, :1])

        # add trend at given time series index
        Xit = add_trend(Xs, coefs=self.coefs_, time_index=time_index)

        # convert back into nested format
        Xit = tabulariser.inverse_transform(pd.DataFrame(Xit))
        Xit.columns = X.columns
        return Xit


Deseasonalizer = Deseasonaliser