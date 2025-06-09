"""Topdown reconciliation."""

import pandas as pd

from sktime.transformations.hierarchical.reconcile._base import _ReconcilerTransformer
from sktime.transformations.hierarchical.reconcile._utils import (
    _get_total_level_idxs,
    _loc_series_idxs,
    _promote_hierarchical_indexes_and_keep_timeindex,
    _recursively_propagate_topdown,
)
from sktime.transformations.hierarchical.squeeze_hierarchy import (  # noqa: E501
    SqueezeHierarchy,
)

__all__ = ["TopdownReconciler"]


class TopdownReconciler(_ReconcilerTransformer):
    """
    Apply Topdown hierarchical reconciliation.

    Forecast proportions keep the original series during `transform`,
    and propagate the "proportions" of each forecast with respect to its
    total during `inverse_transform`.

    Topdown share, on the other hand, transforms the series to share the
    forecast with respect to their parent, and then uses the total forecast
    to multiply the shares.

    For more information, see "Single level approaches" in [1].

    Parameters
    ----------
    method : str, default="td_fcst"
        The method to use for reconciliation.
        - `td_fcst`: Forecast Proportions.
        - `td_share`: Topdown Share.

    Examples
    --------
    >>> from sktime.transformations.hierarchical.reconcile import (
    ...     TopdownReconciler)
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> y = _make_hierarchical()
    >>> pipe = TopdownReconciler() * NaiveForecaster()
    >>> pipe = pipe.fit(y)
    >>> y_pred = pipe.predict(fh=[1,2,3])

    References
    ----------
    .. [1] Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting:
       principles and practice. OTexts.
    """

    def __init__(self, method="td_fcst"):
        self.method = method
        super().__init__()

        if method not in ["td_fcst", "td_share"]:
            raise ValueError(
                f"Method must be one of 'td_fcst' or 'td_share'. Got {method}."
            )

    def _fit_reconciler(self, X, y=None):
        self._drop_redundant_levels = SqueezeHierarchy()
        self._drop_redundant_levels.fit(X)
        X = self._drop_redundant_levels.transform(X)

        self._total_series = _get_total_level_idxs(X)

        return self

    def _transform_reconciler(self, X, y=None):
        """
        Prepare the data for the forecaster.

        If the method is `td_share`, the non total series are transformed to
        share the forecast with respect to their parent, and redundant levels
        are dropped.

        Otherwise, only redundant levels are dropped and the series
        are returned as is.

        Parameters
        ----------
        X : pd.DataFrame
            The series to transform.
        y: pd.DataFrame
            Ignored, for compatibility.

        Returns
        -------
        X : pd.DataFrame
            The transformed series.
        """
        X = self._drop_redundant_levels.transform(X)
        if self.method == "td_share":
            X = self._transform_non_total_to_ratios(X)
        return X

    def _inverse_transform_reconciler(self, X, y=None):
        if self.method == "td_share":
            _X = self._reconcile_td_share(X)
        else:
            _X = self._reconcile_td_fcst(X)

        _X = self._drop_redundant_levels.inverse_transform(_X)
        return _X

    def _reconcile_td_share(self, X):
        """
        Apply topdown share to the hierarchical time series.

        This method assumes that the total level was forecasted in
        absolute terms, and the bottom level was forecasted in shares.

        The steps are the following:

        1. Coerce the children to sum to one, for each parent-children relationship.
        2. Propagate the shares recursively from top to bottom.
        3. Multiply the shares by the total forecast.

        This way, we guarantee that the shares sum to the total forecast,
        and each parent's share is distributed among its children.

        Parameters
        ----------
        X : pd.DataFrame
            The transformed data.

        Returns
        -------
        _X : pd.DataFrame
            The reconciled data.
        """
        X_total, X_non_total = self._split_total_and_non_total(X)

        # Get valid shares with respect to total (1. and 2.)
        X_non_total = self._coerce_children_sum_to_one(X_non_total)
        X_ratios = pd.concat([X_total * 0 + 1, X_non_total], axis=0).sort_index()
        X_ratios = _recursively_propagate_topdown(X_ratios)

        # Now, multiply every level by total (3.)
        total_series = self._total_series[0]
        if isinstance(total_series, str):
            # If self._total_series is instance of pd.Index,
            # the total_series is a string and we force
            # it to be a tuple
            total_series = (total_series,)
        idx_total_expanded = X_ratios.index.map(lambda x: tuple([*total_series, x[-1]]))
        X_total = X_total.reindex(idx_total_expanded)
        X_total.index = X_ratios.index
        _X = X_total * X_ratios
        return _X

    def _reconcile_td_fcst(self, X):
        """
        Apply Forecast Proportions to the hierarchical time series.

        In forecast proportions, we have to recursively apply
        the ratio between the total forecast yhat/sum(yhat_lower)
        to the bottom level series, and then multiply these ratios
        by the original forecast.

        Parameters
        ----------
        X : pd.DataFrame
            The transformed data.

        Returns
        -------
        _X : pd.DataFrame
            The reconciled data.
        """
        X_ratios = self._transform_to_td_fcst_ratios(X)
        # Now, multiply the ratio down to the bottom level, recursively
        X_ratios_propagated = _recursively_propagate_topdown(X_ratios)
        _X = X_ratios_propagated * X
        return _X

    def _transform_non_total_to_ratios(self, X):
        """
        Compute the share of each series with respect to its parent.

        Parameters
        ----------
        X : pd.DataFrame
            The transformed data.

        Returns
        -------
        X_ratios : pd.DataFrame
            The transformed data with the ratios.
        """
        X_total, X_not_total = self._split_total_and_non_total(X)
        idx_map_parents = X_not_total.index.map(
            _promote_hierarchical_indexes_and_keep_timeindex
        )
        X_parents = X.reindex(idx_map_parents)
        X_parents.index = X_not_total.index
        X_ratios = X.loc[X_not_total.index] / X_parents

        return pd.concat([X_total, X_ratios], axis=0).sort_index()

    def _coerce_children_sum_to_one(self, X):
        """
        Coerce the children to sum to one for each parent.

        Parameters
        ----------
        X : pd.DataFrame
            The transformed data.
        """
        X_total, X_not_total = self._split_total_and_non_total(X)
        idx = X_not_total.index
        idx_map_parents = idx.map(_promote_hierarchical_indexes_and_keep_timeindex)

        X_parents_bu = X_not_total.groupby(
            _promote_hierarchical_indexes_and_keep_timeindex
        ).sum()
        X_parents_bu.index = pd.MultiIndex.from_tuples(X_parents_bu.index.tolist())
        X_parents_bu = X_parents_bu.reindex(idx_map_parents)
        X_parents_bu.index = idx
        X_not_total = X_not_total / X_parents_bu

        return pd.concat([X_total, X_not_total], axis=0).sort_index()

    def _transform_to_td_fcst_ratios(self, X):
        """
        Compute the forecast proportions for each parent-child relationship.

        Parameters
        ----------
        X : pd.DataFrame
            The transformed data.

        Returns
        -------
        X_ratios : pd.DataFrame
            The transformed data with the ratios.
        """
        X_total, X_not_total = self._split_total_and_non_total(X)

        idx = X_not_total.index
        # Idx that maps each series to its total series.
        idx_map_parents = idx.map(_promote_hierarchical_indexes_and_keep_timeindex)

        # Compute the bottom-up sum for each parent
        X_children_bu = X_not_total.groupby(
            _promote_hierarchical_indexes_and_keep_timeindex
        ).sum()
        # We need to unnest the index
        X_children_bu.index = pd.MultiIndex.from_tuples(X_children_bu.index.tolist())

        # The parent according to the original forecast
        X_ratios = X.loc[X_children_bu.index] / X_children_bu
        X_ratios = X_ratios.loc[idx_map_parents]
        X_ratios.index = idx

        # Add totals
        X_ratios = pd.concat([X_total * 0 + 1, X_ratios], axis=0).sort_index()

        return X_ratios

    def _split_total_and_non_total(self, X):
        """
        Split the total and non-total series.

        Parameters
        ----------
        X : pd.DataFrame
            The series to split.

        Returns
        -------
        X_total : pd.DataFrame
            The total series.
        X_not_total : pd.DataFrame
            The non-total series.
        """
        X_total = _loc_series_idxs(X, self._total_series)
        X_not_total = X.loc[~X.index.droplevel(-1).isin(self._total_series)]
        return X_total, X_not_total

    @classmethod
    def get_test_params(self, parameter_set="default"):
        """Return test parameters."""
        return [{"method": "td_fcst"}, {"method": "td_share"}]
