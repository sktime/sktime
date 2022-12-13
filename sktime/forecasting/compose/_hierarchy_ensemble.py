# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements forecaster for applying different univariates on hierarchical data."""

__author__ = ["VyomkeshVyas", "ciaran-g"]
__all__ = ["HierarchyEnsembleForecaster"]


import pandas as pd

from sktime.base._meta import flatten
from sktime.forecasting.base._base import BaseForecaster
from sktime.forecasting.base._meta import _HeterogenousEnsembleForecaster


class HierarchyEnsembleForecaster(_HeterogenousEnsembleForecaster):
    """Aggregates hierarchical data, fit forecasters and make predictions.

    Can apply different univariate forecaster either on different
    level of aggregation or on different hierarchical nodes.

    `HierarchyEnsembleForecaster` is passed forecaster/level or
    forecaster/node pairs. Level can only be int >=0 with 0
    signifying the topmost level of aggregation.
    Node can only be a tuple of strings or list of tuples.

    Behaviour in `fit`, `predict`:
    For level pairs f_i, l_i passed, applies forecaster f_i to level l_i.
    For node pairs f_i, n_i passed, applies forecaster f_i on each node of n_i.
    if "default" argument passed, applies "default" forecaster on the
    remaining levels/nodes which are not mentioned in argument 'forecasters'.
    `predict` results are concatenated to one container with same columns as in `fit`.


    Parameters
    ----------
    forecasters : sktime forecaster, or list of tuples
                (estimator, int or (tuple or list of tuples))
        if forecaster, clones of forecaster are applied to all aggregated levels.
        if list of tuples, with estimator is forecaster, level/node
        as int/tuple respectively.

    by : {'node', 'level', default='level'}
        if 'level', applies a univariate forecaster on all the hierarchical
        nodes within a level of aggregation
        if 'node', applies separate univariate forecaster for each
        hierarchical node.

    default : sktime forecaster {default = None}
        if passed, applies 'default' forecaster on the nodes/levels
        not mentioned in the 'forecaster' argument.

    Examples
    --------
    >>> from sktime.forecasting.compose import HierarchyEnsembleForecaster
    >>> from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.theta import ThetaForecaster
    >>> from sktime.utils._testing.hierarchical import _bottom_hier_datagen
    >>> y = _bottom_hier_datagen(
    ...         no_bottom_nodes=7,
    ...         no_levels=2,
    ...         random_seed=123
    ... )

    >>> # Example of by = 'level'
    >>> forecasters = [
    ...     ( NaiveForecaster(), 0),
    ...     (ExponentialSmoothing(trend='add'), 1),
    ...     (ThetaForecaster(), 2)
    ... ]
    >>> forecaster = HierarchyEnsembleForecaster(forecasters=forecasters,by='level')
    >>> forecaster.fit(y, fh=[1, 2, 3])
    >>> y_pred = forecaster.predict()

    >>> # Example of by = 'node'
    >>> forecasters = [
    ...     (ExponentialSmoothing(), ("__total", "__total")),
    ...    (ThetaForecaster(), ('l2_node01', 'l1_node01')
    ... ]
    >>> forecaster = HierarchyEnsembleForecaster(forecasters=forecasters,by='node')
    >>> forecaster.fit(y, fh=[1, 2, 3])
    >>> y_pred = forecaster.predict()
    """

    _tags = {
        "scitype:y": "both",
        "ignores-exogeneous-X": False,
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "capability:pred_int": False,
        "fit_is_empty": False,
        "enforce_index_type": None,
    }

    BY_LIST = ["level", "node"]

    def __init__(self, forecasters, by="level", default=None):

        self.forecasters = forecasters
        self.by = by
        self.default = default
        self.forecasters_ = []

        super(HierarchyEnsembleForecaster, self).__init__(forecasters=forecasters)

        if isinstance(forecasters, BaseForecaster):
            tags_to_clone = [
                "requires-fh-in-fit",
                "capability:pred_int",
                "ignores-exogeneous-X",
                "handles-missing-data",
                "fit_is_empty",
                "enforce_index_type",
            ]
            self.clone_tags(forecasters, tags_to_clone)
        else:
            if isinstance(forecasters, tuple):
                forecasters = [forecasters]
            l_forecasters = [(str(x[0]), x[0]) for x in forecasters]
            self._anytagis_then_set("requires-fh-in-fit", True, False, l_forecasters)
            self._anytagis_then_set("capability:pred_int", False, True, l_forecasters)
            self._anytagis_then_set("ignores-exogeneous-X", False, True, l_forecasters)
            self._anytagis_then_set("handles-missing-data", False, True, l_forecasters)
            self._anytagis_then_set("fit_is_empty", False, True, l_forecasters)
            self._anytagis_then_set("enforce_index_type", False, True, l_forecasters)

    def _aggregate(self, y):
        """Add total levels to y, using Aggregate."""
        from sktime.transformations.hierarchical.aggregate import Aggregator

        return Aggregator().fit_transform(y)

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd-multiindex
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        self : returns an instance of self.
        """
        # Creating  aggregated levels in data
        z = self._aggregate(y)

        if X is not None:
            X = self._aggregate(X)

        # check forecasters
        self.forecasters_ = self._check_forecasters(y, z)

        self.fitted_list = []
        hier_nm = z.index.names

        if self.by == "level":
            hier_dict = self._get_hier_dict(z)
            for (forecaster, level) in self.forecasters_:
                if level in hier_dict.keys():
                    frcstr = forecaster.clone()
                    df = z[z.index.droplevel(-1).isin(hier_dict[level])]
                    if X is not None:
                        x = X.loc[df.index]
                        frcstr.fit(df, fh=fh, X=x)
                    else:
                        frcstr.fit(df, fh=fh, X=X)
                    self.fitted_list.append([frcstr, df.index])

        else:
            node_dict = self._get_node_dict(z)
            for forecaster, nodes in node_dict.items():
                for node in nodes:
                    frcstr = forecaster.clone()
                    df = z.loc[node]
                    df[hier_nm[:-1]] = list(node)
                    df = df.set_index(hier_nm[:-1], append=True).reorder_levels(hier_nm)
                    if X is not None:
                        x = X.loc[df.index]
                        frcstr.fit(df, fh=fh, X=x)
                    else:
                        frcstr.fit(df, fh=fh, X=X)
                    self.fitted_list.append([frcstr, df.index])
        return self

    def _get_hier_dict(self, z):

        hier_dict = {}
        hier = z.index.droplevel(-1).unique()
        nlvls = z.index.nlevels

        _, levels = zip(*self.forecasters_)

        level_flat = flatten(levels)
        level_set = set(level_flat)

        for i in range(1, nlvls + 1):
            if self.default or nlvls - i in level_set:
                if i == 1:
                    level = hier[hier.get_level_values(-i) != "__total"]
                    hier_dict[nlvls - i] = level
                elif i == nlvls:
                    level = hier[hier.get_level_values(-i + 1) == "__total"]
                    hier_dict[nlvls - i] = level
                else:
                    level = hier[hier.get_level_values(-i) != "__total"]
                    level_cp = hier[hier.get_level_values(-i + 1) != "__total"]
                    diff = level.difference(level_cp)
                    if len(diff) != 0:
                        hier_dict[nlvls - i] = diff

        return hier_dict

    def _get_node_dict(self, z):

        node_dict = {}
        nodes = []

        for (forecaster, node) in self.forecasters_:
            node_dict[forecaster] = node
            nodes += node

        if self.default:
            diff_nodes = z.index.droplevel(-1).unique().difference(nodes)
            node_dict[self.default] = diff_nodes

        return node_dict

    def _predict(self, fh=None, X=None):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        preds = []

        if X is not None:
            X = self._aggregate(X)

        for i in range(len(self.fitted_list)):
            if X is not None:
                x = X.loc[self.fitted_list[i][1]]
                pred = self.fitted_list[i][0].predict(fh=fh, X=x)
            else:
                pred = self.fitted_list[i][0].predict(fh=fh, X=X)
            preds.append(pred)

        preds = pd.concat(preds, axis=0)
        preds.sort_index(inplace=True)
        return preds

    def _check_forecasters(self, y, z):
        """Raise error if BY is not defined correctly."""
        if self.by not in self.BY_LIST:
            raise ValueError(f"""BY must be one of {self.BY_LIST}.""")

        if y.index.nlevels < 2:
            raise ValueError(
                "Data should have multiindex with levels greater than or equal to 2."
            )

        # if a single estimator is passed, replicate across levels
        if isinstance(self.forecasters, BaseForecaster):
            lvlrange = range(y.index.nlevels)
            forecaster_list = [self.forecasters.clone() for _ in lvlrange]
            self.by = "level"
            return list(zip(forecaster_list, lvlrange))

        if (
            self.forecasters is None
            or len(self.forecasters) == 0
            or not isinstance(self.forecasters, list)
        ):
            raise ValueError(
                "Invalid 'forecasters' attribute, 'forecasters' should be a list"
                " of (estimator, int or tuple) tuples."
            )
        if not isinstance(self.default, BaseForecaster) and self.default is not None:
            raise ValueError(
                "Invalid 'default' attribute, 'default' should be a Forecaster"
            )
        if self.by == "node":
            for i in range(len(self.forecasters)):
                if type(self.forecasters[i]) == tuple:
                    self.forecasters[i] = list(self.forecasters[i])
                if type(self.forecasters[i][1]) == tuple:
                    self.forecasters[i][1] = [self.forecasters[i][1]]

        forecasters, level_nd = zip(*self.forecasters)

        for forecaster in forecasters:
            if not isinstance(forecaster, BaseForecaster):
                raise ValueError(
                    f"The estimator {forecaster.__class__.__name__} should be a "
                    f"Forecaster."
                )

        if self.by == "level":
            level_flat = flatten(level_nd)
            level_set = set(level_flat)
            not_in_y_idx = level_set.difference(range(y.index.nlevels))
            y_lvls_not_found = set(range(y.index.nlevels)).difference(level_set)

            if len(not_in_y_idx) > 0:
                raise ValueError(
                    f"Level identifier must be integers within "
                    f"the range of the total number of levels, "
                    f"but found level identifiers that are not: {list(not_in_y_idx)}"
                )

            if len(level_set) != len(level_flat):
                raise ValueError(
                    f"One estimator per level required. Found {len(level_flat)} unique"
                    f" level names in forecasters arg, required: {y.index.nlevels}"
                )
            if self.default:
                forecaster_list = [self.default.clone() for _ in y_lvls_not_found]
                return self.forecasters + list(zip(forecaster_list, y_lvls_not_found))
        else:
            for nodes in level_nd:
                for node in nodes:
                    if len(node) != z.index.nlevels - 1:
                        raise ValueError(
                            f"Individual node length must be "
                            f"equal to {z.index.nlevels-1} "
                            f"but found : {len(node)}"
                        )
                    if (
                        type(node) != tuple
                        or node not in z.index.droplevel(-1).unique()
                    ):
                        raise ValueError(
                            "Individual node value must be a tuple "
                            "within the multi-index of aggregated "
                            "dataframe and must not include timepoint index."
                        )

        return self.forecasters

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        # imports
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.trend import TrendForecaster

        params1 = {
            "forecasters": [(NaiveForecaster(), 0), (TrendForecaster(), 1)],
            "by": "level",
            "default": None,
        }
        params2 = {
            "forecasters": [(NaiveForecaster(), 0), (TrendForecaster(), 1)],
            "by": "level",
            "default": NaiveForecaster(),
        }
        params3 = {
            "forecasters": [(TrendForecaster(), ("__total", "__total"))],
            "by": "node",
            "default": None,
        }
        params4 = {
            "forecasters": [(TrendForecaster(), ("__total", "__total"))],
            "by": "node",
            "default": NaiveForecaster(),
        }

        return [params1, params2, params3, params4]
