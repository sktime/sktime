#!/usr/bin/env python3
"""
Causal Feature Engineering Transformer for Time Series.

Automatically discovers and generates causally-informed features for time series
forecasting.
"""

__author__ = ["XAheli"]
__all__ = ["CausalFeatureEngineer"]

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer


class CausalFeatureEngineer(BaseTransformer):
    """Causal Feature Engineering Transformer for Time Series.

    This transformer automatically discovers causal relationships in time series data
    and generates causally-informed features for forecasting models. It leverages pgmpy
    for causal discovery and creates features based on identified causal relationships.

    The transformer supports several causal discovery algorithms from pgmpy and can
    generate various types of features based on the discovered causal structure.

    Parameters
    ----------
    causal_method : str, default="pc"
        Causal discovery algorithm to use, options:
        - "pc" : PC algorithm, a constraint-based method
        - "hill_climb" : Hill Climbing Search, a score-based method
    max_lag : int, default=5
        Maximum time lag to consider for temporal causal relationships
    feature_types : List[str], default=["direct", "interaction", "temporal"]
        Types of features to generate:
        - "direct" : Direct causal features (parents in the causal graph)
        - "interaction" : Interaction features (cross-products of parents)
        - "temporal" : Time-aware features capturing temporal patterns
    weighting_strategy : str, default="causal_strength"
        Method for computing feature weights:
        - "causal_strength" : Weight by causal strength/effect size
        - "uniform" : Equal weights for all causal features
        - "inverse_lag" : Weight inversely proportional to lag length
    significance_level : float, default=0.05
        Significance level for conditional independence tests in constraint-based
        algorithms
    min_causal_strength : float, default=0.1
        Minimum causal strength required for a relationship to generate features
    expert_knowledge : dict, default=None
        Optional domain knowledge constraints:
        - "forbidden_edges" : List of tuples representing forbidden causal edges
        - "required_edges" : List of tuples representing required causal edges
        - "temporal_tiers" : Dict mapping variables to their temporal tier

    scoring_method : str, default="auto"
        Scoring method for hill climb search. Options:
        - "auto" : Automatically select based on data type
        - "k2" : K2 score for discrete data
        - "bdeu" : BDeu score for discrete data
        - "bds" : BDs score for discrete data
        - "bic-d" : BIC score for discrete data
        - "aic-d" : AIC score for discrete data
        - "bic-g" : BIC score for Gaussian (continuous) data
        - "aic-g" : AIC score for Gaussian (continuous) data
        - "ll-g" : Log-likelihood for Gaussian data


    Attributes
    ----------
    causal_graph_ : pgmpy.models.BayesianNetwork
        Discovered causal graph stored as pgmpy BayesianNetwork
    feature_importance_weights_ : dict
        Dictionary of feature importance weights based on causal strength
    features_generated_ : List[str]
        List of names of all generated features
    n_features_generated_ : int
        Number of features generated

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.transformations.series.causal_feature_engineer import (
    ...     CausalFeatureEngineer
    ... )
    >>> y = load_airline()
    >>> transformer = CausalFeatureEngineer(max_lag=3)
    >>> Xt = transformer.fit_transform(y)
    """

    _tags = {
        "authors": ["Aheli"],
        "python_dependencies": "pgmpy>=0.1.20",
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": True,
        "X_inner_mtype": ["pd.DataFrame", "pd.Series"],
        "y_inner_mtype": ["pd.Series", "pd.DataFrame", "None"],
        "fit_is_empty": False,
        "transform-returns-same-time-index": False,
        "capability:inverse_transform": False,
        "univariate-only": False,
        "capability:missing_values": True,
    }

    def __init__(
        self,
        causal_method: str = "pc",
        max_lag: int = 5,
        feature_types: Optional[list[str]] = None,
        weighting_strategy: str = "causal_strength",
        significance_level: float = 0.05,
        min_causal_strength: float = 0.1,
        expert_knowledge: Optional[dict] = None,
        scoring_method: str = "auto",
    ):
        self.causal_method = causal_method
        self.max_lag = max_lag
        self.feature_types = (
            feature_types if feature_types else ["direct", "interaction", "temporal"]
        )
        self.weighting_strategy = weighting_strategy
        self.significance_level = significance_level
        self.min_causal_strength = min_causal_strength
        self.expert_knowledge = expert_knowledge
        self.scoring_method = scoring_method

        super().__init__()

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        Private _fit method called from fit.

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Time series data to fit
        y : pd.Series or pd.DataFrame, default=None
            Additional time series data to use for causal discovery

        Returns
        -------
        self : object
            Returns self
        """
        data = self._prepare_data_for_causal_discovery(X, y)

        expert_knowledge = self._initialize_expert_knowledge()

        self.causal_graph_ = self._discover_causal_structure(data, expert_knowledge)

        self.feature_importance_weights_ = self._calculate_feature_weights(data)

        self.features_generated_ = self._define_causal_features()
        self.n_features_generated_ = len(self.features_generated_)

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        Private _transform method called from transform.

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to be transformed
        y : pd.Series or pd.DataFrame, default=None
            Additional data, e.g., target for transformation

        Returns
        -------
        Xt : pd.DataFrame
            Transformed X with causally-informed features
        """
        data = self._prepare_data_for_feature_generation(X)

        Xt = self._generate_causal_features(data)

        Xt = self._align_index_with_forecast_horizon(X, Xt)

        return Xt

    def _prepare_data_for_causal_discovery(self, X, y=None):
        """Prepare time series data for causal discovery."""
        # Combines X and y if both are provided
        if y is not None:
            if isinstance(X, pd.Series) and isinstance(y, pd.Series):
                combined_data = pd.DataFrame(
                    {X.name if X.name else "X": X, y.name if y.name else "y": y}
                )
            elif isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
                combined_data = X.copy()
                combined_data[y.name if y.name else "y"] = y
            else:
                combined_data = pd.concat([X, y], axis=1)
        else:
            if isinstance(X, pd.Series):
                combined_data = pd.DataFrame({X.name if X.name else "X": X})
            else:
                combined_data = X.copy()

        combined_data.columns = combined_data.columns.astype(str)

        # Validate data size vs max_lag
        min_required_rows = self.max_lag + 5  # Need at least 5 rows after lagging
        if len(combined_data) < min_required_rows:
            warnings.warn(
                f"Dataset has {len(combined_data)} rows but requires at least "
                f"{min_required_rows} rows for max_lag={self.max_lag}. "
                f"Reducing max_lag to {max(1, len(combined_data) - 5)}."
            )
            # Adjust max_lag for small datasets
            self.max_lag = max(1, len(combined_data) - 5)

        # Creates lagged variables up to max_lag
        lagged_data = combined_data.copy()
        for col in combined_data.columns:
            for lag in range(1, self.max_lag + 1):
                lag_name = f"{col}_lag_{lag}"
                lagged_data[lag_name] = combined_data[col].shift(lag)

        # Drops rows with NaN values (due to lagging)
        lagged_data = lagged_data.dropna()

        # Final validation
        if len(lagged_data) < 3:
            warnings.warn(
                f"After preprocessing, only {len(lagged_data)} rows remain. "
                "Causal discovery may not be reliable with very small datasets."
            )

        return lagged_data

    def _initialize_expert_knowledge(self):
        """Initialize expert knowledge for causal discovery."""
        if self.expert_knowledge is None:
            return None

        from pgmpy.estimators.ExpertKnowledge import ExpertKnowledge

        from sktime.utils.dependencies import _check_soft_dependencies

        _check_soft_dependencies("pgmpy>=0.1.20", obj=self)

        ek = ExpertKnowledge()

        try:
            # Newer pgmpy API first
            if "forbidden_edges" in self.expert_knowledge:
                for edge in self.expert_knowledge["forbidden_edges"]:
                    ek.add_edge(edge[0], edge[1], constraint_type="forbidden")

            if "required_edges" in self.expert_knowledge:
                for edge in self.expert_knowledge["required_edges"]:
                    ek.add_edge(edge[0], edge[1], constraint_type="required")

        except (AttributeError, TypeError):
            # Fallback for older pgmpy versions or different API
            try:
                if "forbidden_edges" in self.expert_knowledge:
                    for edge in self.expert_knowledge["forbidden_edges"]:
                        ek.forbid_edge(*edge)

                if "required_edges" in self.expert_knowledge:
                    for edge in self.expert_knowledge["required_edges"]:
                        ek.require_edge(*edge)

            except AttributeError:
                # Final fallback (creates basic ExpertKnowledge without constraints)
                warnings.warn(
                    "Expert knowledge constraints not supported in this pgmpy version. "
                    "Proceeding without expert knowledge constraints."
                )
                return None

        return ek

    def _discover_causal_structure(self, data, expert_knowledge=None):
        """Discover causal structure from time series data."""
        from pgmpy.estimators import PC, HillClimbSearch
        from pgmpy.estimators.CITests import chi_square, pearsonr

        from sktime.utils.dependencies import _check_soft_dependencies

        _check_soft_dependencies("pgmpy>=0.1.20", obj=self)

        if self.causal_method == "pc":
            # Use PC algorithm (constraint based)
            ci_test = chi_square if _is_discrete(data) else pearsonr
            pc = PC(data=data)
            model = pc.estimate(
                ci_test=ci_test,
                significance_level=self.significance_level,
                expert_knowledge=expert_knowledge,
                return_type="dag",
            )
            return model

        elif self.causal_method == "hill_climb":
            # Use Hill Climbing Search (score based)
            hc = HillClimbSearch(data=data)

            # Determine scoring method
            if self.scoring_method == "auto":
                # Auto select based on data type
                if _is_discrete(data):
                    scoring_method = "bic-d"  # Discrete BIC
                else:
                    scoring_method = "bic-g"  # Gaussian BIC for continuous data
            else:
                # Validate user specified scoring method
                valid_methods = [
                    "k2",
                    "bdeu",
                    "bds",
                    "bic-d",
                    "aic-d",
                    "bic-g",
                    "aic-g",
                    "ll-g",
                ]
                if self.scoring_method not in valid_methods:
                    raise ValueError(
                        f"Invalid scoring method: {self.scoring_method}. "
                        f"Valid methods: {valid_methods}"
                    )
                scoring_method = self.scoring_method

            model = hc.estimate(
                scoring_method=scoring_method,
                max_indegree=self.max_lag + 2,
                show_progress=False,
            )

            if expert_knowledge:
                model = expert_knowledge.apply_expert_knowledge(model)

            return model

        else:
            raise ValueError(
                f"Unsupported causal discovery method: {self.causal_method}. "
                "Use 'pc' or 'hill_climb'."
            )

    def _calculate_feature_weights(self, data):
        """Calculate feature importance weights based on causal strength."""
        weights = {}

        if self.weighting_strategy == "uniform":
            for edge in self.causal_graph_.edges():
                weights[f"{edge[0]}_to_{edge[1]}"] = 1.0

        elif self.weighting_strategy == "causal_strength":
            for edge in self.causal_graph_.edges():
                cause, effect = edge
                if cause in data.columns and effect in data.columns:
                    strength = abs(data[cause].corr(data[effect]))
                    weights[f"{cause}_to_{effect}"] = strength
                else:
                    weights[f"{cause}_to_{effect}"] = 0.5

        elif self.weighting_strategy == "inverse_lag":
            for edge in self.causal_graph_.edges():
                cause, effect = edge
                if "_lag_" in cause:
                    try:
                        lag = int(cause.split("_lag_")[1])
                        weights[f"{cause}_to_{effect}"] = 1.0 / lag
                    except (IndexError, ValueError):
                        weights[f"{cause}_to_{effect}"] = 0.5
                else:
                    weights[f"{cause}_to_{effect}"] = 1.0

        # Filter out weak relationships
        weights = {k: v for k, v in weights.items() if v >= self.min_causal_strength}

        return weights

    def _define_causal_features(self):
        """Define features based on discovered causal graph."""
        features = []

        for node in self.causal_graph_.nodes():
            if "_lag_" in node:
                continue

            parents = list(self.causal_graph_.get_parents(node))

            if "direct" in self.feature_types:
                features.extend(parents)

            if "interaction" in self.feature_types and len(parents) > 1:
                for i in range(len(parents)):
                    for j in range(i + 1, len(parents)):
                        features.append(f"{parents[i]}_x_{parents[j]}")

            if "temporal" in self.feature_types:
                lagged_parents = [p for p in parents if "_lag_" in p]
                for lag_var in lagged_parents:
                    base_var = lag_var.split("_lag_")[0]
                    lag = int(lag_var.split("_lag_")[1])
                    if lag > 1:
                        features.append(f"{base_var}_rate_{lag}")

        # Remove duplicates
        unique_features = []
        for f in features:
            if f not in unique_features:
                unique_features.append(f)

        return unique_features

    def _prepare_data_for_feature_generation(self, X):
        """Prepare data for feature generation."""
        if isinstance(X, pd.Series):
            data = pd.DataFrame({X.name if X.name else "X": X})
        else:
            data = X.copy()

        data.columns = data.columns.astype(str)

        for col in data.columns:
            for lag in range(1, self.max_lag + 1):
                lag_name = f"{col}_lag_{lag}"
                data[lag_name] = data[col].shift(lag)

        return data

    def _generate_causal_features(self, data):
        """Generate features based on causal structure."""
        Xt = pd.DataFrame(index=data.index)

        for feature in self.features_generated_:
            if feature in data.columns:
                Xt[feature] = data[feature]
                continue

            if "_x_" in feature:
                var1, var2 = feature.split("_x_")
                if var1 in data.columns and var2 in data.columns:
                    Xt[feature] = data[var1] * data[var2]
                continue

            if "_rate_" in feature:
                base_var, lag = feature.split("_rate_")
                try:
                    lag = int(lag)
                    lag_col_curr = f"{base_var}_lag_{lag - 1}"
                    lag_col_prev = f"{base_var}_lag_{lag}"
                    if lag_col_curr in data.columns and lag_col_prev in data.columns:
                        Xt[feature] = (data[lag_col_curr] - data[lag_col_prev]) / lag
                except (ValueError, KeyError):
                    continue

        # Apply feature weights
        if hasattr(self, "feature_importance_weights_"):
            for feature in Xt.columns:
                weight_key = next(
                    (
                        k
                        for k in self.feature_importance_weights_.keys()
                        if feature in k
                    ),
                    None,
                )
                if weight_key:
                    weight = self.feature_importance_weights_[weight_key]
                    Xt[feature] = Xt[feature] * weight

        # Empty features case
        if Xt.shape[1] == 0:
            valid_index = data.dropna().index

            # Detect if this was originally univariate input
            original_columns = [col for col in data.columns if "_lag_" not in col]

            if len(original_columns) == 1:
                # Univariate case: creates placeholder column
                col_name = f"{original_columns[0]}_no_causal_features"
                Xt = pd.DataFrame({col_name: np.nan}, index=valid_index)
            else:
                # Multivariate case: returns empty DataFrame
                Xt = pd.DataFrame(index=valid_index)
        else:
            Xt = Xt.dropna()

        return Xt

    def _align_index_with_forecast_horizon(self, X, Xt):
        """Align the transformed data with the expected forecast horizon."""
        if len(Xt) < len(X):
            warnings.warn(
                f"Generated features have fewer observations ({len(Xt)}) than "
                f"input data ({len(X)}) due to lags of up to {self.max_lag}. "
                f"Features are aligned with the end of the series."
            )

            self._reduced_index_ = Xt.index
            self._original_index_ = X.index
        else:
            # If Xt has more rows than X, we can assume it is aligned correctly
            self._reduced_index_ = None
            self._original_index_ = None

        return Xt

    def get_aligned_target_index(self):
        """Return the index that target variable should use for proper alignment."""
        if hasattr(self, "_reduced_index_"):
            return self._reduced_index_
        return None

    def get_safe_target_index(self, target_index):
        """Return a safe index that only contains values present in target_index."""
        aligned_index = self.get_aligned_target_index()
        if aligned_index is not None:
            # Only returns intersection to avoid KeyError
            safe_index = aligned_index.intersection(target_index)
            return safe_index if len(safe_index) > 0 else None
        return None

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter for comprehensive coverage."""
        if parameter_set == "default":
            # Core algorithm coverage
            params1 = {
                "causal_method": "pc",
                "max_lag": 2,
                "feature_types": ["direct", "interaction"],
                "significance_level": 0.1,
                "min_causal_strength": 0.05,
            }

            params2 = {
                "causal_method": "hill_climb",
                "max_lag": 3,
                "feature_types": ["direct", "temporal"],
                "scoring_method": "bic-g",
                "significance_level": 0.05,
                "min_causal_strength": 0.1,
            }

            # Different feature type combinations
            params3 = {
                "causal_method": "pc",
                "max_lag": 2,
                "feature_types": ["direct", "interaction", "temporal"],
                "weighting_strategy": "uniform",
                "significance_level": 0.2,
                "min_causal_strength": 0.01,
            }

            # Different weighting strategies
            params4 = {
                "causal_method": "hill_climb",
                "max_lag": 3,
                "feature_types": ["direct"],
                "weighting_strategy": "inverse_lag",
                "scoring_method": "aic-g",
                "significance_level": 0.05,
                "min_causal_strength": 0.3,
            }

            # Edge case: minimal settings
            params5 = {
                "causal_method": "pc",
                "max_lag": 1,
                "feature_types": ["direct"],
                "significance_level": 0.5,
                "min_causal_strength": 0.01,
            }

            # Edge case: strict settings
            params6 = {
                "causal_method": "hill_climb",
                "max_lag": 1,
                "feature_types": ["interaction"],
                "scoring_method": "auto",
                "significance_level": 0.001,
                "min_causal_strength": 0.8,
            }

            # Expert knowledge test
            params7 = {
                "causal_method": "pc",
                "max_lag": 2,
                "feature_types": ["direct", "temporal"],
                "significance_level": 0.1,
                "min_causal_strength": 0.1,
                "expert_knowledge": {"forbidden_edges": [("X1", "X2")]},
            }

            # Different scoring methods
            params8 = {
                "causal_method": "hill_climb",
                "max_lag": 3,
                "feature_types": ["direct", "interaction"],
                "scoring_method": "ll-g",
                "significance_level": 0.05,
                "min_causal_strength": 0.2,
            }

            return [
                params1,
                params2,
                params3,
                params4,
                params5,
                params6,
                params7,
                params8,
            ]

        return cls.get_test_params("default")


def _is_discrete(data):
    """Check if data appears to be discrete or continuous."""
    for col in data.columns:
        if not np.issubdtype(data[col].dtype, np.integer):
            return False
        if len(data[col].unique()) > 10:
            return False
    return True
