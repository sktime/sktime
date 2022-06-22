# -*- coding: utf-8 -*-
"""Implement and register validations, which compute benchmark results."""
import functools
from typing import Callable, Dict, List, Union

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection._split import BaseSplitter
from sktime.performance_metrics.base import BaseMetric


def forecasting_validation(
    dataset_loader: Callable,
    cv_splitter: BaseSplitter,
    scorers: List[BaseMetric],
    estimator: BaseForecaster,
    **kwargs,
) -> Dict[str, Union[float, str]]:
    """Run validation for forecasting benchmarks."""
    y = dataset_loader()
    results = {}
    for scorer in scorers:
        scorer_name = scorer.name
        scores_df = evaluate(forecaster=estimator, y=y, cv=cv_splitter, scoring=scorer)
        for ix, row in scores_df.iterrows():
            results[f"{scorer_name}_fold_{ix}_test"] = row[f"test_{scorer_name}"]
        results[f"{scorer_name}_mean"] = scores_df[f"test_{scorer_name}"].mean()
        results[f"{scorer_name}_std"] = scores_df[f"test_{scorer_name}"].std()
    return results


def factory_forecasting_validation(
    dataset_loader: Callable,
    cv_splitter: BaseSplitter,
    scorers: List[BaseMetric],
) -> Callable:
    """Build forecasting validation func which just takes an estimator."""
    return functools.partial(
        forecasting_validation,
        dataset_loader,
        cv_splitter,
        scorers,
    )
