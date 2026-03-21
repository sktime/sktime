"""Change scores as interval evaluators."""

from sktime.detection._skchange.change_scores._continuous_linear_trend_score import (
    ContinuousLinearTrendScore,
)
from sktime.detection._skchange.change_scores._cusum import CUSUM
from sktime.detection._skchange.change_scores._esac_score import ESACScore
from sktime.detection._skchange.change_scores._from_cost import (
    ChangeScore,
    to_change_score,
)
from sktime.detection._skchange.change_scores._multivariate_gaussian_score import (
    MultivariateGaussianScore,
)
from sktime.detection._skchange.change_scores._rank_score import RankScore

CHANGE_SCORES = [
    ContinuousLinearTrendScore,
    ChangeScore,
    MultivariateGaussianScore,
    CUSUM,
    ESACScore,
    RankScore,
]

__all__ = CHANGE_SCORES + [to_change_score]
