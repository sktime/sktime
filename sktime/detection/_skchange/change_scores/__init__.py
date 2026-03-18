"""Change scores as interval evaluators."""

from ._continuous_linear_trend_score import (
    ContinuousLinearTrendScore,
)
from ._cusum import CUSUM
from ._esac_score import ESACScore
from ._from_cost import ChangeScore, to_change_score
from ._multivariate_gaussian_score import (
    MultivariateGaussianScore,
)
from ._rank_score import RankScore

CHANGE_SCORES = [
    ContinuousLinearTrendScore,
    ChangeScore,
    MultivariateGaussianScore,
    CUSUM,
    ESACScore,
    RankScore,
]

__all__ = CHANGE_SCORES + [to_change_score]
