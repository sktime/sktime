import sys
from importlib.metadata import PackageNotFoundError, version

# Vendored package modules use absolute imports via the original package name.
sys.modules.setdefault("tsbootstrap", sys.modules[__name__])

try:
    __version__ = version("tsbootstrap")
except PackageNotFoundError:
    __version__ = "0+vendored"

from .base_bootstrap import (
    BaseDistributionBootstrap,
    BaseMarkovBootstrap,
    BaseResidualBootstrap,
    BaseSieveBootstrap,
    BaseStatisticPreservingBootstrap,
    BaseTimeSeriesBootstrap,
)
from .base_bootstrap_configs import (
    BaseDistributionBootstrapConfig,
    BaseMarkovBootstrapConfig,
    BaseResidualBootstrapConfig,
    BaseSieveBootstrapConfig,
    BaseStatisticPreservingBootstrapConfig,
    BaseTimeSeriesBootstrapConfig,
)
from .block_bootstrap import (
    BartlettsBootstrap,
    BaseBlockBootstrap,
    BlackmanBootstrap,
    BlockBootstrap,
    CircularBlockBootstrap,
    HammingBootstrap,
    HanningBootstrap,
    MovingBlockBootstrap,
    NonOverlappingBlockBootstrap,
    StationaryBlockBootstrap,
    TukeyBootstrap,
)
from .block_bootstrap_configs import (
    BartlettsBootstrapConfig,
    BaseBlockBootstrapConfig,
    BlackmanBootstrapConfig,
    BlockBootstrapConfig,
    CircularBlockBootstrapConfig,
    HammingBootstrapConfig,
    HanningBootstrapConfig,
    MovingBlockBootstrapConfig,
    NonOverlappingBlockBootstrapConfig,
    StationaryBlockBootstrapConfig,
    TukeyBootstrapConfig,
)
from .block_generator import BlockGenerator
from .block_length_sampler import BlockLengthSampler
from .block_resampler import BlockResampler
from .bootstrap import (
    BlockDistributionBootstrap,
    BlockMarkovBootstrap,
    BlockResidualBootstrap,
    BlockSieveBootstrap,
    BlockStatisticPreservingBootstrap,
    WholeDistributionBootstrap,
    WholeMarkovBootstrap,
    WholeResidualBootstrap,
    WholeSieveBootstrap,
    WholeStatisticPreservingBootstrap,
)
from .markov_sampler import (
    BlockCompressor,
    MarkovSampler,
    MarkovTransitionMatrixCalculator,
)
from .ranklags import RankLags
from .time_series_model import TimeSeriesModel
from .time_series_simulator import TimeSeriesSimulator
from .tsfit import TSFit, TSFitBestLag
