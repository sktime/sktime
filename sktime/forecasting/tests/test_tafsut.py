# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Regression tests for the Tafsut forecaster against upstream outputs.

The reference outputs in this module are generated from the pretrained
``Tafsut-FM/tafsut-univariate-base`` checkpoint (105M parameters, 421 MB in
float32), which fits comfortably in memory, so no seeded-small-model fallback
is used.
"""

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.tafsut import TafsutForecaster
from sktime.tests.test_switch import run_test_for_class

# Reference forecasts were generated on CPU from the upstream ``tafsut`` package,
# version 0.1.0 from PyPI (the only release at the time of writing; sdist sha256
# 41c1f3a512995090decc2c8a46041559d9d64356806a3da81ae97494ea7cca4a), with the
# weights of the Tafsut-FM/tafsut-univariate-base Hugging Face repository at
# the immutable commit a17c92cbab2cccecbf033406644af88586f6731c, bypassing
# sktime entirely.
#
# ``TafsutForecaster`` has no ``revision`` parameter, so the tests below load
# the ``main`` revision of the checkpoint, which at the time of writing is the
# commit above. sktime does not vendor any Tafsut code: the model code comes
# from the installed ``tafsut`` package.
#
# Environment: Python 3.12.14, tafsut 0.1.0, torch 2.14.1, numpy 2.5.3.
#
# The reference values were produced by the following standalone script:
#
#     import numpy as np
#     import torch
#     from tafsut import forecast, load_tafsut
#
#     # sktime is only used as a data source here (airline passengers).
#     from sktime.datasets import load_airline
#
#     REPO = "Tafsut-FM/tafsut-univariate-base"
#     REVISION = "a17c92cbab2cccecbf033406644af88586f6731c"
#     SEED = 0
#
#     # The pretrained checkpoint at REVISION is loaded on CPU. The model is
#     # deterministic: the checkpoint has dropout_rate=0.0 and ``forecast``
#     # puts the model in eval mode, so SEED only pins the RNG state for good
#     # measure.
#     #
#     # Each case mirrors TafsutForecaster(device="cpu") on the first 132
#     # airline values:
#     #
#     # * "default": fh=1..12. One forward pass, which yields
#     #   config.prediction_length = 1024 values, truncated to 12 values.
#     # * "two-steps-1100": fh=1..1100. The 1024 values of the first forward
#     #   pass are followed by a second pass, in which the median forecast of
#     #   the first pass is appended to the context, for the remaining 76
#     #   values.
#     CASES = {"default": 12, "two-steps-1100": 1100}
#
#
#     def fmt(values):
#         return "[" + ", ".join(f"{float(v):.9g}" for v in values) + "]"
#
#
#     def main():
#         model = load_tafsut(REPO, device="cpu", revision=REVISION)
#         quantiles = list(model.cfg.quantiles)
#
#         y = load_airline().to_numpy()[:132].astype(np.float32)
#
#         for name, horizon in CASES.items():
#             torch.manual_seed(SEED)
#             output = forecast(model, y, horizon=horizon)
#             # output: [batch=1, horizon, quantiles]
#             assert output.shape == (1, horizon, len(quantiles)), output.shape
#             print(f"== {name}")
#             for level in [0.1, 0.5, 0.9]:
#                 values = output[0, :, quantiles.index(level)].numpy()
#                 print(f"  q{level} head :", fmt(values[:3]))
#                 print(f"  q{level} tail :", fmt(values[-3:]))
#
#
#     if __name__ == "__main__":
#         main()

_TAFSUT_QUANTILES = [0.1, 0.5, 0.9]


def _load_airline_train():
    """Airline passengers without the last 12 months."""
    return load_airline().iloc[:-12]


# (horizon, expected heads, expected tails), one row per level in
# _TAFSUT_QUANTILES; the 0.5 quantile is also the point forecast
_TAFSUT_UPSTREAM_REFERENCE_CASES = [
    pytest.param(
        12,
        [
            [389.012268, 378.42041, 438.45047],
            [417.771271, 408.086792, 474.493073],
            [447.742126, 439.556641, 512.844116],
        ],
        [
            [387.592743, 345.220459, 376.202209],
            [439.638885, 383.486298, 425.368866],
            [499.876801, 433.576233, 479.803894],
        ],
        id="default",
    ),
    pytest.param(
        1100,
        [
            [389.012268, 378.42041, 438.45047],
            [417.771271, 408.086792, 474.493073],
            [447.742126, 439.556641, 512.844116],
        ],
        [
            [944.87854, 993.90387, 1000.82037],
            [1148.16504, 1239.31421, 1263.88135],
            [1378.38196, 1498.79077, 1531.81055],
        ],
        id="two-steps-1100",
    ),
]


@pytest.mark.skipif(
    not run_test_for_class(TafsutForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "horizon,expected_heads,expected_tails",
    _TAFSUT_UPSTREAM_REFERENCE_CASES,
)
def test_tafsut_predictions_match_upstream_reference(
    horizon, expected_heads, expected_tails
):
    """Tafsut predictions match the upstream tafsut-univariate-base outputs."""
    y_train = _load_airline_train()
    fh = np.arange(1, horizon + 1)
    forecaster = TafsutForecaster(device="cpu")

    forecaster.fit(y_train, fh=fh)
    y_pred = forecaster.predict(fh=fh)
    y_quantiles = forecaster.predict_quantiles(fh=fh, alpha=_TAFSUT_QUANTILES)

    expected_heads = np.asarray(expected_heads, dtype=np.float32).T
    expected_tails = np.asarray(expected_tails, dtype=np.float32).T
    median = _TAFSUT_QUANTILES.index(0.5)

    assert len(y_pred) == horizon
    assert y_quantiles.shape == (horizon, len(_TAFSUT_QUANTILES))
    np.testing.assert_allclose(
        y_pred.iloc[:3].to_numpy(), expected_heads[:, median], rtol=1e-5, atol=1e-4
    )
    np.testing.assert_allclose(
        y_pred.iloc[-3:].to_numpy(), expected_tails[:, median], rtol=1e-5, atol=1e-4
    )
    np.testing.assert_allclose(
        y_quantiles.iloc[:3].to_numpy(), expected_heads, rtol=1e-5, atol=1e-4
    )
    np.testing.assert_allclose(
        y_quantiles.iloc[-3:].to_numpy(), expected_tails, rtol=1e-5, atol=1e-4
    )
