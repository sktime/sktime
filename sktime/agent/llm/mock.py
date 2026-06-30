# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Mock LLM for testing."""

from sktime.agent.llm.base import BaseLLM

class MockLLM(BaseLLM):
    """Mock LLM that returns hardcoded code for specific queries."""

    def __init__(self, responses=None):
        self.responses = responses or {
            "predict next 30 days sales": (
                "from sktime.forecasting.arima import ARIMA\n"
                "from sktime.forecasting.compose import ForecastingPipeline\n"
                "from sktime.transformations.series.adapt import TabularToSeriesAdaptor\n"
                "from sklearn.preprocessing import StandardScaler\n\n"
                "forecaster = ForecastingPipeline([\n"
                "    ('scaler', TabularToSeriesAdaptor(StandardScaler())),\n"
                "    ('arima', ARIMA(order=(1, 1, 1)))\n"
                "])"
            )
        }
        super().__init__()

    def generate_code(self, prompt: str) -> str:
        """Return a hardcoded response if the prompt contains a known query."""
        for query, response in self.responses.items():
            if query in prompt:
                return response
        return "raise ValueError('MockLLM: Query not recognized')"
