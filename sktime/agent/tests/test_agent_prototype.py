import pytest
from sktime.agent.forecasting import ForecastingAgent
from sktime.agent.llm.mock import MockLLM
from sktime.forecasting.base import BaseForecaster

def test_forecasting_agent_prototype():
    llm = MockLLM()
    agent = ForecastingAgent(llm=llm)
    
    query = "predict next 30 days sales"
    forecaster = agent.generate_pipeline(query)
    
    assert isinstance(forecaster, BaseForecaster)
    print("\nGenerated Forecaster:")
    print(forecaster)
    
    # Check if it has the expected steps (from MockLLM)
    from sktime.forecasting.compose import ForecastingPipeline
    assert isinstance(forecaster, ForecastingPipeline)
    assert len(forecaster.steps) == 2
    assert forecaster.steps[0][0] == 'scaler'
    assert forecaster.steps[1][0] == 'arima'

if __name__ == "__main__":
    test_forecasting_agent_prototype()
    print("Agent prototype test passed!")
