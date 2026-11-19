
from sktime.datasets import load_airline
from sktime.forecasting.llm import LLMForecaster
from sktime.utils.estimator_checks import check_estimator


def test_basic_fit_predict():
    y = load_airline()

    mock_llm = lambda prompt: (
        '{"estimator": "NaiveForecaster", "params": {"strategy": "mean"}, '
        '"explanation": "Fallback Naive model due to missing info."}'
    )

    agent = LLMForecaster(
        prompt="Use a Naive model to forecast the next 12 periods.", llm=mock_llm
    )

    print("Fitting agent...")
    agent.fit(y)
    print("Agent fitted:", agent.estimator_)
    print("Agent summary:", agent.llm_summary_)

    preds = agent.predict(fh=[1, 2, 3])
    print("Predictions:\n", preds)

    print("Running standard check_estimator...")
    results = check_estimator(LLMForecaster, raise_exceptions=False)
    print("Check estimator results:", results)


if __name__ == "__main__":
    test_basic_fit_predict()
