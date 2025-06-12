
from causal_structure_forecaster.forecaster import CausalStructureForecaster

def test_forecaster():
    X = [1, 2, 3, 4, 5, 6]
    y = [2, 4, 6, 8, 10, 12]
    model = CausalStructureForecaster()
    model.fit(X, y)
    predictions = model.predict([7, 8, 9])
    assert isinstance(predictions, list)
