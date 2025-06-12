
from typing import List, Dict

def create_lagged_features(X: List[float], y: List[float], lag: int = 1) -> Dict[str, List[float]]:
    X_lagged = X[:-lag] if lag < len(X) else []
    y_lagged = y[lag:] if lag < len(y) else []
    return {'X': X_lagged, 'y': y_lagged}
