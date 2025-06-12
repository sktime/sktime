
from pgmpy.inference import VariableElimination
import pandas as pd
from .utils import create_lagged_features
from .causal_graph import learn_dag
from .visualizer import plot_dag

class CausalStructureForecaster:
    def __init__(self):
        self.model = None
        self.graph = None
        self.inference = None

    def fit(self, X, y):
        data = create_lagged_features(X, y)
        df = pd.DataFrame(data)
        self.graph = learn_dag(df)
        self.inference = VariableElimination(self.graph)
        print("Fitted Causal DAG.")

    def predict(self, X):
        if self.inference is None:
            raise ValueError("Model not fitted yet.")

        predictions = []
        for val in X:
            try:
                pred = self.inference.query(variables=['y'], evidence={'X': val})
                predictions.append(pred.values.argmax())
            except:
                predictions.append(0)
        return predictions

    def plot_graph(self):
        if self.graph:
            plot_dag(self.graph)
        else:
            print("Graph not available. Fit the model first.")
