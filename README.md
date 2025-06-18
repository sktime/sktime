
# 📈 Causal Structure Forecaster

An open-source module that combines **causal graph learning** (`pgmpy`) and **time series forecasting** (`sktime` style API).

## 🌟 Features
- Causal DAG structure learning via `HillClimbSearch`
- `fit()` / `predict()` forecaster class
- DAG visualization with `matplotlib`
- `VariableElimination` inference engine for prediction
- Modular & unit-tested codebase
- Jupyter notebook for community demo

## 🛠️ How to Use
```bash
pip install pgmpy matplotlib networkx
```

```python
from causal_structure_forecaster.forecaster import CausalStructureForecaster

X = [1, 2, 3, 4, 5, 6]
y = [2, 4, 6, 8, 10, 12]

model = CausalStructureForecaster()
model.fit(X, y)
model.plot_graph()
print(model.predict([7, 8, 9]))
```
