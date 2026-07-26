# LLM-based Time Series Assistant (sktime)

A minimal but clean prototype that combines a lightweight LLM-style query parser with `sktime` time-series functionality.

## What it does

Given a natural language query, the assistant:

1. Parses the query into structured intent
2. Selects an `sktime` pipeline
3. Runs forecasting or analysis
4. Returns a human-readable response

Supported tasks:

- Forecasting: e.g. `forecast next 12 steps`
- Basic analysis:
  - Trend: e.g. `what is the trend?`
  - Mean/stats: e.g. `show summary stats`
  - Anomaly hints: e.g. `detect anomalies`

## Project structure

- `main.py` - CLI entrypoint and demo flow
- `agent.py` - Query parsing, task routing, and response generation
- `utils.py` - Dataset loading and analysis helpers

## Setup

```bash
pip install -r requirements.txt
```

## Run

### Scripted demo

```bash
python main.py
```

### Single query

```bash
python main.py --query "forecast next 12 steps"
python main.py --query "what is the trend?"
python main.py --query "detect anomalies"
```

### Interactive mode

```bash
python main.py --interactive
```

## Notes

- This prototype uses a deterministic `MockLLM` parser for reliability and low dependency overhead.
- The forecasting model is `NaiveForecaster(strategy="last")` for simplicity.
- Anomaly detection is a basic z-score hint, intended as a lightweight analysis baseline.
- You can later replace `MockLLM` with an OpenAI-backed parser without changing the agent interface.
