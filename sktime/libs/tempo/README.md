# TEMPO Foundation Model (Vendored)

This directory contains a vendored/forked implementation of the TEMPO (Prompt-based Generative Pre-trained Transformer for Time Series Forecasting) foundation model.

## Original Source

This implementation is derived from the official TEMPO repository:
- **Repository:** [DC-research/TEMPO](https://github.com/DC-research/TEMPO)
- **Paper:** TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting (ICLR 2024)
- **Authors:** Defu Cao, Furong Jia, Sercan O Arik, Tomas Pfister, Yixiang Zheng, Wen Ye, Yan Liu
- **License:** MIT License

## Why Vendored?

The external `timeagi` package used to provide TEMPO introduces dependency
constraints that are not suitable for sktime's supported Python environments.

By vendoring the minimal required TEMPO implementation, sktime can:

- manage the required dependencies directly;
- avoid relying on the external `timeagi` package;
- integrate TEMPO with the same vendoring approach used by other foundation
  models in sktime.

## What Is Vendored

Only the minimal code required for inference:
- `tempo.py` - Main TEMPO model class with `load_pretrained_model()` and `predict()`
- `embed.py` - Data embedding layers (DataEmbedding, DataEmbedding_wo_time)
- `rev_in.py` - Reversible Instance Normalization layer

Training-specific code (layers, data providers, metrics) is not included as it is not needed for the sktime forecaster wrapper.

## License

This vendored implementation is distributed under the same MIT License as the original TEMPO project. See the license text in `__init__.py` for details.

## Usage

The vendored TEMPO model is used by `sktime.forecasting.tempo.TEMPOForecaster`. Users should not import directly from this directory unless maintaining the vendored implementation.

## Migration Date

September 2026
