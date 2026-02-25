# Forecaster Wishlist

Wishlist of time series forecasting algorithms requested for implementation in sktime.

## How to Add Items

1. Open an issue on the [sktime issue tracker](https://github.com/sktime/sktime/issues) describing the algorithm
2. Add the item below following the format template
3. Submit a PR with your addition

## Format Template

```markdown
- [ ] **Algorithm Name** - Brief description
  - Paper: [Title](link) or Reference: [Package](link)
  - Issue: #1234
```

---

## Foundation Models & Deep Learning

- [ ] **TimesFM** - Google's Time Series Foundation Model
  - Reference: [google-research/timesfm](https://github.com/google-research/timesfm)
  - Issue: TBD

- [ ] **Lag-Llama** - Foundation model for time series forecasting using LLaMA architecture
  - Paper: [Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting](https://arxiv.org/abs/2310.08278)
  - Issue: TBD

- [ ] **TimeGPT** - Nixtla's foundation model for time series
  - Reference: [Nixtla/nixtla](https://github.com/Nixtla/nixtla)
  - Issue: TBD

- [ ] **PatchTST** - Patch Time Series Transformer
  - Paper: [A Time Series is Worth 64 Words](https://arxiv.org/abs/2211.14730)
  - Issue: TBD

- [ ] **iTransformer** - Inverted Transformer for time series forecasting
  - Paper: [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625)
  - Issue: TBD

## Statistical Methods

- [ ] **MSTL** - Multiple Seasonal-Trend decomposition using LOESS
  - Paper: [MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with Multiple Seasonal Patterns](https://arxiv.org/abs/2107.13462)
  - Issue: TBD

- [ ] **Robust STL** - Robust Seasonal-Trend decomposition
  - Reference: Part of statsmodels
  - Issue: TBD

## M Competition Methods

- [ ] **ES-RNN (M4 Winner)** - Exponential Smoothing with RNN
  - Paper: [A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting](https://www.sciencedirect.com/science/article/pii/S0169207019301153)
  - Issue: TBD

- [ ] **N-BEATS** - Neural Basis Expansion Analysis for Time Series
  - Paper: [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](https://arxiv.org/abs/1905.10437)
  - Issue: TBD

- [ ] **N-HiTS** - Neural Hierarchical Interpolation for Time Series
  - Paper: [N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting](https://arxiv.org/abs/2201.12886)
  - Issue: TBD

## Probabilistic Forecasting

- [ ] **DeepAR** - Probabilistic forecasting with autoregressive RNNs
  - Paper: [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/abs/1704.04110)
  - Issue: TBD

- [ ] **TFT** - Temporal Fusion Transformer
  - Paper: [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)
  - Issue: TBD

## Global/Hierarchical Forecasting

- [ ] **Hierarchical Reconciliation Methods** - Various reconciliation approaches
  - Reference: [hts](https://cran.r-project.org/web/packages/hts/index.html) R package
  - Issue: TBD

## LLM-Based

- [x] **TimeCopilot** - LLM-based forecasting agent
  - Reference: [TimeCopilot/timecopilot](https://github.com/TimeCopilot/timecopilot)
  - Issue: #8090

---

*Last updated: February 2026*
