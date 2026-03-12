# global_forecaster Wishlist

Wishlist of global time series forecasting algorithms requested for implementation in sktime.

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

## Wishlist Items

### Transformer-based global forecasters

- [ ] **N-BEATS** - Neural basis expansion analysis for interpretable global forecasting.
- [ ] **N-HiTS** - N-BEATS inspired model for high-throughput forecasting with multiresolution blocks.
- [ ] **DeepAR** - Autoregressive RNN-based probabilistic global forecaster.
- [ ] **Temporal Fusion Transformer (TFT)** - Attention-based global model with static and time-varying covariates.
- [ ] **Informer** - Efficient transformer for long sequence forecasting with sparse attention.
- [ ] **Autoformer** - Transformer with series decomposition for long-term forecasting.
- [ ] **FEDformer** - Frequency-enhanced decomposed transformer model.
- [ ] **PatchTST** - Patch-based transformer for long-horizon forecasting.
- [ ] **TimesNet** - Neural operator style model for time series forecasting.
- [ ] **Crossformer** - Transformer with cross-dimension dependency modeling.
- [ ] **Pyraformer** - Pyramidal attention transformer for long sequence forecasting.
- [ ] **ETSformer** - Exponential smoothing inspired transformer.
- [ ] **Non-stationary Transformer** - Transformer variants explicitly modeling non-stationarity.
- [ ] **iTransformer** - Instance-dependent transformer architecture for time series.
- [ ] **TimeXer** - Lightweight transformer model for time series.
- [ ] **Timer** - Transformer-style model for time-aware forecasting.

### State-of-the-art deep learning forecasters

- [ ] **DLinear** - Decomposition-based linear global forecaster.
- [ ] **NLinear** - Simple normalized linear global forecasting baseline.
- [ ] **RLinear** - Residual-style linear forecasting model.
- [ ] **TSMixer** - MLP-based architecture with mixing over time and features.
- [ ] **TiDE (Time-series Dense Encoder)** - Deep learning model with dense encoders for time series.
- [ ] **ModernTCN** - Modern temporal convolutional network architectures for forecasting.
- [ ] **TimeMachine** - Advanced deep forecasting architectures (family).
- [ ] **SCINet** - Series decomposition network for interpretable forecasting.
- [ ] **MICN** - Multi-scale inception convolutional network for time series.
- [ ] **FreTS** - Frequency enhanced time series model.
- [ ] **Global RNN variants (GRU, LSTM, BiLSTM)** - Global recurrent models for multiple series.
- [ ] **WaveNet-style forecasters** - Dilated causal convolutional networks for forecasting.
- [ ] **Temporal Convolutional Networks (TCN)** - Global TCN architectures for time series.

### Foundation and large time series models

- [ ] **TimesFM (Google)** - Time series foundation model exposed as a global forecaster.
- [ ] **Chronos (Amazon)** - Pre-trained probabilistic foundation model for time series.
- [ ] **Lag-Llama** - LLM-style foundation model for time series.
- [ ] **TimeGPT (Nixtla)** - Forecasting foundation model as a service.
- [ ] **MOMENT** - Large-scale multi-task time series model.
- [ ] **UniTime** - Unified foundation model for multiple time series tasks.
- [ ] **Timer (foundation model variants)** - Foundation model variants built on Timer-style architectures.
- [ ] **Time-LLM** - LLM-enhanced time series forecasting models.

### Probabilistic deep global forecasters

- [ ] **DeepVAR** - Vector autoregressive deep probabilistic model.
- [ ] **GPVAR** - Gaussian process-based vector autoregression.
- [ ] **DeepState** - State space model with deep learning components.
- [ ] **DeepFactor** - Global-local factor model for time series.
- [ ] **MQ-CNN (Multi-Quantile CNN)** - CNN-based global quantile forecaster.
- [ ] **MQ-RNN** - RNN-based multi-quantile forecaster.
- [ ] **Normalizing flows for forecasting** - Flow-based global probabilistic models for time series.
- [ ] **Neural Prophet** - Neural network forecaster inspired by Prophet with global components.
- [ ] **TFT with probabilistic outputs** - Temporal Fusion Transformer configured with probabilistic/quantile outputs.

### Tree-based and classical global forecasters

- [ ] **LightGBM for forecasting** - Global gradient-boosted tree models for forecasting.
- [ ] **XGBoost for forecasting** - Global XGBoost-based forecasting pipelines.
- [ ] **CatBoost for forecasting** - Categorical-boosting models adapted for time series.
- [ ] **NGBoost for time series** - Probabilistic gradient boosting models for forecasting.
- [ ] **Random Forest global forecasters** - Global random forest models for multiple series.
- [ ] **Extra Trees global forecasters** - Extremely randomized trees for global forecasting.

### Traditional global / panel / hierarchical models

- [ ] **Global pooled regression models** - Linear or generalized linear models pooled across series.
- [ ] **Fixed effects panel models** - Fixed-effects panel regressions for forecasting tasks.
- [ ] **Random effects panel models** - Random-effects panel models for grouped series.
- [ ] **Bayesian hierarchical models** - Hierarchical Bayesian forecasting models across series.
- [ ] **Multi-level models** - Mixed-effects and hierarchical multi-level time series models.
- [ ] **Meta-learning forecasters** - Models that transfer forecasting knowledge across tasks/series.

### Hybrid global models

- [ ] **N-BEATS variants (Neural Basis Expansion Analysis)** - Variants and hybrids of N-BEATS architectures.
- [ ] **Exponential smoothing RNN hybrids** - Models combining exponential smoothing with RNNs.
- [ ] **ESRNN** - Exponential smoothing with recurrent neural networks.
- [ ] **SMYL and related hybrids** - Hybrid statistical-neural forecasting methods.

---

*Last updated: February 2026*
