# metric_forecasting_proba Wishlist

Wishlist of performance metrics for probabilistic time series forecasting requested for implementation in sktime.

## How to Add Items

1. Open an issue on the [sktime issue tracker](https://github.com/sktime/sktime/issues) describing the metric
2. Add the item below following the format template
3. Submit a PR with your addition

## Format Template

```markdown
- [ ] **Metric Name** - Brief description
  - Paper: [Title](link) or Reference: [Package](link)
  - Issue: #1234
```

## Wishlist Items

- [ ] **CRPS (Continuous Ranked Probability Score)** - Proper scoring rule for full predictive distributions.
- [ ] **Ranked Probability Score (RPS)** - Discrete analogue of CRPS for categorical outcomes.
- [ ] **Quantile Score/Loss (all quantiles)** - Quantile-based scoring for probabilistic forecasts.
- [ ] **Winkler Score (Interval Score)** - Interval score for prediction intervals.
- [ ] **Pinball Loss** - Standard quantile regression loss for probabilistic forecasts.
- [ ] **Energy Score** - Proper scoring rule for multivariate predictive distributions.
- [ ] **Variogram Score** - Score focusing on dependence structures in multivariate forecasts.
- [ ] **Dawid-Sebastiani Score** - Proper score for Gaussian predictive distributions.
- [ ] **Logarithmic Score / LogScore** - Log-likelihood based proper scoring rule.
- [ ] **Ignorance Score** - Synonym of negative log-likelihood for probabilistic forecasts.
- [ ] **Brier Score for probabilistic forecasts** - Proper scoring rule for binary/finite outcomes.
- [ ] **CRPS decomposition (reliability + resolution)** - Decomposition tools for CRPS diagnostics.

### Calibration metrics

- [ ] **PIT (Probability Integral Transform) histograms** - Calibration diagnostics for continuous forecasts.
- [ ] **Reliability diagrams** - Visual and quantitative reliability assessment.
- [ ] **ECE (Expected Calibration Error)** - Scalar measure of miscalibration.
- [ ] **MCE (Maximum Calibration Error)** - Worst-case calibration error across bins.
- [ ] **Beta calibration metrics** - Beta-calibrated evaluation frameworks.

### Sharpness and rationality

- [ ] **Sharpness metrics** - Measures of concentration of predictive distributions.
- [ ] **Forecast rationality tests** - Statistical tests for rational probabilistic forecasts.
- [ ] **Density forecast evaluation** - General density forecast goodness-of-fit measures.
- [ ] **Censored likelihood scores** - Scores for censored or truncated predictive distributions.
- [ ] **Diebold-Mariano test for probabilistic forecasts** - Comparative test adapted to probabilistic scores.
- [ ] **Proper scoring rules (general framework)** - Generic framework for proper scoring rule evaluation.

---

*Last updated: February 2026*
