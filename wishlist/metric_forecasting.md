# metric_forecasting Wishlist

Wishlist of performance metrics for time series forecasting (point forecasts) requested for implementation in sktime.

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

- [ ] **MASE (Mean Absolute Scaled Error)** - Scale-free error metric relative to naive forecast.
- [ ] **RMSSE (Root Mean Squared Scaled Error)** - Squared and scaled variant of MASE.
- [ ] **MARRE (Mean Absolute Ranged Relative Error)** - Range-normalized absolute relative error.
- [ ] **sMAPE (symmetric MAPE)** - Symmetric mean absolute percentage error.
- [ ] **MAAPE (Mean Arctangent Absolute Percentage Error)** - Robust alternative to MAPE using arctangent.
- [ ] **RMSPE (Root Mean Squared Percentage Error)** - Root mean square of percentage errors.
- [ ] **Geometric RMSE** - Geometric mean based RMSE variant.
- [ ] **MDA (Mean Directional Accuracy)** - Fraction of correctly predicted directions of change.
- [ ] **GMRAE (Geometric Mean Relative Absolute Error)** - Geometric mean of relative absolute errors.
- [ ] **RelMAE / RelRMSE** - Relative MAE/RMSE compared to baseline forecasts.
- [ ] **MSIS (Mean Scaled Interval Score)** - Scaled interval score for predictive intervals.
- [ ] **AvgRelMAE** - Average relative MAE over series or horizons.
- [ ] **Theil's U statistic** - Relative forecast accuracy statistic compared to naive baseline.
- [ ] **Forecast Skill Score (FSS)** - Skill scores relative to baseline models.
- [ ] **Normalized RMSE** - RMSE normalized by range, mean, or other scale.
- [ ] **Coefficient of Variation of RMSE** - RMSE scaled by mean to obtain CV(RMSE).
- [ ] **Trimmed MAE/RMSE** - Robust MAE/RMSE with trimming of extreme errors.
- [ ] **Huber loss** - Robust loss combining L1 and L2 regions.
- [ ] **Log-Cosh loss** - Smooth robust loss for forecasting errors.
- [ ] **Quantile loss (deterministic version)** - Quantile-based loss for deterministic quantile forecasts.
- [ ] **RMDSPE (Root Median Squared Percentage Error)** - Median-based robust percentage error metric.
- [ ] **Directional accuracy metrics** - Additional metrics for sign/direction correctness.
- [ ] **Sign accuracy** - Proportion of correctly predicted sign of change.

---

*Last updated: February 2026*
