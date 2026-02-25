# Regressor Wishlist

Wishlist of time series regression algorithms requested for implementation in sktime.

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

## Kernel Methods

- [ ] **GPR for Time Series** - Gaussian Process Regression for time series
  - Reference: scikit-learn GPR wrapper
  - Issue: TBD

---

## Already Implemented in sktime

The following algorithms are already available in sktime:

- **ResNet for TSR** - `ResNetRegressor` in `sktime.regression.deep_learning`
- **InceptionTime for TSR** - `InceptionTimeRegressor` in `sktime.regression.deep_learning`
- **ROCKET Regressor** - `RocketRegressor` in `sktime.regression.kernel_based`
- **FCN Regressor** - `FCNRegressor` in `sktime.regression.deep_learning`
- **CNN Regressor** - `CNNRegressor` in `sktime.regression.deep_learning`

---

*Last updated: February 2026*
