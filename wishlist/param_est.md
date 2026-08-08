# Parameter Estimator Wishlist

Wishlist of parameter fitting estimators requested for implementation in sktime.

## How to Add Items

1. Open an issue on the [sktime issue tracker](https://github.com/sktime/sktime/issues) describing the estimator
2. Add the item below following the format template
3. Submit a PR with your addition

## Format Template

```markdown
- [ ] **Estimator Name** - Brief description
  - Paper: [Title](link) or Reference: [Package](link)
  - Issue: #1234
```

---

## Seasonality Detection

- [ ] **Autocorrelation-based Seasonality** - Detect seasonality via ACF
  - Reference: Standard time series analysis
  - Issue: TBD

- [ ] **Fourier-based Seasonality** - FFT-based seasonality detection
  - Reference: Spectral analysis methods
  - Issue: TBD

## Stationarity Tests

- [ ] **KPSS Test** - Kwiatkowski-Phillips-Schmidt-Shin test
  - Reference: statsmodels
  - Issue: TBD

- [ ] **Phillips-Perron Test** - Unit root test
  - Reference: statsmodels
  - Issue: TBD

## Order Selection

- [ ] **Auto ARIMA Order Selection** - Automatic (p,d,q) selection
  - Reference: pmdarima
  - Issue: TBD

---

*Last updated: February 2026*
