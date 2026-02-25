# Detector Wishlist

Wishlist of anomaly, outlier, and change point detection algorithms requested for implementation in sktime.

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

## Anomaly Detection

- [ ] **Anomaly Transformer** - Transformer-based anomaly detection
  - Paper: [Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy](https://arxiv.org/abs/2110.02642)
  - Issue: TBD

- [ ] **USAD** - UnSupervised Anomaly Detection on multivariate time series
  - Paper: [USAD: UnSupervised Anomaly Detection on Multivariate Time Series](https://dl.acm.org/doi/10.1145/3394486.3403392)
  - Issue: TBD

- [ ] **OmniAnomaly** - Stochastic RNN for multivariate time series anomaly detection
  - Paper: [Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network](https://dl.acm.org/doi/10.1145/3292500.3330672)
  - Issue: TBD

- [ ] **THOC** - Temporal Hierarchical One-Class network
  - Paper: [A Deep Learning Model for Anomaly Detection in Time Series](https://arxiv.org/abs/2302.03091)
  - Issue: TBD

## Change Point Detection

- [ ] **BOCPD** - Bayesian Online Change Point Detection
  - Paper: [Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742)
  - Issue: TBD

- [ ] **PELT** - Pruned Exact Linear Time (ruptures interface)
  - Reference: [ruptures](https://centre-borelli.github.io/ruptures-docs/)
  - Issue: TBD

- [ ] **KernelCPD** - Kernel Change Point Detection
  - Reference: [ruptures](https://centre-borelli.github.io/ruptures-docs/)
  - Issue: TBD

## Outlier Detection

- [ ] **SR (Spectral Residual)** - Spectral residual for time series
  - Paper: [Time-Series Anomaly Detection Service at Microsoft](https://arxiv.org/abs/1906.03821)
  - Issue: TBD

- [ ] **Matrix Profile Variants** - Additional matrix profile algorithms
  - Reference: [STUMPY](https://stumpy.readthedocs.io/)
  - Issue: TBD

---

*Last updated: February 2026*
