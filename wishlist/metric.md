# Metric Wishlist

Wishlist of performance metrics requested for implementation in sktime.

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

---

## Forecasting Metrics

- [ ] **MSIS** - Mean Scaled Interval Score
  - Paper: [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
  - Issue: TBD

- [ ] **QuantileLoss** - Quantile loss for probabilistic forecasting
  - Reference: Standard quantile loss
  - Issue: TBD

- [ ] **CRPS** - Continuous Ranked Probability Score (improvements)
  - Paper: [Strictly Proper Scoring Rules](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)
  - Issue: TBD

## Classification Metrics

- [ ] **Time-aware Accuracy** - Accuracy weighted by earliness
  - Reference: Early classification metrics
  - Issue: TBD

## Detection Metrics

- [ ] **NAB Score** - Numenta Anomaly Benchmark scoring
  - Reference: [NAB](https://github.com/numenta/NAB)
  - Issue: TBD

- [ ] **Range-based F1** - F1 score for range-based anomaly detection
  - Paper: [Precision and Recall for Time Series](https://arxiv.org/abs/1803.03639)
  - Issue: TBD

- [ ] **VUS** - Volume Under Surface for threshold-free evaluation
  - Paper: [VUS metrics](https://arxiv.org/abs/2205.08354)
  - Issue: TBD

---

*Last updated: February 2026*
