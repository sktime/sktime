# Changelog

You can find the sktime changelog on our [website](https://www.sktime.net/en/latest/changelog.html).

## Unreleased

### Enhancements

* RecursiveReductionForecaster performance: added guarded optimized recursive prediction paths (`_predict_out_of_sample_v2_local`, `_predict_out_of_sample_v2_global`, and tail-window fast path `_predict_out_of_sample_v1_fasttail`). These reduce repeated full-history lag recomputation, yielding substantial speedups (observed >20x in local, single-series, no-exogenous scenarios) while preserving exact numerical parity under enforced guard conditions. Falls back automatically to legacy logic when any safety condition (multiindex/pooling != local/exogenous/imputation/custom estimator fallback) is not met. PR #7380.
