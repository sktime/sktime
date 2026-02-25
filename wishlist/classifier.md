# Classifier Wishlist

Wishlist of time series classification algorithms requested for implementation in sktime.

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

## Deep Learning

- [ ] **H-InceptionTime** - Hierarchical InceptionTime
  - Paper: [H-InceptionTime](https://arxiv.org/abs/2204.07885)
  - Issue: TBD

- [ ] **ConvTran** - Convolutional Transformer for time series classification
  - Paper: [ConvTran: Improving Position Encoding of Transformers for Multivariate Time Series Classification](https://arxiv.org/abs/2304.03032)
  - Issue: TBD

## Shapelet-Based

- [ ] **Random Shapelet Forest** - Random forest with shapelet features
  - Reference: Various implementations
  - Issue: TBD

## Dictionary-Based

- [ ] **TDE Improvements** - Temporal Dictionary Ensemble improvements
  - Reference: sktime existing implementation improvements
  - Issue: TBD

## Interval-Based

- [ ] **Quant** - Quantile-based interval classifier
  - Paper: [Quant: A Minimalist Interval Method for Time Series Classification](https://arxiv.org/abs/2308.00928)
  - Issue: TBD

## Foundation Models

- [ ] **Moment** - Masked pre-training for time series
  - Paper: [MOMENT: A Family of Open Time-series Foundation Models](https://arxiv.org/abs/2402.03885)
  - Issue: TBD

---

## Already Implemented in sktime

The following algorithms are already available in sktime:

- **InceptionTime** - `InceptionTimeClassifier` in `sktime.classification.deep_learning`
- **ROCKET** - `RocketClassifier` in `sktime.classification.kernel_based`
- **MiniRocket** - `MiniRocketClassifier` in `sktime.classification.kernel_based`
- **MultiRocket** - `MultiRocketClassifier` in `sktime.classification.kernel_based`
- **HIVE-COTE** - `HIVECOTEV1`, `HIVECOTEV2` in `sktime.classification.hybrid`
- **BOSS** - `BOSSEnsemble` in `sktime.classification.dictionary_based`
- **TDE** - `TemporalDictionaryEnsemble` in `sktime.classification.dictionary_based`
- **TSFresh** - `TSFreshClassifier` in `sktime.classification.feature_based`

---

*Last updated: February 2026*
