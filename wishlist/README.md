# sktime Wishlist

This folder contains curated wishlists of algorithms and objects requested for implementation in sktime.

## Purpose

- **Streamline feature requests**: Core devs can review, approve/reject, and prioritize algorithm requests
- **Bulk management**: Easier to manage many requests than scattered issues
- **Contributor guidance**: Helps contributors find concrete tasks to work on
- **Contribution tracking**: Adding to the wishlist counts as a proper contribution

## How to Use

### For Contributors Looking for Tasks

1. Browse the wishlist files below by scitype (e.g., `forecaster.md`, `classifier.md`)
2. Find an item marked as `[ ]` (not yet implemented)
3. Check the linked issue for details and discussion
4. Comment on the issue to claim the task
5. Implement and submit a PR!

### For Adding New Items

When adding a new algorithm to the wishlist:

1. **Create an issue first** - Open an issue on the [sktime issue tracker](https://github.com/sktime/sktime/issues) describing the algorithm
2. **Add to the appropriate wishlist file** - Include:
   - Name of the algorithm/object
   - Brief description
   - Link to paper or reference implementation
   - Link to the GitHub issue
3. **Submit a PR** - Adding to the wishlist is a contribution!

## Wishlist Files by Scitype

| Scitype | Description | File |
|---------|-------------|------|
| Aligner | Time series alignment algorithms | [aligner.md](aligner.md) |
| Classifier | Time series classification algorithms | [classifier.md](classifier.md) |
| Clusterer | Time series clustering algorithms | [clusterer.md](clusterer.md) |
| Detector | Anomaly, outlier, and change point detectors | [detector.md](detector.md) |
| Early Classifier | Early time series classifiers | [early_classifier.md](early_classifier.md) |
| Forecaster | Time series forecasting algorithms | [forecaster.md](forecaster.md) |
| Metric | Performance metrics | [metric.md](metric.md) |
| Network | Deep learning network architectures | [network.md](network.md) |
| Param Estimator | Parameter fitting estimators | [param_est.md](param_est.md) |
| Regressor | Time series regression algorithms | [regressor.md](regressor.md) |
| Splitter | Time series cross-validation splitters | [splitter.md](splitter.md) |
| Transformer | Time series transformations | [transformer.md](transformer.md) |

## Item Format

Each wishlist item should follow this format:

```markdown
- [ ] **Algorithm Name** - Brief description
  - Paper: [Title](link) or Reference: [Package](link)
  - Issue: #1234
  - Priority: low/medium/high (optional)
  - Notes: Any additional context (optional)
```

## Status Legend

- `[ ]` - Not yet implemented, available for contributors
- `[x]` - Implemented in sktime
- `[~]` - Work in progress (link to PR in notes)

## Questions?

Join the [sktime Discord](https://discord.com/invite/54ACzaFsn7) and ask in the `#dev-chat` channel!
