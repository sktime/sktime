# Splitter Wishlist

Wishlist of time series cross-validation splitters requested for implementation in sktime.

## How to Add Items

1. Open an issue on the [sktime issue tracker](https://github.com/sktime/sktime/issues) describing the splitter
2. Add the item below following the format template
3. Submit a PR with your addition

## Format Template

```markdown
- [ ] **Splitter Name** - Brief description
  - Paper: [Title](link) or Reference: [Package](link)
  - Issue: #1234
```

---

## Cross-Validation Schemes

- [ ] **Blocked Time Series Split** - Blocked CV for time series with gaps
  - Reference: Common CV strategy
  - Issue: TBD

- [ ] **Purged K-Fold** - K-Fold with purging for financial data
  - Paper: [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
  - Issue: TBD

- [ ] **Combinatorial Purged CV** - Combinatorial cross-validation with purging
  - Paper: [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
  - Issue: TBD

## Specialized Splitters

- [ ] **Seasonal Splitter** - Split respecting seasonal boundaries
  - Reference: Seasonal cross-validation
  - Issue: TBD

- [ ] **Gap Splitter** - Splitter with configurable gaps between train/test
  - Reference: Common for multi-step ahead forecasting
  - Issue: TBD

---

*Last updated: February 2026*
