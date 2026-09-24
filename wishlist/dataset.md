# dataset Wishlist

Wishlist of datasets requested for inclusion in sktime.

## How to Add Items

1. Open an issue on the [sktime issue tracker](https://github.com/sktime/sktime/issues) describing the dataset
2. Add the item below following the format template
3. Submit a PR with your addition

## Format Template

```markdown
- [ ] **Dataset Name** - Brief description
  - Source: [Link](link)
  - Issue: #1234
```

## Wishlist Items

- [ ] **General synthetic time series generators** - Flexible generators for configurable synthetic univariate and multivariate series.
- [ ] **Noise generators (Gaussian, Laplacian, Cauchy, Student-t, Lévy)** - Pluggable noise processes for augmenting or simulating data.
- [ ] **Pattern generators (seasonal, trend, cyclic, chaotic)** - Reusable generators for common time series patterns and regimes.
- [ ] **Multi-modal time series generators** - Generators for time series with multiple regimes or modes.
- [ ] **ARMA/ARIMA synthetic generators** - Generators for stationary and non-stationary ARMA/ARIMA processes.
- [ ] **State space model generators** - Synthetic data from linear and non-linear state space models.
- [ ] **Regime-switching time series** - Hidden Markov and Markov-switching model datasets.
- [ ] **Jump diffusion processes** - Time series with jumps and diffusion components (e.g., Merton models).
- [ ] **Fractional integration processes (ARFIMA)** - Long-memory synthetic time series.
- [ ] **Long memory process generators** - General tools for simulating long-range dependent processes.
- [ ] **Copula-based multivariate generators** - Multivariate time series built via copulas with flexible dependence.
- [ ] **Functional time series generators** - Datasets where each observation is a function observed over time.
- [ ] **Point process generators** - Event time processes (e.g., Hawkes, Poisson) with temporal structure.
- [ ] **Event sequence generators** - Synthetic sequences of timestamped events with marks and dependencies.

---

*Last updated: February 2026*
