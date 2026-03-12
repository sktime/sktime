# reconciler Wishlist

Wishlist of time series reconciliation transformers requested for implementation in sktime.

## How to Add Items

1. Open an issue on the [sktime issue tracker](https://github.com/sktime/sktime/issues) describing the reconciler
2. Add the item below following the format template
3. Submit a PR with your addition

## Format Template

```markdown
- [ ] **Reconciler Name** - Brief description
  - Paper: [Title](link) or Reference: [Package](link)
  - Issue: #1234
```

## Wishlist Items

### Base reconciliation methods

- [ ] **Bottom-up** - Base series forecasts aggregated up the hierarchy.
- [ ] **Top-down (average historical proportions)** - Disaggregation using average historical proportions.
- [ ] **Top-down (proportions of historical averages)** - Disaggregation via proportions of series means.
- [ ] **Top-down (forecast proportions)** - Forecast-based proportional disaggregation.
- [ ] **Gross method / Gross-Sohl variants** - Classical top-down reconciliation methods.
- [ ] **Middle-out** - Combination of bottom-up and top-down from an intermediate level.
- [ ] **Level-conditional forecasting** - Level-specific reconciliation strategies.

### Optimal reconciliation (MinT family)

- [ ] **MinT-OLS (Ordinary Least Squares)** - Minimum trace reconciliation with OLS covariance.
- [ ] **MinT-WLS (Weighted Least Squares)** - MinT with heterogeneous variances/weights.
- [ ] **MinT with structural scaling** - Structure-based scaling of covariance matrices.
- [ ] **MinT with variance scaling** - Variance-based scaling and shrinkage.
- [ ] **MinT-Shrink** - Shrinkage covariance estimators for MinT.
- [ ] **MinT-Sample** - Sample covariance-based MinT reconciliation.

### Advanced reconciliation approaches

- [ ] **ERM (Empirical Risk Minimization) reconciliation** - Reconciliation via risk minimization frameworks.
- [ ] **PERMBU (Partially-Expected Reconciliation)** - Partially-expected bottom-up reconciliation.
- [ ] **Forecast combination reconciliation** - Reconciliation through linear or nonlinear forecast combinations.
- [ ] **State-space reconciliation** - Reconciliation based on coherent state space models.
- [ ] **Bayesian reconciliation** - Bayesian coherent forecasting with hierarchical priors.
- [ ] **Bootstrap reconciliation** - Bootstrap-based uncertainty and reconciliation schemes.
- [ ] **Cross-temporal reconciliation** - Joint cross-sectional and temporal aggregation reconciliation.
- [ ] **Game-theoretic reconciliation** - Reconciliation as allocation games with coherence constraints.
- [ ] **Optimal combination reconciliation** - Optimal combinations of base forecasts under coherence.
- [ ] **Reconciliation with constraints (non-negativity, bounds)** - Coherent forecasts subject to constraints.
- [ ] **Immutable forecasts reconciliation** - Reconciliation respecting immutability constraints.
- [ ] **Sparse reconciliation** - Sparsity-promoting approaches for large hierarchies.
- [ ] **Robust reconciliation** - Robust reconcilers under outliers or model misspecification.
- [ ] **Trace minimization with custom covariance** - User-specified covariance structures for MinT-like schemes.

### Probabilistic and grouped/temporal reconciliation

- [ ] **Joint probabilistic reconciliation** - Reconciliation for full joint forecast distributions.
- [ ] **Gaussian reconciliation** - Coherent multivariate Gaussian reconciliation.
- [ ] **Bootstrap-based probabilistic coherence** - Probabilistic reconciliation via bootstrap ensembles.
- [ ] **GAMLSS reconciliation** - Distributional regression approaches for coherent hierarchical forecasts.
- [ ] **Copula-based reconciliation** - Reconciliation using copula-based dependence models.
- [ ] **Temporal aggregation reconciliation** - Reconciliation across temporal aggregation levels.
- [ ] **Grouped hierarchy reconciliation** - Reconciliation for grouped time series structures.
- [ ] **Graph-based reconciliation** - Reconciliation over general graph structures, not only trees.

---

*Last updated: February 2026*
