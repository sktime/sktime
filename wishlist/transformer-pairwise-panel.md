# transformer-pairwise-panel Wishlist

Wishlist of pairwise transformer (panel/time series) algorithms requested for implementation in sktime.

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

## Wishlist Items

- [ ] **Panel DTW (Dynamic Time Warping)** - DTW-based distances and kernels defined over panels of series.
- [ ] **Panel Euclidean distance** - Euclidean distances aggregated across multiple series per panel.
- [ ] **Panel Manhattan distance** - L1 distances for multi-series panels.
- [ ] **Panel correlation-based distances** - Correlation-based measures extended to panels.
- [ ] **Panel Frobenius distance** - Frobenius norm distances on matrix- or tensor-shaped panels.
- [ ] **Cross-series similarity measures** - Measures capturing similarity patterns across multiple series.
- [ ] **Multi-series alignment algorithms** - Algorithms to jointly align multiple series within panels.
- [ ] **Subspace distances for panel data** - Subspace/projection-based distances for panels.
- [ ] **Tensor-based distances** - Distances on tensor representations of panel data.
- [ ] **Panel kernel similarities** - Kernels defined over panel objects (e.g., graph, tensor kernels).
- [ ] **Multi-series synchronization measures** - Synchronization and phase-locking metrics for panels.
- [ ] **Canonical correlation analysis distance** - Distances based on multivariate canonical correlations.
- [ ] **Grassmannian distance** - Distances between subspaces representing panel time series.
- [ ] **Procrustes distance for aligned series** - Shape-based distances for aligned multivariate series.
- [ ] **Multi-dimensional scaling distances** - Distances induced by MDS embeddings of panels.
- [ ] **Distance correlation for panels** - Distance correlation measures adapted to panel time series.

---

*Last updated: February 2026*
