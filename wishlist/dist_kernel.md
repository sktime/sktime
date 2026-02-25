# Distance & Kernel Wishlist

Wishlist of pairwise distance and kernel algorithms for time series requested for implementation in sktime.

This covers both `transformer-pairwise` (tabular) and `transformer-pairwise-panel` (time series) scitypes.

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

## Time Series Distances

- [ ] **Soft-DTW** - Differentiable variant of Dynamic Time Warping
  - Paper: [Soft-DTW: a Differentiable Loss Function for Time-Series](https://arxiv.org/abs/1703.01541)
  - Issue: TBD

- [ ] **Time Warp Edit Distance (TWED)** - Elastic distance with stiffness parameter
  - Paper: [Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching](https://ieeexplore.ieee.org/document/4752764)
  - Issue: TBD

- [ ] **Move-Split-Merge (MSM)** - Edit distance for time series
  - Paper: [MSM: A metric for time series](https://link.springer.com/article/10.1007/s10618-012-0275-2)
  - Issue: TBD

## Time Series Kernels

- [ ] **Global Alignment Kernel (GAK)** - Kernel based on DTW alignment
  - Paper: [A Global Averaging Method for Dynamic Time Warping](https://arxiv.org/abs/1003.3339)
  - Issue: TBD

- [ ] **Signature Kernel** - Kernel based on path signatures
  - Paper: [The Signature Kernel is the solution of a Goursat PDE](https://arxiv.org/abs/2006.14794)
  - Issue: TBD

---

## Already Implemented in sktime

The following algorithms are already available in sktime:

- **DTW** - `DtwDist` in `sktime.dists_kernels`
- **Edit Distance on Real Sequences (EDR)** - `EdrDist` in `sktime.dists_kernels`
- **Longest Common Subsequence (LCSS)** - `LcssDist` in `sktime.dists_kernels`
- **Euclidean Distance** - `EuclideanDist` in `sktime.dists_kernels`
- **Shape-based Distance (SBD)** - Available via clustering

---

*Last updated: February 2026*
