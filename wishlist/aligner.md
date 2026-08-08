# Aligner Wishlist

Wishlist of time series alignment algorithms requested for implementation in sktime.

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

## Dynamic Time Warping Variants

- [ ] **ShapeDTW** - Shape-based DTW
  - Paper: [shapeDTW: shape Dynamic Time Warping](https://arxiv.org/abs/1606.01601)
  - Issue: TBD

- [ ] **Soft-DTW** - Soft Dynamic Time Warping (differentiable)
  - Paper: [Soft-DTW: a Differentiable Loss Function for Time-Series](https://arxiv.org/abs/1703.01541)
  - Issue: TBD

- [ ] **CDTW** - Constrained DTW variants
  - Reference: Various band constraints
  - Issue: TBD

## Other Alignment Methods

- [ ] **CTW** - Canonical Time Warping
  - Paper: [Canonical Time Warping for Alignment of Human Behavior](https://papers.nips.cc/paper/2009/hash/dd45045f8c68db9f54e70c67048d32e8-Abstract.html)
  - Issue: TBD

- [ ] **GAK** - Global Alignment Kernel
  - Paper: [Fast Global Alignment Kernels](https://arxiv.org/abs/1103.3785)
  - Issue: TBD

---

*Last updated: February 2026*
