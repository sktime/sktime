# transformer-pairwise Wishlist

Wishlist of pairwise transformer (tabular) algorithms requested for implementation in sktime.

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

### Elastic distance measures

- [ ] **DTW (Dynamic Time Warping)** - Classic DTW alignment-based distance.
- [ ] **Constrained DTW (Sakoe-Chiba, Itakura)** - DTW with global warping constraints.
- [ ] **Derivative DTW (first/second derivatives)** - DTW on derivative-transformed series.
- [ ] **Weighted DTW / WDTW** - DTW with position- or gap-dependent weights.
- [ ] **Shape DTW** - DTW variants emphasizing local shape similarity.
- [ ] **DTW with variable warping window** - Adaptive or learned warping window DTW.
- [ ] **Subsequence DTW** - DTW for subsequence matching within longer series.
- [ ] **Open-begin/open-end DTW** - DTW variants with relaxed boundary conditions.
- [ ] **FastDTW / approximate DTW** - Approximate and scalable DTW algorithms.
- [ ] **SparseDTW** - Sparse matrix-based DTW approximations.
- [ ] **Soft-DTW** - Differentiable DTW-based distance suitable for learning.

### Edit-based distances

- [ ] **ERP (Edit Distance with Real Penalty)** - Edit distance adapted to real-valued sequences.
- [ ] **EDR (Edit Distance on Real sequences)** - Edit distance with tolerance thresholds.
- [ ] **LCSS (Longest Common Subsequence)** - Similarity based on longest common subsequences.
- [ ] **MSM (Move-Split-Merge)** - Edit distance with move/split/merge operations for time series.
- [ ] **TWE (Time Warp Edit)** - Distance mixing time warping and edit operations.
- [ ] **SWALE (Subsequence Weighted Alignment Evaluation)** - Weighted alignment-based distance.

### Shape-based distances

- [ ] **SBD (Shape-Based Distance)** - Distance based on normalized cross-correlation.
- [ ] **CID (Complexity-Invariant Distance)** - Distance adjusted for series complexity.
- [ ] **Shapelets distance** - Distances based on discriminative shapelet features.
- [ ] **SAX-based distances (MINDIST, MINDIST_SAX)** - Distances in symbolic aggregate approximation space.
- [ ] **BOSS distance** - Distance in Bag-of-SFA-Symbols representation.
- [ ] **Angular distance** - Angle-based distances between normalized time series.
- [ ] **Slope distance** - Distances focusing on local slope/gradient patterns.

### Frequency and spectral distances

- [ ] **DFT/FFT distance** - Distances based on Fourier coefficient representations.
- [ ] **Spectral distance** - Distances between spectral density estimates.
- [ ] **Cepstral distance** - Distances in the cepstral domain.
- [ ] **Periodogram distance** - Distances between periodograms.
- [ ] **Power spectral density distance** - PSD-based distance measures.
- [ ] **Wavelet distance / scaleogram distance** - Wavelet or time-frequency based distances.

### Symbolic and dictionary-based distances

- [ ] **SAX distance** - Distances defined in the SAX symbolic space.
- [ ] **BOSS / SFA-based distances** - Dictionary-based distances on symbolic Fourier approximations.
- [ ] **SAX-VSM distance** - Distances using vector space model representations of symbolic words.
- [ ] **Word-based distances** - Distances based on word or n-gram frequency distributions.

### Statistical and information-theoretic distances

- [ ] **Pearson correlation distance** - 1 minus Pearson correlation.
- [ ] **Spearman correlation distance** - Rank correlation-based distances.
- [ ] **Kendall tau distance** - Distance based on Kendall rank correlation.
- [ ] **Partial correlation distance** - Distances correcting for confounders.
- [ ] **Distance correlation** - Non-linear dependence-based distance.
- [ ] **Maximal information coefficient (MIC)** - Dependency-based similarity measure.
- [ ] **Mutual information distance** - Distances based on mutual information.
- [ ] **Kullback-Leibler divergence** - Divergence between probabilistic time series models.
- [ ] **Jensen-Shannon divergence** - Symmetrized and smoothed version of KL divergence.
- [ ] **Bhattacharyya distance** - Overlap-based distance between distributions.
- [ ] **Hellinger distance** - Metric derived from Bhattacharyya coefficient.
- [ ] **Wasserstein / Earth Mover's distance** - Optimal transport distances for time series distributions.

### Geometric distances

- [ ] **Euclidean distance (L2)** - Standard pointwise L2 distance.
- [ ] **Manhattan distance (L1)** - L1 distance over time.
- [ ] **Minkowski distance (Lp)** - General Lp distances.
- [ ] **Chebyshev distance** - Max-norm distance across time.
- [ ] **Mahalanobis distance** - Covariance-aware distance over features/time.
- [ ] **Fréchet distance** - Curve-based distance between time series trajectories.
- [ ] **Hausdorff distance** - Set/curve-based distance between trajectories.
- [ ] **Procrustes distance** - Shape-based alignment distance.
- [ ] **Geodesic distance** - Distances on manifolds representing time series.

### Compression-based distances

- [ ] **NCD (Normalized Compression Distance)** - Distance induced by compression algorithms.
- [ ] **CDM (Compression-based Dissimilarity Measure)** - Compression-based dissimilarity for sequences.
- [ ] **Kolmogorov complexity approximations** - Compression-based proxies to algorithmic complexity distances.

### Model-based distances

- [ ] **ARMA/ARIMA model distance (e.g. Piccolo)** - Distances between fitted parametric models.
- [ ] **Cepstral distance for ARMA models** - Model-based cepstral distances.
- [ ] **Likelihood-based distances** - Distances induced by likelihood or pseudo-likelihoods.
- [ ] **State-space model distances** - Distances between state-space parameterizations/filters.

### Kernel similarities

- [ ] **RBF/Gaussian kernel** - Radial basis function kernels on time series representations.
- [ ] **Polynomial kernel** - Polynomial kernels on feature or raw series.
- [ ] **Laplacian kernel** - L1-based kernel similarities.
- [ ] **String kernels** - Kernels defined on symbolic representations of series.
- [ ] **Global Alignment Kernel (GAK)** - DTW-inspired positive definite kernel.
- [ ] **Signature kernel** - Kernels based on path signatures.
- [ ] **Reservoir kernel** - Kernels induced by random recurrent feature maps.

### Complexity, entropy, and feature-based distances

- [ ] **Permutation/sample/approximate entropy distances** - Distances based on entropy-like complexity measures.
- [ ] **Kolmogorov and LZ complexity distances** - Distances based on sequence complexity estimators.
- [ ] **Catch22 and TSFresh-based distances** - Distances in interpretable feature spaces.
- [ ] **Signature-based distance** - Distances in signature feature space of time series.
- [ ] **Topological data analysis distances** - Distances via persistent homology and related TDA summaries.

### Special-purpose and multivariate distances

- [ ] **DISSIM** - Specialized dissimilarity for time series.
- [ ] **PDC (Partial Directed Coherence) distance** - Causality-based spectral distances.
- [ ] **Granger causality-based distance** - Distances based on causal influence patterns.
- [ ] **Transfer entropy distance** - Information flow-based distances.
- [ ] **Cross-recurrence / recurrence plot-based distances** - Distances defined via recurrence structures.
- [ ] **Synchronization measures (phase synchronization, etc.)** - Phase and synchronization-based similarities.
- [ ] **Multivariate LCSS-MS / ED-MS** - Multivariate LCSS and Euclidean distances.
- [ ] **Dimension-weighted distances** - Distances with learned or specified dimension weights.
- [ ] **Independent component distances** - Distances in independent component representations.

---

*Last updated: February 2026*
