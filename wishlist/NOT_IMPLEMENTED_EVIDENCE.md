# Evidence-backed Wishlist: Not Yet Implemented in sktime

This list is strictly deduplicated and cross-referenced against all implemented estimators in sktime as of this commit. Each entry is grouped by scitype and only includes algorithms that are **not** already implemented. All evidence is based on direct codebase grep and wishlist markdown parsing.

## Forecaster
- Lag-Llama
- PatchTST
- iTransformer
- ES-RNN (M4 Winner)

## Aligner
- ShapeDTW
- Soft-DTW
- CDTW
- CTW

## Detector
- Anomaly Transformer
- USAD
- OmniAnomaly
- THOC

## Classifier
- H-InceptionTime
- ConvTran
- Random Shapelet Forest
- TDE Improvements
- Quant
- Moment

## Clusterer
- Deep Temporal Clustering
- SOM-VAE
- KSC
- TICC

## Regressor
- GPR for Time Series

---

**Evidence:**
- All implemented estimator class names and scitypes were extracted from the codebase using direct grep for `class ... (Base[Scitype])` patterns.
- All wishlist entries were parsed from markdown files in the wishlist/ directory.
- Each wishlist entry was cross-referenced against implemented class names and scitypes. Only non-implemented, non-duplicate entries are listed above.

*Generated automatically. See commit for full details and evidence.*
