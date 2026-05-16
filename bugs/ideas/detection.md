Extending ClaSP: Semi-Supervised Change Point Detection with Incomplete Labels
Core Idea

Extend ClaSP's unsupervised time series segmentation to a (self-)supervised setting where:

    We have partial change point labels (only 1s, no reliable 0s) → PU learning setting
    Exogenous variables are available
    A held-out test set with complete annotations exists for final evaluation

Approach: Split the time series into evenly sized windows, use the partial labels to train a classifier (RF, LightGBM), and iteratively improve the label set.
Feature Engineering

    Summary statistics + lags: Hand-crafted features over each window (mean, variance, slope, spectral features, etc.) for both target and exogenous channels
    Lags before windowing: Enrich each timestamp with temporal context before the reduction step, so window-level statistics implicitly capture dynamics, not just snapshots
    Alternative: catch22 (curated 22 features from hctsa) or tsfresh (brute-force hundreds of features with relevance filtering), though hand-crafted features are preferable when domain knowledge exists

Label Cleaning: Confident Learning Loop

Borrowed from Northcutt et al. (2021) — estimate the joint distribution of noisy and true labels using out-of-fold predicted probabilities, then flag disagreements.

    Missing annotations (false negatives): Unlabeled window where model predicts p ≈ 1 across folds → promote to positive
    False positive labels: Labeled window where model predicts p ≈ 0 across folds → demote / exclude from training
    Use Northcutt's self-confidence threshold (average predicted probability per class) rather than an arbitrary threshold like 0.5

Using OOB Instead of CV

    Random Forest's out-of-bag predictions are a direct substitute for out-of-fold predictions
    Each sample gets predictions from ~1/3 of trees that never saw it during training → unbiased probability estimates for free
    max_samples parameter controls bootstrap sample size per tree: lower values → more OOB trees per sample → more stable probability estimates (at the cost of weaker individual trees)
    Makes each relabeling iteration a single model fit rather than k-fold CV

Iteration Protocol

    Train RF with oob_score=True
    Read oob_decision_function_ for per-window probabilities
    Apply confident learning thresholds to promote and demote labels
    Retrain on updated labels
    Repeat for N iterations, snapshot model at each step
    Evaluate all snapshots on held-out test set → performance-vs-iterations curve

Convergence Criteria (for simpler self-training variant)

    Label stability: Stop when fewer than ε fraction of windows change labels between iterations
    CV/OOB score plateau: Stop when AUC improvement falls below threshold δ
    Confidence calibration: Stop when predicted probability distribution stops shifting

Meta-Algorithm for Label Optimization

Train a meta-algorithm that predicts +1, 0, -1 for each window:

    +1 → is a change point
    0 → is not a change point
    -1 → should be excluded from training (false positive)

The objective is the downstream RF's CV/OOB score — the meta-algorithm should label data such that the RF performs best.

Challenge: 3^N search space is combinatorial. Options:

    Policy gradient / RL approach
    Bilevel optimization (outer loop: labels, inner loop: RF)
    Resembles Snorkel-style data programming but with a discriminative objective

Overfitting concern: Iterating over the same dataset repeatedly. Mitigated by the held-out test set — iterate as aggressively as needed, the test set is the final arbiter.
False Positive Detection Specifically
Leave-One-Out Influence via OOB

For each labeled positive, check its OOB prediction. If the forest consistently predicts negative when it hasn't memorized that sample, it's suspicious. Removing it should improve performance.
Consensus Across Perturbations (Stability Selection on Labels)

Train multiple RFs with different seeds, feature subsets, bootstrap samples. True change points are detected consistently; false positives are "fragile" — only classified correctly by models that memorized them.
Confident Learning Handles Both Directions

The framework naturally supports demotion (labeled 1 → suspected false positive) alongside promotion (unlabeled → suspected missing annotation). No separate mechanism needed.
Hyperparameter Optimization

    Optuna with time-aware CV: Hold out last x days of the time series for evaluation, do HPO with temporal CV on the remaining data
    Include in the search space:
        Asymmetric loss / scale_pos_weight (to handle PU label contamination)
        Tree hyperparameters (depth, min samples, n_estimators)
        max_samples (row subsampling, affects OOB quality)
        max_features (column subsampling per split)
        Possibly window size itself
    Gap between folds: Ensure sufficient gap to prevent autocorrelation leakage

Tuning max_samples for Label Cleaning Quality

Unusual but natural objective: optimize max_samples not for predictive accuracy but for stability of OOB probability estimates. Measure consistency of label assignments across random seeds for a given setting.
Architectural Extensions
Curriculum Learning on Relabeling

Order unlabeled windows by distance-to-decision-boundary. Promote easy cases first (high confidence), retrain, then move toward ambiguous middle. Produces a "difficulty score" per window — hard-to-classify windows are likely gradual regime changes rather than sharp change points.
Multi-Scale Windowing

Run the full pipeline at multiple window sizes and ensemble. A window flagged at 3/5 scales is more trustworthy than 1/5. Also helps with the boundary problem where a change point falls between windows. Precedent: ClaSP*.
Temporal Consistency Regularization

Add a CRF or HMM layer on top of per-window RF probabilities to enforce temporal smoothness. HMM transition matrix encodes prior on segment length. Cheap to fit, reduces false positives from noisy probability estimates.
Change Point Type Embeddings

Not all change points are the same (mean shift, variance change, trend break, distributional shift). Cluster the RF's feature importance vectors across detected change points → taxonomy of change types for free. Operationally valuable for real-world deployment.
Counterfactual Exogenous Analysis (SHAP)

For each detected change point, use SHAP on the RF to identify which exogenous variables best explain the transition. Compare feature attributions near change points vs stable regimes. Gives practitioners the "why" alongside the "where."
Active Learning

After each confident learning iteration, surface the top-k most uncertain windows to a human annotator instead of auto-promoting/demoting. Maximum ROI per human label. The confident learning framework provides the selection criterion for free.
Constraints and Considerations

    SCAR assumption: Labeled positives should be selected completely at random from all true positives. If annotators only label obvious change points, the classifier learns easy ones and self-training reinforces that bias.
    Spatial coherence: Newly promoted positives are likely adjacent to existing ones. Can constrain the relabeling step accordingly.
    Confirmation bias in self-training: Model becomes confident in its own mistakes. Mitigate with cautious thresholds (require all CV folds / high OOB consensus), regularization penalizing deviation from original labels, or limited flips per iteration.
    Test set integrity: Annotations should come from a different process/annotator than training labels to avoid measuring agreement with a biased annotator.
    LightGBM vs RF tradeoff: LightGBM has better calibrated probabilities out of the box (matters for confident learning) but lacks OOB estimates. RF gives OOB for free but may need calibration.

Suggested Pipeline Order

    Feature engineering with lags + summary statistics
    Windowing
    Optuna HPO (including asymmetric loss, tree params, window size) with time-aware CV
    Confident learning loop with OOB estimates — both promotion and demotion — for 2-3+ iterations
    Temporal HMM smoothing
    SHAP for change point attribution
    Evaluate on held-out test set

