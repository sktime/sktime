# metric_detection Wishlist

Wishlist of performance metrics for time series detection tasks requested for implementation in sktime.

## How to Add Items

1. Open an issue on the [sktime issue tracker](https://github.com/sktime/sktime/issues) describing the metric
2. Add the item below following the format template
3. Submit a PR with your addition

## Format Template

```markdown
- [ ] **Metric Name** - Brief description
  - Paper: [Title](link) or Reference: [Package](link)
  - Issue: #1234
```

## Wishlist Items

- [ ] **Precision/Recall for anomaly detection** - Standard precision and recall adapted to rare-event detection.
- [ ] **F1 score for anomalies** - Harmonic mean of precision and recall for anomaly events.
- [ ] **AUC-ROC for anomaly detection** - Area under ROC curve for anomaly scoring functions.
- [ ] **AUC-PR (Precision-Recall)** - Area under precision-recall curve for detection tasks.
- [ ] **NAB (Numenta Anomaly Benchmark) score** - Time-aware anomaly detection scoring metric.
- [ ] **Affiliation metrics** - Metrics capturing how well detected anomalies align with ground-truth events.
- [ ] **Range-based precision/recall** - Event-range aware precision and recall metrics.
- [ ] **Point-adjust metrics** - Point-adjusted scoring that tolerates offset detections within anomaly windows.
- [ ] **VUS (Volume Under Surface)** - Volume under ROC/PR surfaces for multi-threshold evaluation.
- [ ] **Time-to-detect metrics** - Latency from anomaly onset to detection.
- [ ] **False alarm rate** - Rate of false positive alarms over time.
- [ ] **Detection delay** - Distribution or expectation of detection delays.
- [ ] **Matthews Correlation Coefficient (MCC) for anomalies** - Balanced metric for rare-event classification.
- [ ] **Intersection over Union (IoU) for anomaly segments** - Overlap-based scoring for detected vs true segments.
- [ ] **Event-based F1** - F1 defined at the event/segment level.
- [ ] **Segment-based metrics** - Metrics defined on contiguous anomaly segments instead of individual points.

---

*Last updated: February 2026*
