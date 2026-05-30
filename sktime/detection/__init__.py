"""Time series anomaly, changepoint detection, segmentation."""
try:
    from .evaluation import evaluate_detection
except Exception:
    # Avoid breaking docs build due to optional dependencies
    evaluate_detection = None
