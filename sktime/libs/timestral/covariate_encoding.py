"""Target encoding of categorical covariates.

Timestral was trained on numeric, target-scaled covariates, so a categorical covariate
is turned into numbers before it reaches the model. The default is a
per-series smoothed mean-target encoding: in series `i`, category `c` becomes

    (smooth * mean_i + sum_{i,c}) / (smooth + count_{i,c})

over the series' context, with `smooth = 1`. Missing values (NaN) form their own
category. A category that never occurs in the context (for example a new value in the
horizon of a known covariate) resolves to the series mean.
"""

import numpy as np


def target_encode(
    target_context: np.ndarray,
    past_codes: np.ndarray,
    future_codes: np.ndarray | None = None,
    smooth: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Per-series smoothed mean-target encoding.

    Args:
        target_context: `(T,)` target values. Non-finite entries are left out of the means.
        past_codes: `(T,)` category codes aligned with `target_context`; NaN marks a
            missing value, which is encoded as a category of its own.
        future_codes: Optional `(H,)` codes over the horizon (known covariates).
        smooth: Weight of the series mean in each category mean.

    Returns:
        `(encoded_past, encoded_future)` float32 arrays; `encoded_future` is None when
        `future_codes` is None.
    """
    target = np.asarray(target_context, dtype=np.float64)
    past_codes = np.asarray(past_codes, dtype=np.float64)
    assert target.shape == past_codes.shape, f"target {target.shape} and codes {past_codes.shape} must align"
    observed = np.isfinite(target)
    target = np.where(observed, target, 0.0)
    item_count = float(observed.sum())
    item_mean = float(target.sum() / item_count) if item_count > 0 else 0.0

    # Slot per distinct finite code, plus a final slot for NaN.
    categories = np.unique(past_codes[np.isfinite(past_codes)])
    nan_slot = len(categories)
    past_slots = _slots(past_codes, categories, nan_slot)
    sums = np.bincount(past_slots, weights=target * observed, minlength=nan_slot + 1)
    counts = np.bincount(past_slots, weights=observed.astype(np.float64), minlength=nan_slot + 1)
    lookup = np.append((smooth * item_mean + sums) / (smooth + counts), item_mean)  # last entry: unseen

    encoded_past = lookup[past_slots].astype(np.float32)
    if future_codes is None:
        return encoded_past, None
    future_slots = _slots(np.asarray(future_codes, dtype=np.float64), categories, nan_slot, unseen=nan_slot + 1)
    return encoded_past, lookup[future_slots].astype(np.float32)


def _slots(codes: np.ndarray, categories: np.ndarray, nan_slot: int, unseen: int | None = None) -> np.ndarray:
    """Slot of each code in `categories`; NaN goes to `nan_slot`, unknown codes to `unseen`."""
    idx = np.searchsorted(categories, np.nan_to_num(codes, nan=0.0))
    idx = np.minimum(idx, max(len(categories) - 1, 0))
    known = np.isfinite(codes) & (len(categories) > 0)
    if len(categories) > 0:
        known &= categories[idx] == codes
    assert unseen is not None or (known | np.isnan(codes)).all(), "every past code must be in categories"
    return np.where(np.isnan(codes), nan_slot, np.where(known, idx, unseen if unseen is not None else 0)).astype(np.intp)
