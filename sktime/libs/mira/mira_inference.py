# ruff: noqa
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
F = _safe_import("torch.nn.functional")


# ---------------------------
# Normalization Helpers
# ---------------------------
def normalize(values, mean, std):
    return (values - mean) / std


def denormalize(values, mean, std):
    return values * std + mean


# ---------------------------
# Autoregressive Prediction (Normalized)
# ---------------------------
def mira_predict_autoregressive_norm(
    model,
    values,  # [1, T]
    times,  # [1, T]
    context_len,
    pred_len,
    mean,
    std,
):
    """
    Normalized autoregressive prediction.
    """
    device = next(model.parameters()).device
    values = values.to(device)
    times = times.to(device)

    # 1. normalize
    values_norm = normalize(values, mean, std)

    B, T = values_norm.shape
    hist_vals = values_norm[:, :context_len]  # [1, 12]
    fut_times = times[:, context_len:]  # [1, 6]

    preds_norm = []
    cur_vals = hist_vals.clone()
    cur_times = times[:, :context_len].clone()

    # 2. Predict step-by-step
    for i in range(pred_len):
        inp_vals = cur_vals.unsqueeze(-1)  # [1, L, 1]
        inp_times = cur_times  # [1, L]

        with torch.no_grad():
            out = model(input_ids=inp_vals, time_values=inp_times, return_dict=True)

        next_norm = out.logits[:, -1, :]  # [1,1]
        preds_norm.append(next_norm.squeeze(0))  # -> [1]

        # append
        cur_vals = torch.cat([cur_vals, next_norm], dim=1)
        cur_times = torch.cat([cur_times, fut_times[:, i : i + 1]], dim=1)

    preds_norm = torch.stack(preds_norm, dim=1)  # [1,6]

    # 3. Denormalize back
    preds = denormalize(preds_norm, mean, std)

    return preds.squeeze(0)  # [6]


# ---------------------------
# Evaluate a single sample
# ---------------------------
def mira_evaluate_one_window_norm(
    model,
    values,  # [1, T]
    times,  # [1, T]
    context_len=12,
    pred_len=6,
    mean=0.0,
    std=1.0,
):
    """
    Returns:
        preds:  [pred_len]
        gt:     [pred_len]
        rmse:   float
        mae:    float
    """

    device = next(model.parameters()).device
    values = values.to(device)
    times = times.to(device)

    # ground truth (original scale)
    gt = values[:, context_len:].squeeze(0)  # [6]

    # prediction (auto-regressive)
    preds = mira_predict_autoregressive_norm(
        model, values, times, context_len, pred_len, mean, std
    )

    rmse = torch.sqrt(F.mse_loss(preds, gt))
    mae = F.l1_loss(preds, gt)

    return preds, gt, rmse.item(), mae.item()
