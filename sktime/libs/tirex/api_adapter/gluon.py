# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import pandas as pd
import torch
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.forecast import QuantileForecast

from .standard_adapter import _batch_pad_iterable

DEF_TARGET_COLUMN = FieldName.TARGET  # target
DEF_META_COLUMNS = (FieldName.START, FieldName.ITEM_ID)


def _get_gluon_ts_map(**gluon_kwargs):
    target_col = gluon_kwargs.get("target_column", DEF_TARGET_COLUMN)
    meta_columns = gluon_kwargs.get("meta_columns", DEF_META_COLUMNS)

    def extract_gluon(series):
        ctx = torch.Tensor(series[target_col])
        meta = {k: series[k] for k in meta_columns if k in series}
        meta["length"] = len(ctx)
        return ctx, meta

    return extract_gluon


def get_gluon_batches(gluonDataset: Dataset, batch_size: int, **gluon_kwargs):
    return _batch_pad_iterable(
        map(_get_gluon_ts_map(**gluon_kwargs), gluonDataset), batch_size
    )


def format_gluonts_output(
    quantile_forecasts: torch.Tensor, mean_forecasts, meta: list[dict], quantile_levels
):
    forecasts = []
    for i in range(quantile_forecasts.shape[0]):
        start_date = meta[i].get(
            FieldName.START, pd.Period("01-01-2000", freq=meta[i].get("freq", "h"))
        )
        start_date += meta[i].get("length", 0)
        forecasts.append(
            QuantileForecast(
                forecast_arrays=torch.cat(
                    (quantile_forecasts[i], mean_forecasts[i].unsqueeze(1)), dim=1
                )
                .T.cpu()
                .numpy(),
                start_date=start_date,
                item_id=meta[i].get(FieldName.ITEM_ID, None),
                forecast_keys=list(map(str, quantile_levels)) + ["mean"],
            )
        )
    return forecasts
