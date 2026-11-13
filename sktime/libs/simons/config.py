# Copyright 2026 Abel AI Lab.
# Source: https://github.com/Abel-ai-lab/Simons
# Apache-2.0; vendored from Abel-ai-lab/Simons at commit 8e338e915d26c8960713889ef12ab1cf5c08d591.
# Model weights are not bundled; see the Simons model license.
# ruff: noqa

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class SimonsConfig:
    """Architecture and forecasting settings of a Simons checkpoint (its config.json)."""

    # Transformer
    context_length: int = 32768  # raw history steps the model reads
    prediction_length: int = 512  # steps forecast in one pass; longer horizons roll out
    patch_length: int = 32
    width: int = 1024
    layers: int = (
        12  # causal time layers; one variate (cross-series) layer follows them
    )
    heads: int = 16
    # Long context: the last `near_context` steps are patched at full resolution, older steps
    # are mean-pooled by `far_pool` into coarser tokens.
    near_context: int = 4096
    far_pool: int = 8
    far_min_history: int = (
        7680  # far tokens only for series observed further back than this
    )
    stats_window: int = 8192  # causal mean/std over a sliding window of this many steps
    # Period-locked rotary embedding
    period_slots: int = 4
    period_alpha: float = 0.05
    period_local: float = 5.0
    detector_mid: int = (
        7680  # series observed only within this many steps use a detector over them
    )
    # Test-time rules (forecast only; they change no weights)
    views: bool = (
        True  # median of native, sign-flipped and square-root views; pooled tail
    )
    # Quantile paths rolled out past prediction_length: fixed to the 0.2, 0.5 and 0.8 paths.
    tail_paths: tuple[int, ...] = (1, 4, 7)
    sign_clip: bool = True  # clip forecasts to the sign of a one-signed history
    integer_snap: bool = True  # round forecasts of non-negative integer histories
    # Covariate modules (simons.covariates); unused when no covariates are supplied.
    covariates: bool = (
        False  # covariate rows: covariates join the target's variate group
    )
    binary_rows: bool = (
        False  # two-level known-future covariates enter unscaled in [0, 1]
    )
    fit_row: bool = (
        True  # single-target rows: add the in-context ridge fit as one more row
    )
    past_rows: bool = False  # always False: past-only covariates are ignored
    read_after: int = (
        -1
    )  # >= 0: target rows read the covariate rows after this time block
    read_binary: bool = (
        True  # always True: the read also keys on two-level (indicator) rows
    )
    xreg: bool = False  # forecast y - x'beta (in-context ridge) and add x'beta back
    xreg_gate: float = (
        0.05  # keep a row's ridge fit only if its held-out R^2 exceeds this
    )
    xreg_square: bool = (
        False  # always False: XReg regresses on the covariates themselves
    )
    xreg_guard: bool = False  # per row, XReg only if it wins a one-origin backtest
    xreg_guard_margin: float = 0.9  # drop XReg iff loss(no XReg) < margin * loss(XReg)
    xreg_guard_min: int = (
        0  # rows with fewer observed steps than max(this, H + 64) keep XReg
    )
    rows_only_history: int = (
        0  # > 0: rows observed for more steps than this: no XReg, no guard
    )

    def __post_init__(self):
        object.__setattr__(self, "tail_paths", tuple(int(p) for p in self.tail_paths))
        if self.read_after >= self.layers - 1:
            raise ValueError("read_after must name a time block before the last")
        if self.xreg_guard and not self.xreg:
            raise ValueError("xreg_guard needs xreg")
        if self.past_rows or not self.read_binary or self.xreg_square:
            raise ValueError(
                "only past_rows: false, read_binary: true and xreg_square: false "
                "are supported"
            )
        if self.rows_only_history < 0 or self.xreg_guard_min < 0:
            raise ValueError("rows_only_history and xreg_guard_min must be nonnegative")
        if self.context_length % self.patch_length or self.width % self.heads:
            raise ValueError(
                "context_length must divide into patches and width into heads"
            )
        if (self.width // self.heads) % 2:
            raise ValueError("head size must be even")
        far = self.context_length - self.near_context
        if far <= 0 or far % (self.patch_length * self.far_pool):
            raise ValueError(
                "context_length - near_context must be a positive multiple of "
                "patch_length * far_pool"
            )
        if 2 * self.period_slots > self.width // self.heads:
            raise ValueError("too many period slots for the head size")
        if self.tail_paths != (1, 4, 7):
            raise ValueError("tail_paths must be (1, 4, 7)")

    @classmethod
    def from_json(cls, path) -> "SimonsConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tail_paths"] = list(self.tail_paths)
        return {"model_type": "simons", **data}
