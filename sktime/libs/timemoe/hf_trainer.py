"""Hugging Face Trainer extensions for Time-MoE."""

import inspect
import math
from dataclasses import dataclass, field
from functools import partial

from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
Trainer = _safe_import("transformers.Trainer")
TrainingArguments = _safe_import("transformers.TrainingArguments")
LambdaLR = _safe_import("torch.optim.lr_scheduler.LambdaLR")
get_scheduler = _safe_import("transformers.get_scheduler")


class TimeMoeTrainer(Trainer):
    """Trainer for Time-MoE with loss-mask signature support."""

    epsilon = 1e-8

    def __init__(
        self,
        label_column: str = "labels",
        loss_mask_column: str = "loss_mask",
        *positional_args,
        **kwargs,
    ):
        super().__init__(*positional_args, **kwargs)
        self.tokenizer = kwargs.get("tokenizer", None)
        self.label_column = label_column
        self.loss_mask_column = loss_mask_column

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        """Create LR scheduler, with optional cosine min-LR support."""
        optimizer = self.optimizer if optimizer is None else optimizer
        min_lr_ratio = self.args.min_learning_rate / self.args.learning_rate
        if self.lr_scheduler is None:
            if self.args.lr_scheduler_type == "cosine":
                self.lr_scheduler = get_cosine_schedule_with_warmup_min_lr(
                    optimizer=optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    min_lr_ratio=min_lr_ratio,
                )
            else:
                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            params = list(signature.parameters.keys())
            # Labels may be named label or label_ids.
            self._signature_columns = list(
                set(
                    params
                    + self.label_names
                    + [
                        "label",
                        "label_ids",
                        self.label_column,
                        self.loss_mask_column,
                    ]
                )
            )


@dataclass
class TimeMoETrainingArguments(TrainingArguments):
    """TrainingArguments with Time-MoE min learning rate."""

    min_learning_rate: float = field(
        default=0, metadata={"help": "Minimum learning rate for cosine_schedule"}
    )


def _get_cosine_schedule_with_warmup_and_min_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    min_lr_ratio: float,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))

    return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_ratio)


def get_cosine_schedule_with_warmup_min_lr(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0,
    last_epoch: int = -1,
):
    """Cosine schedule with warmup that floors at ``min_lr_ratio``."""
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_and_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
