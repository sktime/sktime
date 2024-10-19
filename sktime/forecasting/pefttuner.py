"""Tuner module for global forecasters."""

from peft import get_peft_model
from transformers import Trainer


class BaseTuner:
    """Base class for the tuner methods."""

    pass


class PeftTuner(BaseTuner):
    """Peft Tuner implementation."""

    def __init__(self, peft_config):
        self.peft_config = peft_config

    def train(
        self,
        model,
        train_dataset,
        eval_dataset,
        training_args,
        compute_metrics=None,
        callbacks=None,
    ):
        """Train method for the PeftTuner."""
        peft_model = get_peft_model(model, self.peft_config)

        trainer = Trainer(
            peft_model,
            training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

        trainer.train()
        return peft_model


# keep in mind to investigate: help the user figure out how target_modules
# fix html representation for peft
# global forecaster can be recursively called on the previous one
# drawback - investigations of how to integrate peft library methods to this design
