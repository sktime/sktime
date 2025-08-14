# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import os
from abc import ABC, abstractmethod
from typing import TypeVar

from huggingface_hub import hf_hub_download

T = TypeVar("T", bound="PretrainedModel")


def parse_hf_repo_id(path):
    parts = path.split("/")
    return "/".join(parts[0:2])


class PretrainedModel(ABC):
    REGISTRY: dict[str, "PretrainedModel"] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.REGISTRY[cls.register_name()] = cls

    @classmethod
    def from_pretrained(
        cls: type[T], path, device: str = "cuda:0", hf_kwargs=None, ckp_kwargs=None
    ) -> T:
        if hf_kwargs is None:
            hf_kwargs = {}
        if ckp_kwargs is None:
            ckp_kwargs = {}
        if os.path.exists(path):
            print("Loading weights from local directory")
            checkpoint_path = path
        else:
            repo_id = parse_hf_repo_id(path)
            checkpoint_path = hf_hub_download(
                repo_id=repo_id, filename="model.ckpt", **hf_kwargs
            )
        model = cls.load_from_checkpoint(
            checkpoint_path, map_location=device, **ckp_kwargs
        )
        model.after_load_from_checkpoint()
        return model

    @classmethod
    @abstractmethod
    def register_name(cls) -> str:
        pass

    def after_load_from_checkpoint(self):
        pass


def load_model(
    path: str, device: str = "cuda:0", hf_kwargs=None, ckp_kwargs=None
) -> PretrainedModel:
    """Loads a TiRex model. This function attempts to load the specified model.

    Args:
        path (str): Hugging Face path to the model (e.g. NX-AI/TiRex)
        device (str, optional): The device on which to load the model (e.g., "cuda:0", "cpu").
                                If you want to use "cpu" you need to deactivate the sLSTM CUDA kernels (check repository FAQ!).
        hf_kwargs (dict, optional): Keyword arguments to pass to the Hugging Face Hub download method.
        ckp_kwargs (dict, optional): Keyword arguments to pass when loading the checkpoint.

    Returns
    -------
        PretrainedModel: The loaded model.

    Examples
    --------
        model: ForecastModel = load_model("NX-AI/TiRex")
    """
    try:
        _, model_id = parse_hf_repo_id(path).split("/")
    except:
        raise ValueError(f"Invalid model path {path}")
    model_cls = PretrainedModel.REGISTRY.get(model_id, None)
    if model_cls is None:
        raise ValueError(f"Invalid model id {model_id}")
    return model_cls.from_pretrained(
        path, device=device, hf_kwargs=hf_kwargs, ckp_kwargs=ckp_kwargs
    )
