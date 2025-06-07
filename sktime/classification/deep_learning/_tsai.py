# sktime/classification/deep_learning/_tsai.py
from importlib import import_module

from sktime.base.adapters._tsai import _TsaiAdapter


def _safe_import_tsai_model(path: str, name: str):
    """Import tsai model class only if installed; raises ImportError otherwise."""
    module = import_module(path)
    return getattr(module, name)


class InceptionTimeClassifierTsai(_TsaiAdapter):
    """InceptionTime (tsai) as an sktime classifier (algorithm-first naming)."""

    def __init__(self, **kwargs):
        # Dynamically import the tsai InceptionTime class at runtimeAl
        InceptionTime = _safe_import_tsai_model(
            "tsai.models.InceptionTime", "InceptionTime"
        )
        super().__init__(InceptionTime, **kwargs)


class TSTClassifierTsai(_TsaiAdapter):
    """Temporal Self-Attention (TST) (tsai) wrapped as an sktime classifier."""

    def __init__(self, **kwargs):
        # Dynamically import the tsai TST class at runtime
        TST = _safe_import_tsai_model("tsai.models.TST", "TST")
        super().__init__(TST, **kwargs)
