MODEL_REGISTRY = {}

def register_model(name, cls):
    MODEL_REGISTRY[name] = cls

def get_model(name, **kwargs):
    return MODEL_REGISTRY[name](**kwargs)


from .chronos import ChronosForecaster
from .registry import register_model
from .tabpfn_ts.py import TabFNTimeSeriesForecaster

register_model("chronos", ChronosForecaster)
register_model("tabpfn_ts", TabPFNTimeSeriesForecaster)