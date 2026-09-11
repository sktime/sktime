"""Concrete catalogue classes for classification experiments."""

from sktime.catalogues.classification.bakeoff import BakeOffCatalogue
from sktime.catalogues.classification.dummy import DummyClassificationCatalogue

__all__ = [
    BakeOffCatalogue,
    DummyClassificationCatalogue,
]
