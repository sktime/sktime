# -*- coding: utf-8 -*-
from sktime.base._registry_enum import BaseRegistryEnum


class MtypeEnum(BaseRegistryEnum):
    """Creates registry enums.

    Parameters
    ----------
    value: str
        String value of the enum
    description: str
        String description of what the enum represents
    instance: Any
        Instance of the enum
    is_lossy: bool
        Boolean that defines if the conversion is lossy
    """

    def __init__(
        self, value: str, description: str, instance: str = None, is_lossy: bool = True
    ):
        super(MtypeEnum, self).__init__(value, description, instance)
        self.is_lossy: bool = is_lossy
