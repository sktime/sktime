# -*- coding: utf-8 -*-
__all__ = ["BaseRegistryEnum"]

from typing import Any
from enum import Enum, EnumMeta


class _RegistryMetaEnum(EnumMeta):
    def __contains__(self, item):
        return any(x.value == item for x in self)


class BaseRegistryEnum(Enum, metaclass=_RegistryMetaEnum):
    def __init__(self, value: str, description: str, instance: Any = None):
        self._value_: str = value
        self.description: str = description
        self.instance = instance

    def __iter__(self):
        yield self.value
        if self.instance is not None:
            yield self.instance
        yield self.description
