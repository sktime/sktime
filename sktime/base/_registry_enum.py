# -*- coding: utf-8 -*-
__all__ = ["BaseRegistryEnum"]

from enum import Enum, EnumMeta
from typing import TypeVar


class _RegistryMetaEnum(EnumMeta):
    def __contains__(self, item):
        return any(x.name == item for x in self)

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            for val in self:
                if str(val) == item:
                    return val

    def __str__(self):
        enum_val = list(self)[0]
        return enum_val.instance


T = TypeVar("T")


class BaseRegistryEnum(Enum, metaclass=_RegistryMetaEnum):
    """Creates registry enums.

    Parameters
    ----------
    description: str
        String description of what the enum represents
    type: T
        type of the enum
    """

    def __init__(self, description: str, instance: str, type_of: T = None):
        self.description: str = description
        self.instance: str = instance
        self.type_of: T = type_of

    def __iter__(self):
        """Iterate over enum values.

        Returns
        -------
        Generator
            Yields values of enum in value, instance, description order.
        """
        yield self.name
        if self.instance is not None:
            yield self.instance
        yield self.description

    def __str__(self):
        """Value of enum.

        Returns
        -------
        str
            String version of enum
        """
        return self.name
