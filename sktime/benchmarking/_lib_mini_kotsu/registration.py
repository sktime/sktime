"""Registration of entities (models, or validations).

Enables registering specifications of entities and parameters combinations,
under a unique ID, which can be passed to kotsu's run interface.

Based on: https://github.com/openai/gym/blob/master/gym/envs/registration.py
"""

import importlib
import logging
import re
import warnings
from collections.abc import Callable
from typing import Generic, Optional, TypeVar, Union

logger = logging.getLogger(__name__)


Entity = TypeVar("Entity")

# A unique ID for an entity; a name followed by a version number.
# Entity-name is group 1, version is group 2.
# [username/](entity-name)-v(major).(minor)
# See tests for examples of well formed IDs.
entity_id_re = re.compile(r"^(?:[\w:-]+\/)?([\w:.\-{}=\[\]]+)-v([\d.]+)$")


def _load(name: str):
    """Load a python object from string.

    Parameters
    ----------
    name: str
        of the form `path.to.module:object`
    """
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class _Spec(Generic[Entity]):
    """A specification for a particular instance of an entity.

    Used to register entity and parameters full specification for evaluations.

    Parameters
    ----------
    id: str
        A unique entity ID
        Required format; [username/](entity-name)-v(version)
        [username/] is optional.
    entry_point: Callable or str
        The python entrypoint of the entity class. Should be one of:
        - the string path to the python object (e.g.module.name:factory_func, or
            module.name:Class)
        - the python object (class or factory) itself
    deprecated: bool, default=False
        Flag to denote whether this entity should be skipped in validation runs and
        considered deprecated and replaced by a more recent/better validation/model
    nondeterministic: bool, default=False
        Whether this entity is non-deterministic even after seeding
    kwargs: dict, default={}
        The kwargs to pass to the entity entry point when instantiating the entity
    """

    def __init__(
        self,
        id: str,
        entry_point: Union[Callable, str],
        deprecated: bool = False,
        nondeterministic: bool = False,
        kwargs: Optional[dict] = None,
    ):
        self.id = id
        self.entry_point = entry_point
        self.deprecated = deprecated
        self.nondeterministic = nondeterministic
        self._kwargs = {} if kwargs is None else kwargs

        match = entity_id_re.search(id)
        if not match:
            raise ValueError(
                f"Attempted to register malformed entity ID: [id={id}]. "
                f"(Currently all IDs must be of the form {entity_id_re.pattern}.)"
            )

    def make(self, **kwargs) -> Entity:
        """Instantiate an instance of the entity."""
        if self.deprecated:
            raise RuntimeError(
                f"Attempting to make deprecated entity {self.id}. "
                "(HINT: is there a newer registered version of this entity?)"
            )
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            factory = self.entry_point
        else:
            factory = _load(self.entry_point)
        entity = factory(**_kwargs)

        return entity

    def __repr__(self):
        return f"Spec({self.id})"


class _Registry(Generic[Entity]):
    """Register an entity by ID.

    IDs should remain stable over time and should be guaranteed to resolve
    to the same entity dynamics (or be desupported).
    """

    def __init__(self):
        self.entity_specs = {}

    def make(self, id: str, **kwargs) -> Entity:
        """Instantiate an instance of an entity of the given ID."""
        logger.info(f"Making new entity: {id} ({kwargs})")
        try:
            return self.entity_specs[id].make(**kwargs)
        except KeyError:
            raise KeyError(f"No registered entity with ID {id}")

    def all(self):
        """Return all the entities in the registry."""
        return self.entity_specs.values()

    def register(
        self,
        id: str,
        entry_point: Union[Callable, str],
        deprecated: bool = False,
        nondeterministic: bool = False,
        kwargs: Optional[dict] = None,
    ):
        """Register an entity.

        Parameters
        ----------
        id: str
            A unique entity ID
            Required format; [username/](entity-name)-v(version)
            [username/] is optional.
        entry_point: Callable or str
            The python entrypoint of the entity class. Should be one of:
            - the string path to the python object (e.g.module.name:factory_func, or
                module.name:Class)
            - the python object (class or factory) itself
        deprecated: bool, default=False
            Flag to denote whether this entity should be skipped in validation runs and
            considered deprecated and replaced by a more recent/better validation/model
        nondeterministic: bool, default=False
            Whether this entity is non-deterministic even after seeding
        kwargs: dict, default={}
            The kwargs to pass to the entity entry point when instantiating the entity
        """
        if id in self.entity_specs:
            warnings.warn(
                f"Entity with ID [id={id}] already registered, now registering "
                "entity of the same ID, OVERWRITING previous registered entity.",
                UserWarning,
                stacklevel=2,
            )
        self.entity_specs[id] = _Spec(
            id,
            entry_point,
            deprecated=deprecated,
            nondeterministic=nondeterministic,
            kwargs=kwargs,
        )
