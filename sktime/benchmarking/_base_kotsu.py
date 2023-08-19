"""Interface to overide kotsu code for sktime use."""

import re
import warnings
from typing import Callable, Optional, Union

from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("kotsu", severity="error")
from kotsu.registration import _Registry, _Spec  # noqa: E402


def _check_entity_id_format(entity_id_format: str, id: str) -> None:
    """Check if given input id followed regex specified in entity_id_format."""
    # if not isinstance(entity_id_format, (str, None)):
    #     raise TypeError(
    #         f"entity_id_format must be a str but receive {type(entity_id_format)}"
    #     )

    if entity_id_format is not None:
        entity_id_re = re.compile(entity_id_format)
        match = entity_id_re.search(id)
        if not match:
            raise ValueError(
                f"Attempted to register malformed entity ID: [id={id}]. "
                f"(Currently all IDs must be of the form {entity_id_re.pattern}.)"
            )


class _SktimeSpec(_Spec):
    """A specification for a particular instance of an entity.

    Used to register entity and parameters full specification for evaluations.

    Parameters
    ----------
    id: str
        A unique entity ID. Required format; [username/](entity-name)-v(version)
        [username/] is optional.
    entry_point: Callable or str
        The python entrypoint of the entity class. Should be one of:
        - the string path to the python object (e.g.module.name:factory_func, or
            module.name:Class)
        - the python object (class or factory) itself
    deprecated: Bool, optional (default=False)
        Flag to denote whether this entity should be skipped in validation runs and
        considered deprecated and replaced by a more recent/better validation/model
    nondeterministic: Bool, optional (default=False)
        Whether this entity is non-deterministic even after seeding
    entity_id_fomat: str, optional (default=None)
        Specifying regex to make sure ID follow certain desired format.
    kwargs: Dict, optional (default=None)
        The kwargs to pass to the entity entry point when instantiating the entity

    Notes
    -----
    Taken and adapted from:
    https://github.com/datavaluepeople/kotsu/blob/main/kotsu/registration.py
    """

    def __init__(
        self,
        id: str,
        entry_point: Union[Callable, str],
        deprecated: bool = False,
        nondeterministic: bool = False,
        entity_id_fomat: str = None,
        kwargs: Optional[dict] = None,
    ):
        _check_entity_id_format(entity_id_fomat, id)
        self.id = id
        self.entry_point = entry_point
        self.deprecated = deprecated
        self.nondeterministic = nondeterministic
        self._kwargs = {} if kwargs is None else kwargs


class SktimeModelRegistry(_Registry):
    """Register an entity by ID.

    IDs should remain stable over time and should be guaranteed to resolve to
    the same entity dynamics (or be desupported).
    """

    def __init__(self, entity_id_fomat: str):
        self.entity_id_fomat = entity_id_fomat
        super().__init__()

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
            A unique entity ID. Required format; [username/](entity-name)-v(version)
            [username/] is optional.
        entry_point: Callable or str
            The python entrypoint of the entity class. Should be one of:
            - the string path to the python object (e.g.module.name:factory_func, or
                module.name:Class)
            - the python object (class or factory) itself
        deprecated: Bool, optional (default=False)
            Flag to denote whether this entity should be skipped in validation runs and
            considered deprecated and replaced by a more recent/better validation/model
        nondeterministic: Bool, optional (default=False)
            Whether this entity is non-deterministic even after seeding
        entity_id_fomat: str, optional (default=None)
            Specifying regex to make sure ID follow certain desired format.
        kwargs: Dict, optional (default=None)
            The kwargs to pass to the entity entry point when instantiating the entity
        """
        if id in self.entity_specs:
            warnings.warn(
                message=f"Entity with ID [id={id}] already registered, but the ID is "
                "now being used to register another entity, OVERWRITING previous "
                "registered entity.",
                category=UserWarning,
                stacklevel=2,
            )
        self.entity_specs[id] = _SktimeSpec(
            id,
            entry_point,
            deprecated=deprecated,
            nondeterministic=nondeterministic,
            entity_id_fomat=self.entity_id_fomat,
            kwargs=kwargs,
        )
