# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Chaining of data type converters."""

__author__ = ["adity1raut"]

from sktime.datatypes._base._base import BaseConverter

# key under which ConverterChain bookkeeps the stores of its elements,
# in the store that is passed to the chain
CHAIN_STORE_KEY = "__converter_chain__"


def _mtype_name(mtype):
    """Coerce an mtype tag value to the mtype string.

    Parameters
    ----------
    mtype : str, BaseDatatype subclass, or None
        mtype, as encoded in the ``mtype_from`` or ``mtype_to`` tag

    Returns
    -------
    str or None
        mtype string of ``mtype``; None if it cannot be determined
    """
    if mtype is None or isinstance(mtype, str):
        return mtype
    if hasattr(mtype, "get_class_tag"):
        return mtype.get_class_tag("name")
    return None


def _is_lossy(converter):
    """Return whether a converter is lossy.

    Parameters
    ----------
    converter : callable with signature ``(obj, store=None)``
        converter, e.g., ``BaseConverter`` instance or function

    Returns
    -------
    bool
        whether ``converter`` declares itself as lossy, via the ``lossy`` tag.
        False if ``converter`` does not declare lossiness, e.g., if it is a
        function in converter signature.
    """
    if not hasattr(converter, "get_tag"):
        return False
    return bool(converter.get_tag("lossy", False))


class ConverterChain(BaseConverter):
    """Chain of converters, behaving as the composition of its elements.

    ``ConverterChain`` applies ``converters`` in sequence, in the order given,
    and is used to obtain a conversion from mtype A to mtype C, if conversions
    from A to B and from B to C are defined, but not from A to C.

    The chain is lossy if and only if at least one of its elements is lossy.

    Stores of the elements are kept separate, to avoid collisions between
    metadata that different elements write under the same key. The element
    stores are bookkept in the ``store`` passed to the chain, which allows a
    chain to pick up the stores of the reverse chain, if the same ``store`` is
    passed, and thus to invert a lossy chain conversion.

    Parameters
    ----------
    converters : sequence of converters
        converters to apply, in the order given. Elements must be callable
        with signature ``(obj, store=None)``, e.g., ``BaseConverter``
        instances, or functions in converter signature, see
        ``datatypes._convert``.
    mtype_from : str, optional
        mtype converted from, i.e., mtype that the first element converts from.
        If not provided, is inferred from the first element of ``converters``.
    mtype_to : str, optional
        mtype converted to, i.e., mtype that the last element converts to.
        If not provided, is inferred from the last element of ``converters``.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.datatypes._base._chain import ConverterChain
    >>>
    >>> to_numpy = lambda obj, store=None: obj.to_numpy()
    >>> transpose = lambda obj, store=None: obj.T
    >>> chain = ConverterChain(
    ...     [to_numpy, transpose], mtype_from="pd.DataFrame", mtype_to="np.ndarray"
    ... )
    >>> chain(pd.DataFrame({"a": [1, 2, 3]})).shape
    (1, 3)
    """

    _tags = {
        "object_type": "converter",
        "multiple_conversions": False,
    }

    def __init__(self, converters, mtype_from=None, mtype_to=None):
        self.converters = converters

        self._check_converters()

        if mtype_from is None:
            mtype_from = _mtype_name(self._element_tag(0, "mtype_from"))
        if mtype_to is None:
            mtype_to = _mtype_name(self._element_tag(-1, "mtype_to"))

        super().__init__(mtype_from=mtype_from, mtype_to=mtype_to)

        self.set_tags(**{"lossy": any(_is_lossy(x) for x in self.converters)})

    def _element_tag(self, i, tag_name):
        """Return tag ``tag_name`` of the i-th element, None if not available."""
        element = self.converters[i]
        if not hasattr(element, "get_tag"):
            return None
        return element.get_tag(tag_name, None)

    @classmethod
    def get_conversions(cls):
        """Get all conversions.

        Returns
        -------
        list
            empty list - the conversion of a ``ConverterChain`` is determined by
            its elements, the class itself has no conversions as defaults.
        """
        return []

    def _check_converters(self):
        """Check that ``converters`` is a non-empty sequence of callables.

        Raises
        ------
        TypeError if ``converters`` is not a sequence, or has non-callable elements
        ValueError if ``converters`` is empty
        """
        converters = self.converters

        if isinstance(converters, str) or not hasattr(converters, "__len__"):
            raise TypeError(
                f"Error in instantiating {self.__class__.__name__}: converters "
                f"must be a sequence of converters, but found {type(converters)}."
            )
        if len(converters) == 0:
            raise ValueError(
                f"Error in instantiating {self.__class__.__name__}: converters "
                "must contain at least one converter, but is empty."
            )
        for i, converter in enumerate(converters):
            if not callable(converter):
                raise TypeError(
                    f"Error in instantiating {self.__class__.__name__}: elements "
                    "of converters must be callable with signature "
                    f"(obj, store=None), but element {i} is not callable."
                )

    def _check_conversion_defined(self):
        """Check that the conversion of self is fully and validly specified.

        Raises
        ------
        ValueError if ``mtype_from`` or ``mtype_to`` are not set, and cannot be
        inferred from the elements of the chain
        """
        for tag_name in ["mtype_from", "mtype_to"]:
            if self.get_tag(tag_name) is None:
                raise ValueError(
                    f"Error in instantiating {self.__class__.__name__}: "
                    f"{tag_name} must be passed to the constructor, if it cannot "
                    "be inferred from the converters in the chain."
                )

    def _convert(self, obj, store=None):
        """Convert obj, by applying the converters in the chain in sequence.

        Parameters
        ----------
        obj : any
            Object to convert.
        store : dict, optional (default=None)
            Reference of storage for lossy conversions. If passed, the stores
            of the elements are bookkept in ``store``, by side effect.

        Returns
        -------
        converted_obj : any
            Object obj converted to another machine type.
        """
        stores = self._get_element_stores(store)

        for converter, element_store in zip(self.converters, stores):
            obj = converter(obj, store=element_store)

        self._record_element_stores(store, stores)

        return obj

    def _get_element_stores(self, store):
        """Return the stores to pass to the elements of the chain.

        If ``store`` bookkeeps the element stores of the reverse chain, these
        are reused in reverse order, so a lossy chain conversion can be
        inverted. Otherwise, fresh stores are used.
        """
        n_converters = len(self.converters)

        if not isinstance(store, dict):
            return [None] * n_converters

        bookkept = store.get(CHAIN_STORE_KEY, None)
        if self._is_reverse_of(bookkept):
            return list(reversed(bookkept["stores"]))

        return [{} for _ in range(n_converters)]

    def _is_reverse_of(self, bookkept):
        """Return whether ``bookkept`` was written by the reverse of self."""
        if not isinstance(bookkept, dict):
            return False
        stores = bookkept.get("stores", None)
        if not isinstance(stores, list) or len(stores) != len(self.converters):
            return False
        reverse_from = bookkept.get("mtype_from", None) == self.get_tag("mtype_to")
        reverse_to = bookkept.get("mtype_to", None) == self.get_tag("mtype_from")
        return reverse_from and reverse_to

    def _record_element_stores(self, store, stores):
        """Bookkeep the element stores in ``store``, by side effect."""
        if not isinstance(store, dict):
            return
        store[CHAIN_STORE_KEY] = {
            "mtype_from": self.get_tag("mtype_from"),
            "mtype_to": self.get_tag("mtype_to"),
            "stores": stores,
        }
