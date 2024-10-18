# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base class for data types."""

__author__ = ["fkiraly"]

from sktime.base import BaseObject
from sktime.datatypes._base._common import _ret
from sktime.utils.deep_equals import deep_equals


class BaseDatatype(BaseObject):
    """Base class for data types.

    This class is the base class for all data types in sktime.
    """

    _tags = {
        "object_type": "datatype",
        "scitype": None,
        "name": None,  # any string
        "name_python": None,  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": None,
    }

    def __init__(self):
        super().__init__()

    # call defaults to check
    def __call__(self, obj, return_metadata=False, var_name="obj"):
        """Check if obj is of this data type.

        Parameters
        ----------
        obj : any
            Object to check.
        return_metadata : bool, optional (default=False)
            Whether to return metadata.
        var_name : str, optional (default="obj")
            Name of the variable to check, for use in error messages.

        Returns
        -------
        valid : bool
            Whether obj is of this data type.
        msg : str, only returned if return_metadata is True.
            Error message if obj is not of this data type.
        metadata : instance of self only returned if return_metadata is True.
            Metadata dictionary.
        """
        return self._check(obj=obj, return_metadata=return_metadata, var_name=var_name)

    def check(self, obj, return_metadata=False, var_name="obj"):
        """Check if obj is of this data type.

        If self has parameters set, the check will in addition
        check whether metadata of obj is equal to self's parameters.
        In this case, ``return_metadata`` will always include the
        metadata fields required to check the parameters.

        Parameters
        ----------
        obj : any
            Object to check.
        return_metadata : bool, optional (default=False)
            Whether to return metadata.
        var_name : str, optional (default="obj")
            Name of the variable to check, for use in error messages.

        Returns
        -------
        valid : bool
            Whether obj is of this data type.
        msg : str, only returned if return_metadata is True.
            Error message if obj is not of this data type.
        metadata : instance of self only returned if return_metadata is True.
            Metadata dictionary.
        """
        self_params = self.get_params()

        need_check = [k for k in self_params if self_params[k] is not None]
        self_dict = {k: self_params[k] for k in need_check}

        return_metadata_orig = return_metadata

        # update return_metadata to retrieve any self_params
        # return_metadata_bool has updated condition
        if not len(need_check) == 0:
            if isinstance(return_metadata, bool):
                if not return_metadata:
                    return_metadata = need_check
                    return_metadata_bool = True
            else:
                return_metadata = set(return_metadata).union(need_check)
                return_metadata = list(return_metadata)
                return_metadata_bool = True
        elif isinstance(return_metadata, bool):
            return_metadata_bool = return_metadata
        else:
            return_metadata_bool = True

        # call inner _check
        check_res = self._check(
            obj=obj, return_metadata=return_metadata, var_name=var_name
        )

        if return_metadata_bool:
            valid = check_res[0]
            msg = check_res[1]
            metadata = check_res[2]
        else:
            valid = check_res
            msg = ""

        if not valid:
            return _ret(False, msg, None, return_metadata_orig)

        # now we know the check is valid, but we need to compare fields
        metadata_sub = {k: metadata[k] for k in self_dict}
        eqs, msg = deep_equals(self_dict, metadata_sub, return_msg=True)
        if not eqs:
            msg = f"metadata of type unequal, {msg}"
            return _ret(False, msg, None, return_metadata_orig)

        self_type = type(self)(**metadata)
        return _ret(True, "", self_type, return_metadata_orig)

    def _check(self, obj, return_metadata=False, var_name="obj"):
        """Check if obj is of this data type.

        Parameters
        ----------
        obj : any
            Object to check.
        return_metadata : bool, optional (default=False)
            Whether to return metadata.
        var_name : str, optional (default="obj")
            Name of the variable to check, for use in error messages.

        Returns
        -------
        valid : bool
            Whether obj is of this data type.
        msg : str, only returned if return_metadata is True.
            Error message if obj is not of this data type.
        metadata : dict, only returned if return_metadata is True.
            Metadata dictionary.
        """
        raise NotImplementedError

    def __getitem__(self, key):
        """Get attribute by key.

        Parameters
        ----------
        key : str
            Attribute name.

        Returns
        -------
        value : any
            Attribute value.
        """
        return getattr(self, key)

    def get(self, key, default=None):
        """Get attribute by key.

        Parameters
        ----------
        key : str
            Attribute name.
        default : any, optional (default=None)
            Default value if attribute does not exist.

        Returns
        -------
        value : any
            Attribute value.
        """
        return getattr(self, key, default)

    def _get_key(self):
        """Get unique dictionary key corresponding to self.

        Private function, used in collecting a dictionary of checks.
        """
        mtype = self.get_class_tag("name")
        scitype = self.get_class_tag("scitype")
        return (mtype, scitype)


class BaseConverter(BaseObject):
    """Base class for data type converters.

    This class is the base class for all data type converters in sktime.
    """

    _tags = {
        "object_type": "converter",
        "mtype_from": None,  # type to convert from - BaseDatatype class or str
        "mtype_to": None,  # type to convert to - BaseDatatype class or str
        "multiple_conversions": False,  # whether converter encodes multiple conversions
        "python_version": None,
        "python_dependencies": None,
    }

    def __init__(self, mtype_from=None, mtype_to=None):
        self.mtype_from = mtype_from
        self.mtype_to = mtype_to
        super().__init__()

        if mtype_from is not None:
            self.set_tags(**{"mtype_from": mtype_from})
        if mtype_to is not None:
            self.set_tags(**{"mtype_to": mtype_to})

        mtype_from = self.get_tag("mtype_from")
        mtype_to = self.get_tag("mtype_to")

        if mtype_from is None:
            raise ValueError(
                f"Error in instantiating {self.__class__.__name__}: "
                "mtype_from and mtype_to must be set if the class has no defaults. "
                "For valid pairs of defaults, use get_conversions."
            )
        if mtype_to is None:
            raise ValueError(
                f"Error in instantiating {self.__class__.__name__}: "
                "mtype_to must be set in constructor, as the class has no defaults. "
                "For valid pairs of defaults, use get_conversions."
            )
        if (mtype_from, mtype_to) not in self.__class__.get_conversions():
            raise ValueError(
                f"Error in instantiating {self.__class__.__name__}: "
                "mtype_from and mtype_to must be a valid pair of defaults. "
                "For valid pairs of defaults, use get_conversions."
            )

    # call defaults to convert
    def __call__(self, obj, store=None):
        """Convert obj to another machine type.

        Parameters
        ----------
        obj : any
            Object to convert.
        store : dict, optional (default=None)
            Reference of storage for lossy conversions.

        Returns
        -------
        converted_obj : any
            Object obj converted to another machine type.
        """
        return self.convert(obj=obj, store=store)

    def convert(self, obj, store=None):
        """Convert obj to another machine type.

        Parameters
        ----------
        obj : any
            Object to convert.
        store : dict, optional (default=None)
            Reference of storage for lossy conversions.
        """
        return self._convert(obj, store)

    def _convert(self, obj, store=None):
        """Convert obj to another machine type.

        Parameters
        ----------
        obj : any
            Object to convert.
        store : dict, optional (default=None)
            Reference of storage for lossy conversions.
        """
        raise NotImplementedError

    @classmethod
    def get_conversions(cls):
        """Get all conversions.

        Returns
        -------
        list of tuples (BaseDatatype subclass, BaseDatatype subclass)
            List of all conversions in this class.
        """
        cls_from = cls.get_class_tag("mtype_from")
        cls_to = cls.get_class_tag("mtype_to")

        if cls_from is not None and cls_to is not None:
            return [(cls_from, cls_to)]
        # if multiple conversions are encoded, this should be overridden
        raise NotImplementedError

    def _get_cls_from_to(self):
        """Get classes from and to.

        Returns
        -------
        cls_from : BaseDatatype subclass
            Class to convert from.
        cls_to : BaseDatatype subclass
            Class to convert to.
        """
        cls_from = self.get_tag("mtype_from")
        cls_to = self.get_tag("mtype_to")

        cls_from = _coerce_str_to_cls(cls_from)
        cls_to = _coerce_str_to_cls(cls_to)

        return cls_from, cls_to

    def _get_key(self):
        """Get unique dictionary key corresponding to self.

        Private function, used in collecting a dictionary of checks.
        """
        cls_from, cls_to = self._get_cls_from_to()

        mtype_from = cls_from.get_class_tag("name")
        mtype_to = cls_to.get_class_tag("name")
        scitype = cls_to.get_class_tag("scitype")
        return (mtype_from, mtype_to, scitype)


class BaseExample(BaseObject):
    """Base class for Example fixtures used in tests and get_examples."""

    _tags = {
        "object_type": "datatype_example",
        "scitype": None,
        "mtype": None,
        "python_version": None,
        "python_dependencies": None,
        "index": None,  # integer index of the example to match with other mtypes
        "lossy": False,  # whether the example is lossy
    }

    def __init__(self):
        super().__init__()

    def _get_key(self):
        """Get unique dictionary key corresponding to self.

        Private function, used in collecting a dictionary of examples.
        """
        mtype = self.get_class_tag("mtype")
        scitype = self.get_class_tag("scitype")
        index = self.get_class_tag("index")
        return (mtype, scitype, index)

    def build(self):
        """Build example.

        Returns
        -------
        obj : any
            Example object.
        """
        raise NotImplementedError


def _coerce_str_to_cls(cls_or_str):
    """Get class from string.

    Parameters
    ----------
    cls_or_str : str or class
        Class or string. If string, assumed to be a unique mtype string from
        one of the BaseDatatype subclasses.

    Returns
    -------
    cls : cls_or_str, if was class; otherwise, class corresponding to string.
    """
    if not isinstance(cls_or_str, str):
        return cls_or_str

    # otherwise, we use the string to get the class from the check dict
    # perhaps it is nicer to transfer this to a registry later.
    from sktime.datatypes._check import get_check_dict

    cd = get_check_dict(soft_deps="all")
    cls = [cd[k].__class__ for k in cd if k[0] == cls_or_str]
    if len(cls) > 1:
        raise ValueError(f"Error in converting string to class: {cls_or_str}")
    elif len(cls) < 1:
        return None
    return cls[0]
