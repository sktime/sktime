# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base class for data types."""

__author__ = ["fkiraly"]

from sktime.base import BaseObject
from sktime.datatypes._common import _ret
from sktime.utils.deep_equals import deep_equals


class BaseDatatype(BaseObject):
    """Base class for data types.

    This class is the base class for all data types in sktime.
    """

    _tags = {
        "obj_type": "mtype",
        "scitype": None,
        "name": None,  # any string
        "name_python": None,  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": None,
    }

    def __init__(self):
        super().__init__()

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


class BaseConverter(BaseObject):
    """Base class for data type converters.

    This class is the base class for all data type converters in sktime.
    """

    _tags = {
        "scitype": None,
        "mtype_from": None,  # equal to name field
        "mtype_to": None,  # equal to name field
        "python_version": None,
        "python_dependencies": None,
    }

    def __init__(self):
        super().__init__()

    def convert(self, obj, store=None):
        """Convert obj to another machine type.

        Parameters
        ----------
        obj : any
            Object to convert.
        store : dict, optional (default=None)
            Reference of storage for lossy conversions.
        """
        raise NotImplementedError
