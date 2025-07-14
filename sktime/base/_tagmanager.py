"""Mixin class for flag and configuration settings management."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["_FlagManager"]


import inspect
from copy import deepcopy


class _FlagManager:
    """Mixin class for flag and configuration settings management."""

    @classmethod
    def _get_class_flags(cls, flag_attr_name="_flags"):
        """Get class flags from estimator class and all its parent classes.

        Parameters
        ----------
        flag_attr_name : str, optional, default = "_flags"
            name of the flag attribute that is read

        Returns
        -------
        collected_flags : dict
            Dictionary of flag name : flag value pairs. Collected from _flags
            class attribute via nested inheritance. NOT overridden by dynamic
            flags set by set_flags or clone_flags.
        """
        collected_flags = dict()

        # We exclude the last two parent classes: sklearn.base.BaseEstimator and
        # the basic Python object.
        for parent_class in reversed(inspect.getmro(cls)[:-2]):
            if hasattr(parent_class, flag_attr_name):
                # Need the if here because mixins might not have _more_flags
                # but might do redundant work in estimators
                # (i.e. calling more flags on BaseEstimator multiple times)
                more_flags = getattr(parent_class, flag_attr_name)
                collected_flags.update(more_flags)

        return deepcopy(collected_flags)

    @classmethod
    def _get_class_flag(
        cls, flag_name, flag_value_default=None, flag_attr_name="_flags"
    ):
        """Get flag value from estimator class (only class flags).

        Parameters
        ----------
        flag_name : str
            Name of flag value.
        flag_value_default : any type
            Default/fallback value if flag is not found.
        flag_attr_name : str, optional, default = "_flags"
            name of the flag attribute that is read

        Returns
        -------
        flag_value :
            Value of `flag_name` flag in self. If not found, `flag_value_default`.
        """
        collected_flags = cls._get_class_flags(flag_attr_name=flag_attr_name)

        return collected_flags.get(flag_name, flag_value_default)

    def _init_flags(self, flag_attr_name="_flags"):
        """Create dynamic flag dictionary in self.

        Should be called in __init__ of the host class.
        Creates attribute [flag_attr_name]_dynamic containing an empty dict.

        Parameters
        ----------
        flag_attr_name : str, optional, default = "_flags"
            name of the flag attribute that is read

        Returns
        -------
        self : reference to self
        """
        setattr(self, f"{flag_attr_name}_dynamic", dict())
        return self

    def _get_flags(self, flag_attr_name="_flags"):
        """Get flags from estimator class and dynamic flag overrides.

        Parameters
        ----------
        flag_attr_name : str, optional, default = "_flags"
            name of the flag attribute that is read

        Returns
        -------
        collected_flags : dict
            Dictionary of flag name : flag value pairs. Collected from flag_attr_name
            class attribute via nested inheritance and then any overrides
            and new flags from [flag_attr_name]_dynamic object attribute.
        """
        collected_flags = self._get_class_flags(flag_attr_name=flag_attr_name)

        if hasattr(self, f"{flag_attr_name}_dynamic"):
            collected_flags.update(getattr(self, f"{flag_attr_name}_dynamic"))

        return deepcopy(collected_flags)

    def _get_flag(
        self,
        flag_name,
        flag_value_default=None,
        raise_error=True,
        flag_attr_name="_flags",
    ):
        """Get flag value from estimator class and dynamic flag overrides.

        Parameters
        ----------
        flag_name : str
            Name of flag to be retrieved
        flag_value_default : any type, optional; default=None
            Default/fallback value if flag is not found
        raise_error : bool
            whether a ValueError is raised when the flag is not found
        flag_attr_name : str, optional, default = "_flags"
            name of the flag attribute that is read

        Returns
        -------
        flag_value :
            Value of the `flag_name` flag in self. If not found, returns an error if
            raise_error is True, otherwise it returns `flag_value_default`.

        Raises
        ------
        ValueError if raise_error is True i.e. if flag_name is not in self.get_flags(
        ).keys()
        """
        collected_flags = self._get_flags(flag_attr_name=flag_attr_name)

        flag_value = collected_flags.get(flag_name, flag_value_default)

        if raise_error and flag_name not in collected_flags.keys():
            raise ValueError(f"Tag with name {flag_name} could not be found.")

        return flag_value

    def _set_flags(self, flag_attr_name="_flags", **flag_dict):
        """Set dynamic flags to given values.

        Parameters
        ----------
        flag_dict : dict
            Dictionary of flag name : flag value pairs.
        flag_attr_name : str, optional, default = "_flags"
            name of the flag attribute that is read

        Returns
        -------
        Self : Reference to self.

        Notes
        -----
        Changes object state by setting flag values in flag_dict as dynamic flags
        in self.
        """
        flag_update = deepcopy(flag_dict)
        dynamic_flags = f"{flag_attr_name}_dynamic"
        if hasattr(self, dynamic_flags):
            getattr(self, dynamic_flags).update(flag_update)
        else:
            setattr(self, dynamic_flags, flag_update)

        return self

    def _clone_flags(self, estimator, flag_names=None, flag_attr_name="_flags"):
        """Clone/mirror flags from another estimator as dynamic override.

        Parameters
        ----------
        estimator : estimator inheriting from :class:BaseEstimator
        flag_names : str or list of str, default = None
            Names of flags to clone. If None then all flags in estimator are used
            as `flag_names`.
        flag_attr_name : str, optional, default = "_flags"
            name of the flag attribute that is read

        Returns
        -------
        Self :
            Reference to self.

        Notes
        -----
        Changes object state by setting flag values in flag_set from estimator as
        dynamic flags in self.
        """
        flags_est = deepcopy(estimator._get_flags(flag_attr_name=flag_attr_name))

        # if flag_set is not passed, default is all flags in estimator
        if flag_names is None:
            flag_names = flags_est.keys()
        else:
            # if flag_set is passed, intersect keys with flags in estimator
            if not isinstance(flag_names, list):
                flag_names = [flag_names]
            flag_names = [key for key in flag_names if key in flags_est.keys()]

        update_dict = {key: flags_est[key] for key in flag_names}

        self._set_flags(flag_attr_name=flag_attr_name, **update_dict)

        return self
