# -*- coding: utf-8 -*-
"""Testing utility to play back usage scenarios for estimators.

Contains TestScenario class which applies method/args subsequently
"""

__author__ = ["fkiraly"]

__all__ = ["TestScenario"]


from copy import deepcopy
from inspect import isclass

import numpy as np


class TestScenario:
    """Class to run pre-defined method execution scenarios for objects.

    Parameters
    ----------
    args : dict of dict, default = None
        dict of argument dicts to be used in methods
        names for keys need not equal names of methods these are used in
            but scripted method will look at key with same name as default
        must be passed to constructor, set in a child class
            or dynamically created in get_args
    default_method_sequence : list of str, default = None
        default sequence for methods to be called
        optional, if given, default method sequence to use in `run`
        if not provided, at least one of the sequence arguments must be passed in `run`
            or default_arg_sequence must be provided
    default_arg_sequence : list of str, default = None
        default sequence of keys for keyword argument dicts to be used
        names for keys need not equal names of methods
        if not provided, at least one of the sequence arguments must be passed in `run`
            or default_method_sequence must be provided

    Methods
    -------
    run(obj, args=None, default_method_sequence=None)
        Run a call(args) scenario on obj, and retrieve method outputs.
    is_applicable(obj)
        Check whether scenario is applicable to obj.
    get_args(key, obj)
        Dynamically create args for call defined by key and obj.
        Defaults to self.args[key] if not overridden.
    """

    def __init__(
        self, args=None, default_method_sequence=None, default_arg_sequence=None
    ):
        if default_method_sequence is not None:
            self.default_method_sequence = _check_list_of_str(default_method_sequence)
        elif not hasattr(self, "default_method_sequence"):
            self.default_method_sequence = None
        if default_arg_sequence is not None:
            self.default_arg_sequence = _check_list_of_str(default_arg_sequence)
        elif not hasattr(self, "default_arg_sequence"):
            self.default_arg_sequence = None
        if args is not None:
            self.args = _check_dict_of_dict(args)
        else:
            if not hasattr(self, "args"):
                raise RuntimeError(
                    "args must either be given to __init__ or set in a child class"
                )
            _check_dict_of_dict(self.args)

    def get_args(self, key, obj=None, deepcopy_args=True):
        """Return args for key. Can be overridden for dynamic arg generation.

        If overridden, must not have any side effects on self.args
            e.g., avoid assignments args[key] = x without deepcopying self.args first

        Parameters
        ----------
        key : str, argument key to construct/retrieve args for
        obj : obj, optional, default=None. Object to construct args for.
        deepcopy_args : bool, optional, default=True. Whether to deepcopy return.

        Returns
        -------
        args : argument dict to be used for a method, keyed by `key`
            names for keys need not equal names of methods these are used in
                but scripted method will look at key with same name as default
        """
        args = self.args[key]
        if deepcopy_args:
            args = deepcopy(args)
        return args

    def run(
        self,
        obj,
        method_sequence=None,
        arg_sequence=None,
        return_all=False,
        return_args=False,
        deepcopy_return=False,
    ):
        """Run a call(args) scenario on obj, and retrieve method outputs.

        Runs a sequence of commands
            res_1 = obj.method_1(**args_1)
            res_2 = obj.method_2(**args_2)
            etc, where method_i is method_sequence[i],
                and args_i is self.args[arg_sequence[i]]
        and returns results. Args are passed as deepcopy to avoid side effects.

        if method_i is __init__ (a constructor),
        obj is changed to obj.__init__(**args_i) from the next line on

        Parameters
        ----------
        obj : class or object with methods in method_sequence
        method_sequence : list of str, default = arg_sequence if passed
            if arg_sequence is also None, then default = self.default_method_sequence
            sequence of method names to be run
        arg_sequence : list of str, default = method_sequence if passed
            if method_sequence is also None, then default = self.default_arg_sequence
            sequence of keys for keyword argument dicts to be used
            names for keys need not equal names of methods
        return_all : bool, default = False
            whether all or only the last result should be returned
            if False, only the last result is returned
            if True, list of deepcopies of intermediate results is returned
        return_args : bool, default = False
            whether arguments should also be returned
            if False, there is no second return argument
            if True, "args_after_call" return argument is returned
        deepcopy_return : bool, default = False
            whether returns are deepcopied before returned
            if True, returns are deepcopies of return
            if False, returns are references/assignments, not deepcopies
                NOTE: if self is returned (e.g., in fit), and deepcopy_return=False
                    method calls may continue to have side effects on that return

        Returns
        -------
        results : output of the last method call, if return_all = False
            list of deepcopies of all outputs, if return_all = True
        args_after_call : list of args after method call, only if return_args = True
            i-th element is deepcopy of args of i-th method call, after method call
                this is possibly subject to side effects by the method
        """
        # if both None, fill with defaults if exist
        if method_sequence is None and arg_sequence is None:
            method_sequence = getattr(self, "default_method_sequence", None)
            arg_sequence = getattr(self, "default_arg_sequence", None)

        # if both are still None, raise an error
        if method_sequence is None and arg_sequence is None:
            raise ValueError(
                "at least one of method_sequence, arg_sequence must be not None "
                "if no defaults are set in the class"
            )

        # if only one is None, fill one with the other
        if method_sequence is None:
            method_sequence = _check_list_of_str(arg_sequence)
        else:
            method_sequence = _check_list_of_str(method_sequence)
        if arg_sequence is None:
            arg_sequence = _check_list_of_str(method_sequence)
        else:
            arg_sequence = _check_list_of_str(arg_sequence)

        # check that length of sequences is the same
        num_calls = len(arg_sequence)
        if not num_calls == len(method_sequence):
            raise ValueError("arg_sequence and method_sequence must have same length")

        # execute the commands in sequence, report result(s)
        results = []
        args_after_call = []
        for i in range(num_calls):
            methodname = method_sequence[i]
            args = deepcopy(self.get_args(key=arg_sequence[i], obj=obj))

            if methodname != "__init__":
                res = getattr(obj, methodname)(**args)
            # if constructor is called, run directly and replace obj
            else:
                if isclass(obj):
                    res = obj(**args)
                else:
                    res = type(obj)(**args)
                obj = res

            args_after_call += [args]

            if deepcopy_return:
                res = deepcopy(res)

            if return_all:
                results += [res]
            else:
                results = res

        if return_args:
            return results, args_after_call
        else:
            return results

    def is_applicable(self, obj):
        """Check whether scenario is applicable to obj.

        Abstract method, children should implement. This just returns "true".

        Example for child class: scenario is univariate time series forecasting.
            Then, this returns False on multivariate, True on univariate forecasters.

        Parameters
        ----------
        obj : class or object to check against scenario

        Returns
        -------
        applicable: bool
            True if self is applicable to obj, False if not
            "applicable" is defined as the implementer chooses, as output of this method
                False is typically used as a "skip" flag in unit or integration testing
        """
        return True


def _check_list_of_str(obj, name="obj"):
    """Check whether obj is a list of str.

    Parameters
    ----------
    obj : any object, check whether is list of str
    name : str, default="obj", name of obj to display in error message

    Returns
    -------
    obj, unaltered

    Raises
    ------
    TypeError if obj is not list of str
    """
    if not isinstance(obj, list) or not np.all([isinstance(x, str) for x in obj]):
        raise TypeError(f"{obj} must be a list of str")
    return obj


def _check_dict_of_dict(obj, name="obj"):
    """Check whether obj is a dict of dict, with str keys.

    Parameters
    ----------
    obj : any object, check whether is dict of dict, with str keys
    name : str, default="obj", name of obj to display in error message

    Returns
    -------
    obj, unaltered

    Raises
    ------
    TypeError if obj is not dict of dict, with str keys
    """
    msg = f"{obj} must be a dict of dict, with str keys"
    if not isinstance(obj, dict):
        raise TypeError(msg)
    if not np.all([isinstance(x, dict) for x in obj.values()]):
        raise TypeError(msg)
    if not np.all([isinstance(x, str) for x in obj.keys()]):
        raise TypeError(msg)
    return obj
