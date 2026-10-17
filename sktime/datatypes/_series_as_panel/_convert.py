"""Machine type converters for Series to Panel.

Exports conversion functions for conversions between series scitypes:

convert_Series_to_Panel(obj, store=None)
    converts obj of Series mtype to "adjacent" Panel mtype (e.g., numpy to numpy)
convert_Panel_to_Series(obj, store=None)
    converts obj of Panel mtype to "adjacent" Series mtype (e.g., numpy to numpy)
convert_Series_to_Hierarchical(obj, store=None)
convert_Hierarchical_to_series(obj, store=None)
convert_Panel_to_Hierarchical(obj, store=None)
convert_Hierarchical_to_Panel(obj, store=None)
    converts to pd.DataFrame based data container in the target scitype
"""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd
from skbase.base import BaseObject

from sktime.datatypes import convert_to, scitype


class BaseScitypeConverter(BaseObject):
    """Base class for converters between Series, Panel, and Hierarchical scitypes.

    Mirrors the API of ``BaseConverter``, but for conversions that change the
    scitype. Kept separate from ``BaseConverter``, so these conversions are not
    picked up as same-scitype mtype conversions.

    Each subclass implements a single conversion, declared via tags:

    * ``scitype_from``, ``scitype_to``: scitypes converted from and to
    * ``mtype_from``, ``mtype_to``: mtypes converted from and to,
      ``mtype_from=None`` means any mtype of ``scitype_from`` is accepted
    * ``python_type``: python type of inputs the converter applies to
    """

    _tags = {
        "object_type": "converter",
        "scitype_from": None,  # scitype to convert from, str
        "scitype_to": None,  # scitype to convert to, str
        "mtype_from": None,  # mtype to convert from, str, None = any
        "mtype_to": None,  # mtype to convert to, str
        "python_type": object,  # python type of inputs converted
        "multiple_conversions": False,  # whether converter encodes multiple conversions
        "python_version": None,
        "python_dependencies": None,
    }

    def __call__(self, obj, store=None):
        """Convert obj, see ``convert``."""
        return self.convert(obj=obj, store=store)

    def convert(self, obj, store=None):
        """Convert obj from ``mtype_from`` to ``mtype_to``.

        Parameters
        ----------
        obj : any
            Object to convert, of scitype ``scitype_from``.
        store : dict, optional (default=None)
            Reference of storage for lossy conversions.

        Returns
        -------
        converted_obj : obj converted to ``mtype_to``, of scitype ``scitype_to``
        """
        return self._convert(obj, store)

    def _convert(self, obj, store=None):
        raise NotImplementedError

    @classmethod
    def get_conversions(cls):
        """Get the conversion implemented by this class.

        Returns
        -------
        list of tuples (str, str)
            List with the single conversion ``(mtype_from, mtype_to)``.
        """
        return [(cls.get_class_tag("mtype_from"), cls.get_class_tag("mtype_to"))]

    @classmethod
    def applies_to(cls, obj):
        """Return whether obj is of the python type this converter applies to."""
        return isinstance(obj, cls.get_class_tag("python_type"))


class PdSeriesToDfList(BaseScitypeConverter):
    """Conversion: pd.Series -> df-list, Series to Panel scitype."""

    _tags = {
        "scitype_from": "Series",
        "scitype_to": "Panel",
        "mtype_from": "pd.Series",
        "mtype_to": "df-list",
        "python_type": pd.Series,
    }

    def _convert(self, obj, store=None):
        return [pd.DataFrame(obj)]


class PdDataFrameToDfList(BaseScitypeConverter):
    """Conversion: pd.DataFrame -> df-list, Series to Panel scitype."""

    _tags = {
        "scitype_from": "Series",
        "scitype_to": "Panel",
        "mtype_from": "pd.DataFrame",
        "mtype_to": "df-list",
        "python_type": pd.DataFrame,
    }

    def _convert(self, obj, store=None):
        return [obj]


class NumpyToNumpy3D(BaseScitypeConverter):
    """Conversion: np.ndarray -> numpy3D, Series to Panel scitype."""

    _tags = {
        "scitype_from": "Series",
        "scitype_to": "Panel",
        "mtype_from": "np.ndarray",
        "mtype_to": "numpy3D",
        "python_type": np.ndarray,
    }

    def _convert(self, obj, store=None):
        if obj.ndim not in (1, 2):
            raise ValueError("if obj is np.ndarray, must be of dim 1 or 2")
        # numpy1D = (time), numpy2D = (time, variables)
        # numpy3D = (instances, variables, time)
        obj = obj.reshape(obj.shape[0], -1)
        return np.swapaxes(np.expand_dims(obj, 0), 1, 2)


class DfListToPdDataFrame(BaseScitypeConverter):
    """Conversion: df-list -> pd.DataFrame, Panel to Series scitype."""

    _tags = {
        "scitype_from": "Panel",
        "scitype_to": "Series",
        "mtype_from": "df-list",
        "mtype_to": "pd.DataFrame",
        "python_type": list,
    }

    def _convert(self, obj, store=None):
        if len(obj) != 1:
            raise ValueError("obj must be of length 1")
        return obj[0]


class PdMultiIndexToPdDataFrame(BaseScitypeConverter):
    """Conversion: pd-multiindex -> pd.DataFrame, Panel to Series scitype."""

    _tags = {
        "scitype_from": "Panel",
        "scitype_to": "Series",
        "mtype_from": "pd-multiindex",
        "mtype_to": "pd.DataFrame",
        "python_type": pd.DataFrame,
    }

    def _convert(self, obj, store=None):
        obj.index = obj.index.droplevel(level=0)
        return obj


class Numpy3DToNumpy(BaseScitypeConverter):
    """Conversion: numpy3D -> np.ndarray, Panel to Series scitype."""

    _tags = {
        "scitype_from": "Panel",
        "scitype_to": "Series",
        "mtype_from": "numpy3D",
        "mtype_to": "np.ndarray",
        "python_type": np.ndarray,
    }

    def _convert(self, obj, store=None):
        if obj.ndim != 3 or obj.shape[0] != 1:
            raise ValueError("if obj is np.ndarray, must be of dim 3, with shape[0]=1")
        # numpy2D = (time, variables)
        # numpy3D = (instances, variables, time)
        obj = np.reshape(obj, (obj.shape[1], obj.shape[2]))
        return np.swapaxes(obj, 0, 1)


class SeriesToHierarchical(BaseScitypeConverter):
    """Conversion: any mtype -> pd_multiindex_hier, Series to Hierarchical scitype."""

    _tags = {
        "scitype_from": "Series",
        "scitype_to": "Hierarchical",
        "mtype_to": "pd_multiindex_hier",
    }

    def _convert(self, obj, store=None):
        obj_df = convert_to(obj, to_type="pd.DataFrame", as_scitype="Series")
        obj_df = obj_df.copy()
        obj_df["__level1"] = 0
        obj_df["__level2"] = 0
        obj_df = obj_df.set_index(["__level1", "__level2"], append=True)
        return obj_df.reorder_levels([1, 2, 0])


class HierarchicalToSeries(BaseScitypeConverter):
    """Conversion: any mtype -> pd.DataFrame, Hierarchical to Series scitype."""

    _tags = {
        "scitype_from": "Hierarchical",
        "scitype_to": "Series",
        "mtype_to": "pd.DataFrame",
    }

    def _convert(self, obj, store=None):
        obj_df = convert_to(
            obj, to_type="pd_multiindex_hier", as_scitype="Hierarchical"
        )
        obj_df = obj_df.copy()
        obj_df.index = obj_df.index.get_level_values(-1)
        return obj_df


class PanelToHierarchical(BaseScitypeConverter):
    """Conversion: any mtype -> pd_multiindex_hier, Panel to Hierarchical scitype."""

    _tags = {
        "scitype_from": "Panel",
        "scitype_to": "Hierarchical",
        "mtype_to": "pd_multiindex_hier",
    }

    def _convert(self, obj, store=None):
        obj_df = convert_to(obj, to_type="pd-multiindex", as_scitype="Panel")
        obj_df = obj_df.copy()
        obj_df["__level2"] = 0
        obj_df = obj_df.set_index(["__level2"], append=True)
        return obj_df.reorder_levels([2, 0, 1])


class HierarchicalToPanel(BaseScitypeConverter):
    """Conversion: any mtype -> pd-multiindex, Hierarchical to Panel scitype."""

    _tags = {
        "scitype_from": "Hierarchical",
        "scitype_to": "Panel",
        "mtype_to": "pd-multiindex",
    }

    def _convert(self, obj, store=None):
        obj_df = convert_to(
            obj, to_type="pd_multiindex_hier", as_scitype="Hierarchical"
        )
        obj_df = obj_df.copy()
        obj_df.index = obj_df.index.get_level_values([-2, -1])
        return obj_df


def _convert_by_type(converters, obj, store=None, return_to_mtype=False):
    """Convert obj with the first converter in converters that applies to obj.

    If no converter applies, obj is returned unchanged.
    """
    for converter in converters:
        if converter.applies_to(obj):
            obj_converted = converter().convert(obj, store=store)
            if return_to_mtype:
                return obj_converted, converter.get_class_tag("mtype_to")
            return obj_converted
    if return_to_mtype:
        raise TypeError(f"no converter found for obj of type {type(obj)}")
    return obj


def convert_Series_to_Panel(obj, store=None, return_to_mtype=False):
    """Convert series to a single-series panel.

    Adds a dummy dimension to the series.
    For pd.Series or DataFrame, this results in a list of DataFrame (dim added is list).
    For numpy array, this results in a third dimension being added.

    Assumes input is conformant with one of the three Series mtypes.
    This method does not perform full mtype checks, use mtype or check_is_mtype for
    checks.

    Parameters
    ----------
    obj: an object of scitype Series, of mtype pd.DataFrame, pd.Series, or np.ndarray.
    store: dict, optional
        converter store for back-conversion
    return_to_mtype: bool, optional (default=False)
        if True, also returns the str of the mtype converted to

    Returns
    -------
    if obj was pd.Series or pd.DataFrame, returns a panel of mtype df-list
        this is done by possibly converting to pd.DataFrame, and adding a list nesting
    if obj was np.ndarray, returns a panel of mtype numpy3D, by adding one axis at end
    """
    return _convert_by_type(
        (PdSeriesToDfList, PdDataFrameToDfList, NumpyToNumpy3D),
        obj,
        store=store,
        return_to_mtype=return_to_mtype,
    )


def convert_Panel_to_Series(obj, store=None, return_to_mtype=False):
    """Convert single-series panel to a series.

    Removes panel index from the single-series panel to obtain a series.

    Assumes input is conformant with one of three main panel mtypes.
    This method does not perform full mtype checks, use mtype or check_is_mtype for
    checks.

    Parameters
    ----------
    obj: an object of scitype Panel, of mtype pd-multiindex, numpy3d, or df-list.
    store: dict, optional
        converter store for back-conversion
    return_to_mtype: bool, optional (default=False)
        if True, also returns the str of the mtype converted to

    Returns
    -------
    if obj df-list or pd-multiindex, returns a series of type pd.DataFrame
    if obj was numpy3D, returns a panel mtype np.ndarray
    """
    return _convert_by_type(
        (DfListToPdDataFrame, PdMultiIndexToPdDataFrame, Numpy3DToNumpy),
        obj,
        store=store,
        return_to_mtype=return_to_mtype,
    )


def convert_Series_to_Hierarchical(obj, store=None, return_to_mtype=False):
    """Convert series to a single-series hierarchical object.

    Adds two dimensions to the series to obtain a 3-level MultiIndex, 2 levels added.

    Assumes input is conformant with one of the three Series mtypes.
    This method does not perform full mtype checks, use mtype or check_is_mtype for
    checks.

    Parameters
    ----------
    obj: an object of scitype Series, of mtype pd.DataFrame, pd.Series, or np.ndarray.
    store: dict, optional
        converter store for back-conversion
    return_to_mtype: bool, optional (default=False)
        if True, also returns the str of the mtype converted to

    Returns
    -------
    returns a data container of mtype pd_multiindex_hier
    """
    return _convert_by_type(
        (SeriesToHierarchical,),
        obj,
        store=store,
        return_to_mtype=return_to_mtype,
    )


def convert_Hierarchical_to_Series(obj, store=None, return_to_mtype=False):
    """Convert single-series hierarchical object to a series.

    Removes two dimensions to obtain a series, by removing 2 levels from MultiIndex.

    Assumes input is conformant with Hierarchical mtype.
    This method does not perform full mtype checks, use mtype or check_is_mtype for
    checks.

    Parameters
    ----------
    obj: an object of scitype Hierarchical.
    store: dict, optional
        converter store for back-conversion
    return_to_mtype: bool, optional (default=False)
        if True, also returns the str of the mtype converted to

    Returns
    -------
    returns a data container of mtype pd.DataFrame, of scitype Series
    """
    return _convert_by_type(
        (HierarchicalToSeries,),
        obj,
        store=store,
        return_to_mtype=return_to_mtype,
    )


def convert_Panel_to_Hierarchical(obj, store=None, return_to_mtype=False):
    """Convert panel to a single-panel hierarchical object.

    Adds a dimensions to the panel to obtain a 3-level MultiIndex, 1 level is added.

    Assumes input is conformant with one of the Panel mtypes.
    This method does not perform full mtype checks, use mtype or check_is_mtype for
    checks.

    Parameters
    ----------
    obj: an object of scitype Panel.
    store: dict, optional
        converter store for back-conversion
    return_to_mtype: bool, optional (default=False)
        if True, also returns the str of the mtype converted to

    Returns
    -------
    returns a data container of mtype pd_multiindex_hier
    """
    return _convert_by_type(
        (PanelToHierarchical,),
        obj,
        store=store,
        return_to_mtype=return_to_mtype,
    )


def convert_Hierarchical_to_Panel(obj, store=None, return_to_mtype=False):
    """Convert single-series hierarchical object to a series.

    Removes one dimensions to obtain a panel, by removing 1 level from MultiIndex.

    Assumes input is conformant with Hierarchical mtype.
    This method does not perform full mtype checks, use mtype or check_is_mtype for
    checks.

    Parameters
    ----------
    obj: an object of scitype Hierarchical
    store: dict, optional
        converter store for back-conversion
    return_to_mtype: bool, optional (default=False)
        if True, also returns the str of the mtype converted to

    Returns
    -------
    returns a data container of mtype pd-multiindex, of scitype Panel
    """
    return _convert_by_type(
        (HierarchicalToPanel,),
        obj,
        store=store,
        return_to_mtype=return_to_mtype,
    )


def convert_to_scitype(
    obj,
    to_scitype,
    from_scitype=None,
    store=None,
    return_to_mtype=False,
):
    """Convert single-series or single-panel between mtypes.

    Assumes input is conformant with one of the mtypes
        for one of the scitypes Series, Panel, Hierarchical.
    This method does not perform full mtype checks, use mtype or check_is_mtype for
    checks.

    Parameters
    ----------
    obj : an object of scitype Series, Panel, or Hierarchical.
    to_scitype : str, scitype that obj should be converted to
    from_scitype : str, optional. Default = inferred from obj
        scitype that obj is of, and being converted from
        if avoided, function will skip type inference from obj
    store : dict, optional. Converter store for back-conversion.
    return_to_mtype: bool, optional (default=False)
        if True, also returns the str of the mtype converted to

    Returns
    -------
    obj of scitype to_scitype
        if converted to or from Hierarchical, the mtype will always be one of
            pd.DataFrame (Series), pd-multiindex (Panel), or pd_multiindex_hier
        if converted to or from Panel, mtype will attempt to keep python type
            e.g., np.ndarray (Series) converted to numpy3D (Panel) or back
            if not possible, will be one of the mtypes with pd.DataFrame python type
    """
    if from_scitype is None:
        from_scitype = scitype(
            obj, candidate_scitypes=["Series", "Panel", "Hierarchical"]
        )

    if to_scitype == from_scitype:
        return obj

    func_name = f"convert_{from_scitype}_to_{to_scitype}"
    func = eval(func_name)

    return func(obj, store=store, return_to_mtype=return_to_mtype)
