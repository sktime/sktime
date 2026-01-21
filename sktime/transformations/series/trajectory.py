"""Trajectory transformation interfaces for movingpandas."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["neha222222"]
__all__ = ["DouglasPeuckerTrajectoryGeneralizer"]

import pandas as pd

from sktime.transformations.base import BaseTransformer


class DouglasPeuckerTrajectoryGeneralizer(BaseTransformer):
    """Douglas-Peucker trajectory generalization transformer.

    Wraps movingpandas DouglasPeuckerGeneralizer for trajectory simplification.
    Reduces the number of points in a trajectory while preserving its shape.

    Parameters
    ----------
    tolerance : float, default=0.001
        Distance tolerance for the Douglas-Peucker algorithm.
        Higher values result in more aggressive simplification.
    lat_col : str, default="lat"
        Name of the latitude column in input data.
    lon_col : str, default="lon"
        Name of the longitude column in input data.
    datetime_col : str, default=None
        Name of the datetime column. If None, uses the index.

    Examples
    --------
    >>> from sktime.transformations.series.trajectory import (
    ...     DouglasPeuckerTrajectoryGeneralizer
    ... )
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create sample trajectory data
    >>> dates = pd.date_range('2023-01-01', periods=100, freq='h')
    >>> df = pd.DataFrame({
    ...     'lat': np.linspace(0, 1, 100) + np.random.randn(100) * 0.01,
    ...     'lon': np.linspace(0, 1, 100) + np.random.randn(100) * 0.01,
    ... }, index=dates)
    >>> transformer = DouglasPeuckerTrajectoryGeneralizer(tolerance=0.01)
    >>> df_simplified = transformer.fit_transform(df)  # doctest: +SKIP
    """

    _tags = {
        "authors": ["neha222222"],
        "python_dependencies": ["movingpandas", "geopandas", "shapely"],
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "None",
        "fit_is_empty": True,
        "capability:inverse_transform": False,
    }

    def __init__(
        self,
        tolerance=0.001,
        lat_col="lat",
        lon_col="lon",
        datetime_col=None,
    ):
        self.tolerance = tolerance
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.datetime_col = datetime_col

        super().__init__()

    def _transform(self, X, y=None):
        """Transform X by applying Douglas-Peucker generalization.

        Parameters
        ----------
        X : pd.DataFrame
            Trajectory data with lat/lon columns and datetime index.
        y : ignored

        Returns
        -------
        Xt : pd.DataFrame
            Simplified trajectory data.
        """
        import geopandas as gpd
        import movingpandas as mpd
        from shapely.geometry import Point

        df = X.copy()

        if self.datetime_col is not None:
            df = df.set_index(self.datetime_col)

        geometry = [
            Point(xy) for xy in zip(df[self.lon_col], df[self.lat_col])
        ]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        traj = mpd.Trajectory(gdf, traj_id=1)

        generalizer = mpd.DouglasPeuckerGeneralizer(traj)
        simplified_traj = generalizer.generalize(tolerance=self.tolerance)

        result_gdf = simplified_traj.df
        result_df = pd.DataFrame(result_gdf.drop(columns="geometry"))
        result_df[self.lat_col] = [p.y for p in result_gdf.geometry]
        result_df[self.lon_col] = [p.x for p in result_gdf.geometry]

        return result_df

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
        """
        params1 = {"tolerance": 0.01, "lat_col": "lat", "lon_col": "lon"}
        params2 = {"tolerance": 0.001}
        return [params1, params2]

