"""MovingPandas interface class.

Trajectory feature extraction using MovingPandas.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["sktime developers"]
__all__ = ["MovingPandasFeatureExtractor"]

import warnings

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer


class MovingPandasFeatureExtractor(BaseTransformer):
    """Transformer for extracting trajectory features via MovingPandas.

    Direct interface to MovingPandas [1] trajectory analysis features
    as an sktime transformer. Extracts features from trajectory data for use
    in time series classification or forecasting.

    Parameters
    ----------
    features : str or list of str, default="all"
        Which features to extract. Options:
        - "all": Extract all available features
        - "speed": Speed-related features
        - "direction": Direction and heading features
        - "distance": Distance and displacement features
        - "time": Temporal features
        - "geometry": Geometric features
    speed_col : str, default="speed"
        Column name for speed (will be calculated if not present)
    direction_col : str, default="direction"
        Column name for direction (will be calculated if not present)

    References
    ----------
    .. [1] https://github.com/movingpandas/movingpandas

    Examples
    --------
    >>> from sktime.transformations.panel.movingpandas import (
    ...     MovingPandasFeatureExtractor
    ... )
    >>> import pandas as pd
    >>> # Example trajectory data
    >>> trajectory_data = pd.DataFrame({
    ...     'lat': [45.0, 45.1, 45.2],
    ...     'lon': [7.0, 7.1, 7.2],
    ...     'datetime': pd.date_range('2020-01-01', periods=3, freq='1h')
    ... })
    >>> extractor = MovingPandasFeatureExtractor(features="speed")
    >>> features = extractor.fit_transform(trajectory_data)
    """

    _tags = {
        "authors": ["sktime developers"],
        "maintainers": ["sktime developers"],
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": True,
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "None",
        "fit_is_empty": True,
        "python_dependencies": "movingpandas",
        "capability:multivariate": True,
        "capability:missing_values": True,
    }

    def __init__(
        self,
        features="all",
        speed_col="speed",
        direction_col="direction",
    ):
        self.features = features
        self.speed_col = speed_col
        self.direction_col = direction_col

        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return trajectory features.

        Parameters
        ----------
        X : pd.DataFrame
            Trajectory data with geometry column or lat/lon columns
        y : ignored, exists for API consistency

        Returns
        -------
        Xt : pd.DataFrame
            Extracted trajectory features, one row per trajectory instance
        """
        # lazy import to avoid hard dependency
        try:
            import geopandas as gpd
            import movingpandas as mpd
            from movingpandas import Trajectory
            from shapely.geometry import Point
        except ImportError as e:
            raise ImportError(
                "movingpandas and geopandas are required for MovingPandasFeatureExtractor. "
                "You can install them with: pip install movingpandas geopandas"
            ) from e

        X_copy = X.copy()

        # Convert to GeoDataFrame if not already
        if not isinstance(X_copy, gpd.GeoDataFrame):
            if "geometry" not in X_copy.columns:
                # OPTIMIZED: Use points_from_xy
                if "lat" in X_copy.columns and "lon" in X_copy.columns:
                    X_copy["geometry"] = gpd.points_from_xy(
                        X_copy["lon"], X_copy["lat"]
                    )
                elif "latitude" in X_copy.columns and "longitude" in X_copy.columns:
                    X_copy["geometry"] = gpd.points_from_xy(
                        X_copy["longitude"], X_copy["latitude"]
                    )
                else:
                    raise ValueError(
                        "Input must have 'geometry', 'lat'/'lon', or 'latitude'/'longitude'"
                    )

            # Ensure datetime column
            if "datetime" not in X_copy.columns and "t" not in X_copy.columns:
                if isinstance(X_copy.index, pd.DatetimeIndex):
                    X_copy["datetime"] = X_copy.index
                else:
                    raise ValueError(
                        "Trajectory data must have 'datetime' or 't' column, "
                        "or a DatetimeIndex"
                    )
            elif "t" in X_copy.columns:
                X_copy["datetime"] = X_copy["t"]

            # Ensure trajectory ID
            if "traj_id" not in X_copy.columns:
                if isinstance(X_copy.index, pd.MultiIndex):
                    X_copy["traj_id"] = X_copy.index.get_level_values(0)
                else:
                    X_copy["traj_id"] = 0

            # Convert to GeoDataFrame
            gdf = gpd.GeoDataFrame(X_copy, geometry="geometry", crs="EPSG:4326")
        else:
            gdf = X_copy

        # Extract features
        feature_list = []

        if self.features == "all" or "speed" in self.features:
            try:
                speed_features = self._extract_speed_features(gdf)
                feature_list.append(speed_features)
            except Exception as e:
                warnings.warn(f"Could not extract speed features: {e}")

        if self.features == "all" or "distance" in self.features:
            try:
                distance_features = self._extract_distance_features(gdf)
                feature_list.append(distance_features)
            except Exception as e:
                warnings.warn(f"Could not extract distance features: {e}")

        if self.features == "all" or "direction" in self.features:
            try:
                direction_features = self._extract_direction_features(gdf)
                feature_list.append(direction_features)
            except Exception as e:
                warnings.warn(f"Could not extract direction features: {e}")

        if self.features == "all" or "time" in self.features:
            try:
                time_features = self._extract_time_features(gdf)
                feature_list.append(time_features)
            except Exception as e:
                warnings.warn(f"Could not extract time features: {e}")

        if self.features == "all" or "geometry" in self.features:
            try:
                geometry_features = self._extract_geometry_features(gdf)
                feature_list.append(geometry_features)
            except Exception as e:
                warnings.warn(f"Could not extract geometry features: {e}")

        # Combine features
        if feature_list:
            features_df = pd.concat(feature_list, axis=1)

            # Remove duplicate columns
            features_df = features_df.loc[:, ~features_df.columns.duplicated()]

            # Aggregate per trajectory if needed
            if "traj_id" in features_df.columns:
                tid_col = "traj_id"
                agg_dict = {
                    col: "mean"
                    for col in features_df.columns
                    if col != tid_col
                    and features_df[col].dtype in [np.float64, np.float32, np.int64]
                }
                agg_dict.update(
                    {
                        col: "sum"
                        for col in features_df.columns
                        if "total" in col.lower() or "count" in col.lower()
                    }
                )
                features_df = features_df.groupby(tid_col).agg(agg_dict).reset_index()

            return features_df
        else:
            return pd.DataFrame()

    def _extract_speed_features(self, gdf):
        """Extract speed features using MovingPandas."""
        try:
            from movingpandas import Trajectory

            speed_features = pd.DataFrame()

            for traj_id, traj_group in gdf.groupby("traj_id"):
                if len(traj_group) < 2:
                    continue

                # Create Trajectory object
                traj = Trajectory(
                    traj_group,
                    traj_id=traj_id,
                    t="datetime" if "datetime" in traj_group.columns else "t",
                )

                # Calculate speed
                traj.add_speed(overwrite=True)

                speeds = traj.df[
                    self.speed_col if self.speed_col in traj.df.columns else "speed"
                ].dropna()

                if len(speeds) > 0:
                    speed_dict = {
                        "traj_id": traj_id,
                        "speed_mean": speeds.mean(),
                        "speed_std": speeds.std(),
                        "speed_max": speeds.max(),
                        "speed_min": speeds.min(),
                        "speed_median": speeds.median(),
                    }
                    speed_features = pd.concat(
                        [speed_features, pd.DataFrame([speed_dict])], ignore_index=True
                    )

            return (
                speed_features.set_index("traj_id")
                if "traj_id" in speed_features.columns
                else speed_features
            )
        except Exception as e:
            warnings.warn(f"Speed feature extraction failed: {e}")
            return pd.DataFrame()

    def _extract_distance_features(self, gdf):
        """Extract distance features."""
        distance_features = pd.DataFrame()

        for traj_id, traj_group in gdf.groupby("traj_id"):
            if len(traj_group) < 2:
                continue

            try:
                from movingpandas import Trajectory

                traj = Trajectory(
                    traj_group,
                    traj_id=traj_id,
                    t="datetime" if "datetime" in traj_group.columns else "t",
                )

                # Get trajectory length
                length_m = traj.get_length()

                # Displacement
                start = traj.get_start_location()
                end = traj.get_end_location()
                displacement_m = start.distance(end) * 111000  # Approximate conversion

                distance_dict = {
                    "traj_id": traj_id,
                    "total_distance_m": length_m,
                    "displacement_m": displacement_m,
                    "distance_points_count": len(traj_group),
                    "efficiency_ratio": displacement_m / length_m
                    if length_m > 0
                    else 0,
                }
                distance_features = pd.concat(
                    [distance_features, pd.DataFrame([distance_dict])],
                    ignore_index=True,
                )
            except Exception:
                pass

        return (
            distance_features.set_index("traj_id")
            if "traj_id" in distance_features.columns
            else distance_features
        )

    def _extract_direction_features(self, gdf):
        """Extract direction features."""
        try:
            from movingpandas import Trajectory

            direction_features = pd.DataFrame()

            for traj_id, traj_group in gdf.groupby("traj_id"):
                if len(traj_group) < 2:
                    continue

                traj = Trajectory(
                    traj_group,
                    traj_id=traj_id,
                    t="datetime" if "datetime" in traj_group.columns else "t",
                )

                # Add direction
                traj.add_direction(overwrite=True)

                directions = traj.df[
                    self.direction_col
                    if self.direction_col in traj.df.columns
                    else "direction"
                ].dropna()

                if len(directions) > 0:
                    direction_dict = {
                        "traj_id": traj_id,
                        "direction_mean": directions.mean(),
                        "direction_std": directions.std(),
                        "direction_change_total": sum(
                            abs(directions.iloc[i] - directions.iloc[i - 1])
                            for i in range(1, len(directions))
                        ),
                    }
                    direction_features = pd.concat(
                        [direction_features, pd.DataFrame([direction_dict])],
                        ignore_index=True,
                    )

            return (
                direction_features.set_index("traj_id")
                if "traj_id" in direction_features.columns
                else direction_features
            )
        except Exception:
            return pd.DataFrame()

    def _extract_time_features(self, gdf):
        """Extract temporal features."""
        time_features = pd.DataFrame()

        for traj_id, traj_group in gdf.groupby("traj_id"):
            if len(traj_group) < 2:
                continue

            datetime_col = "datetime" if "datetime" in traj_group.columns else "t"
            datetimes = pd.to_datetime(traj_group[datetime_col])

            time_dict = {
                "traj_id": traj_id,
                "duration_seconds": (datetimes.max() - datetimes.min()).total_seconds(),
                "time_points_count": len(traj_group),
                "time_interval_mean": datetimes.diff().dropna().mean().total_seconds()
                if len(datetimes) > 1
                else 0,
            }
            time_features = pd.concat(
                [time_features, pd.DataFrame([time_dict])], ignore_index=True
            )

        return (
            time_features.set_index("traj_id")
            if "traj_id" in time_features.columns
            else time_features
        )

    def _extract_geometry_features(self, gdf):
        """Extract geometric features."""
        geometry_features = pd.DataFrame()

        try:
            from shapely.geometry import LineString

            for traj_id, traj_group in gdf.groupby("traj_id"):
                if len(traj_group) < 2:
                    continue

                # Create line from trajectory
                line = LineString(traj_group.geometry.tolist())

                # Calculate bounding box
                bounds = line.bounds

                geometry_dict = {
                    "traj_id": traj_id,
                    "trajectory_length_deg": line.length,
                    "bbox_width": bounds[2] - bounds[0],
                    "bbox_height": bounds[3] - bounds[1],
                    "bbox_area": (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]),
                }
                geometry_features = pd.concat(
                    [geometry_features, pd.DataFrame([geometry_dict])],
                    ignore_index=True,
                )
        except Exception:
            pass

        return (
            geometry_features.set_index("traj_id")
            if "traj_id" in geometry_features.columns
            else geometry_features
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"features": "speed"}
        params2 = {"features": ["speed", "distance"]}
        return [params1, params2]
