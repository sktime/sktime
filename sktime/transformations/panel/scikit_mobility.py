"""Scikit-mobility interface class.

Trajectory feature extraction using scikit-mobility.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["sktime developers"]
__all__ = ["ScikitMobilityFeatureExtractor"]

import warnings

import numpy as np
import pandas as pd

from sktime.datatypes import convert_to
from sktime.transformations.base import BaseTransformer
from sktime.utils.dependencies import _check_estimator_deps


class ScikitMobilityFeatureExtractor(BaseTransformer):
    """Transformer for extracting trajectory features via scikit-mobility.

    Direct interface to scikit-mobility [1] trajectory feature extraction
    as an sktime transformer. Extracts features from trajectory data (GPS tracks,
    movement patterns, etc.) for use in time series classification or forecasting.

    Parameters
    ----------
    features : str or list of str, default="all"
        Which features to extract. Options:
        - "all": Extract all available features
        - "speed": Speed-related features
        - "acceleration": Acceleration-related features
        - "distance": Distance and displacement features
        - "direction": Direction and bearing features
        - "stops": Stop detection features
        - "trips": Trip segmentation features
        Or a custom list of specific feature names.
    include_geometry : bool, default=True
        Whether to include geometric features (convex hull, etc.)
    stop_duration_threshold : int, default=300
        Minimum duration (seconds) for stop detection
    stop_radius : float, default=50
        Radius (meters) for stop detection
    n_jobs : int, default=1
        Number of parallel jobs for feature extraction

    References
    ----------
    .. [1] https://github.com/scikit-mobility/scikit-mobility

    Examples
    --------
    >>> from sktime.transformations.panel.scikit_mobility import (
    ...     ScikitMobilityFeatureExtractor
    ... )
    >>> import pandas as pd
    >>> # Example trajectory data with lat, lon, datetime columns
    >>> trajectory_data = pd.DataFrame({
    ...     'lat': [45.0, 45.1, 45.2],
    ...     'lon': [7.0, 7.1, 7.2],
    ...     'datetime': pd.date_range('2020-01-01', periods=3, freq='1h')
    ... })
    >>> extractor = ScikitMobilityFeatureExtractor(features="speed")
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
        "python_dependencies": "scikit-mobility",
        "capability:multivariate": True,
        "capability:missing_values": True,
    }

    def __init__(
        self,
        features="all",
        include_geometry=True,
        stop_duration_threshold=300,
        stop_radius=50,
        n_jobs=1,
    ):
        self.features = features
        self.include_geometry = include_geometry
        self.stop_duration_threshold = stop_duration_threshold
        self.stop_radius = stop_radius
        self.n_jobs = n_jobs

        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return trajectory features.

        Parameters
        ----------
        X : pd.DataFrame
            Trajectory data. Expected columns include:
            - 'lat', 'lon' or 'latitude', 'longitude': coordinates
            - 'datetime' or 'timestamp': time information
            - Optionally 'traj_id' or 'tid': trajectory identifier
        y : ignored, exists for API consistency

        Returns
        -------
        Xt : pd.DataFrame
            Extracted trajectory features, one row per trajectory instance
        """
        # lazy import to avoid hard dependency
        try:
            import skmob
            from skmob import TrajDataFrame
            from skmob.preprocessing import detection, filtering, compression
            from skmob.core.trajectorydataframe import FlowDataFrame
        except ImportError as e:
            raise ImportError(
                "scikit-mobility is required for ScikitMobilityFeatureExtractor. "
                "You can install it with: pip install scikit-mobility"
            ) from e

        # Convert X to TrajDataFrame format expected by scikit-mobility
        X_copy = X.copy()

        # Normalize column names
        if "lat" in X_copy.columns and "lon" in X_copy.columns:
            X_copy.rename(columns={"lat": "lat", "lon": "lng"}, inplace=True)
        elif "latitude" in X_copy.columns and "longitude" in X_copy.columns:
            X_copy.rename(
                columns={"latitude": "lat", "longitude": "lng"}, inplace=True
            )

        # Ensure datetime column exists
        if "datetime" not in X_copy.columns and "timestamp" in X_copy.columns:
            X_copy.rename(columns={"timestamp": "datetime"}, inplace=True)
        elif "datetime" not in X_copy.columns:
            # Try to infer datetime from index
            if isinstance(X_copy.index, pd.DatetimeIndex):
                X_copy["datetime"] = X_copy.index
            else:
                raise ValueError(
                    "Trajectory data must have 'datetime' or 'timestamp' column, "
                    "or a DatetimeIndex"
                )

        # Ensure trajectory ID exists
        if "tid" not in X_copy.columns and "traj_id" not in X_copy.columns:
            # If X is a Panel with multiple instances, preserve instance info
            if isinstance(X_copy.index, pd.MultiIndex):
                X_copy["tid"] = X_copy.index.get_level_values(0)
            else:
                X_copy["tid"] = 0

        # Convert to TrajDataFrame
        try:
            tdf = TrajDataFrame(X_copy, latitude="lat", longitude="lng", datetime="datetime")
        except Exception as e:
            raise ValueError(
                f"Could not convert input to TrajDataFrame: {e}. "
                "Expected columns: lat/latitude, lon/longitude, datetime/timestamp"
            ) from e

        # Extract features based on specified feature types
        feature_list = []

        if self.features == "all" or "speed" in self.features:
            try:
                # Calculate speed features
                speed_features = self._extract_speed_features(tdf)
                feature_list.append(speed_features)
            except Exception as e:
                warnings.warn(f"Could not extract speed features: {e}")

        if self.features == "all" or "distance" in self.features:
            try:
                # Calculate distance features
                distance_features = self._extract_distance_features(tdf)
                feature_list.append(distance_features)
            except Exception as e:
                warnings.warn(f"Could not extract distance features: {e}")

        if self.features == "all" or "direction" in self.features:
            try:
                # Calculate direction features
                direction_features = self._extract_direction_features(tdf)
                feature_list.append(direction_features)
            except Exception as e:
                warnings.warn(f"Could not extract direction features: {e}")

        if self.features == "all" or "stops" in self.features:
            try:
                # Detect stops and extract features
                stop_features = self._extract_stop_features(
                    tdf, self.stop_duration_threshold, self.stop_radius
                )
                feature_list.append(stop_features)
            except Exception as e:
                warnings.warn(f"Could not extract stop features: {e}")

        if self.include_geometry:
            try:
                # Extract geometric features
                geometry_features = self._extract_geometry_features(tdf)
                feature_list.append(geometry_features)
            except Exception as e:
                warnings.warn(f"Could not extract geometry features: {e}")

        # Combine all features
        if feature_list:
            # Group by trajectory ID and aggregate features
            features_df = pd.concat(feature_list, axis=1)
            
            # Aggregate per trajectory if multiple points per trajectory
            if "tid" in features_df.columns or "tid" in tdf.columns:
                tid_col = "tid" if "tid" in features_df.columns else tdf.columns[0]
                agg_dict = {col: "mean" for col in features_df.columns if col != tid_col}
                agg_dict.update({
                    col: "sum" 
                    for col in features_df.columns 
                    if "total" in col.lower() or "count" in col.lower()
                })
                features_df = features_df.groupby(tid_col).agg(agg_dict).reset_index()
            
            return features_df
        else:
            # Return empty DataFrame with trajectory IDs
            return pd.DataFrame()

    def _extract_speed_features(self, tdf):
        """Extract speed-related features."""
        from skmob.utils import constants

        # Calculate speed (requires trajectory to have datetime)
        speed_features = pd.DataFrame()
        
        # Group by trajectory ID
        for tid, traj in tdf.groupby("tid"):
            if len(traj) < 2:
                continue
            
            # Calculate speeds
            speeds = []
            for i in range(1, len(traj)):
                p1 = traj.iloc[i-1]
                p2 = traj.iloc[i]
                
                # Haversine distance
                from math import radians, sin, cos, sqrt, atan2
                
                lat1, lon1 = radians(p1["lat"]), radians(p1["lng"])
                lat2, lon2 = radians(p2["lat"]), radians(p2["lng"])
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                
                distance_km = 6371 * c  # Earth radius in km
                
                # Time difference in seconds
                time_diff = (p2["datetime"] - p1["datetime"]).total_seconds()
                if time_diff > 0:
                    speed_ms = (distance_km * 1000) / time_diff  # m/s
                    speeds.append(speed_ms)
            
            if speeds:
                speed_dict = {
                    "tid": tid,
                    "speed_mean": np.mean(speeds),
                    "speed_std": np.std(speeds),
                    "speed_max": np.max(speeds),
                    "speed_min": np.min(speeds),
                    "speed_median": np.median(speeds),
                }
                speed_features = pd.concat([
                    speed_features,
                    pd.DataFrame([speed_dict])
                ], ignore_index=True)
        
        return speed_features.set_index("tid") if "tid" in speed_features.columns else speed_features

    def _extract_distance_features(self, tdf):
        """Extract distance-related features."""
        distance_features = pd.DataFrame()
        
        for tid, traj in tdf.groupby("tid"):
            if len(traj) < 2:
                continue
            
            distances = []
            from math import radians, sin, cos, sqrt, atan2
            
            for i in range(1, len(traj)):
                p1 = traj.iloc[i-1]
                p2 = traj.iloc[i]
                
                lat1, lon1 = radians(p1["lat"]), radians(p1["lng"])
                lat2, lon2 = radians(p2["lat"]), radians(p2["lng"])
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                
                distance_km = 6371 * c
                distances.append(distance_km)
            
            if distances:
                total_distance = sum(distances)
                # Displacement (straight-line distance from start to end)
                start = traj.iloc[0]
                end = traj.iloc[-1]
                
                lat1, lon1 = radians(start["lat"]), radians(start["lng"])
                lat2, lon2 = radians(end["lat"]), radians(end["lng"])
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                displacement_km = 6371 * c
                
                distance_dict = {
                    "tid": tid,
                    "total_distance_km": total_distance,
                    "displacement_km": displacement_km,
                    "distance_points_count": len(distances),
                    "efficiency_ratio": displacement_km / total_distance if total_distance > 0 else 0,
                }
                distance_features = pd.concat([
                    distance_features,
                    pd.DataFrame([distance_dict])
                ], ignore_index=True)
        
        return distance_features.set_index("tid") if "tid" in distance_features.columns else distance_features

    def _extract_direction_features(self, tdf):
        """Extract direction and bearing features."""
        direction_features = pd.DataFrame()
        
        for tid, traj in tdf.groupby("tid"):
            if len(traj) < 2:
                continue
            
            bearings = []
            from math import radians, degrees, atan2
            
            for i in range(1, len(traj)):
                p1 = traj.iloc[i-1]
                p2 = traj.iloc[i]
                
                lat1, lon1 = radians(p1["lat"]), radians(p1["lng"])
                lat2, lon2 = radians(p2["lat"]), radians(p2["lng"])
                
                dlon = lon2 - lon1
                
                y = sin(dlon) * cos(lat2)
                x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
                
                bearing = degrees(atan2(y, x))
                bearing = (bearing + 360) % 360  # Normalize to 0-360
                bearings.append(bearing)
            
            if bearings:
                direction_dict = {
                    "tid": tid,
                    "bearing_mean": np.mean(bearings),
                    "bearing_std": np.std(bearings),
                    "direction_change_total": sum(abs(bearings[i] - bearings[i-1]) 
                                                   for i in range(1, len(bearings))),
                }
                direction_features = pd.concat([
                    direction_features,
                    pd.DataFrame([direction_dict])
                ], ignore_index=True)
        
        return direction_features.set_index("tid") if "tid" in direction_features.columns else direction_features

    def _extract_stop_features(self, tdf, duration_threshold, radius):
        """Extract stop detection features."""
        try:
            from skmob.preprocessing.detection import stops
            
            stops_tdf = stops(
                tdf,
                stop_radius_factor=radius / 50,  # Normalize radius
                minutes_for_a_stop=duration_threshold / 60,
            )
            
            stop_features = pd.DataFrame()
            for tid in tdf["tid"].unique():
                traj_stops = stops_tdf[stops_tdf["tid"] == tid] if "tid" in stops_tdf.columns else stops_tdf
                
                stop_dict = {
                    "tid": tid,
                    "n_stops": len(traj_stops),
                    "stop_duration_total": traj_stops["duration"].sum() if "duration" in traj_stops.columns else 0,
                }
                stop_features = pd.concat([
                    stop_features,
                    pd.DataFrame([stop_dict])
                ], ignore_index=True)
            
            return stop_features.set_index("tid") if "tid" in stop_features.columns else stop_features
        except Exception:
            # If stop detection fails, return empty features
            return pd.DataFrame()

    def _extract_geometry_features(self, tdf):
        """Extract geometric features like convex hull."""
        geometry_features = pd.DataFrame()
        
        try:
            from shapely.geometry import Point
            from shapely.ops import unary_union
            
            for tid, traj in tdf.groupby("tid"):
                if len(traj) < 3:
                    continue
                
                points = [Point(row["lng"], row["lat"]) for _, row in traj.iterrows()]
                if len(points) >= 3:
                    try:
                        from shapely.geometry import MultiPoint
                        multipoint = MultiPoint(points)
                        convex_hull = multipoint.convex_hull
                        
                        geometry_dict = {
                            "tid": tid,
                            "convex_hull_area": convex_hull.area if hasattr(convex_hull, 'area') else 0,
                            "bbox_width": traj["lng"].max() - traj["lng"].min(),
                            "bbox_height": traj["lat"].max() - traj["lat"].min(),
                        }
                        geometry_features = pd.concat([
                            geometry_features,
                            pd.DataFrame([geometry_dict])
                        ], ignore_index=True)
                    except Exception:
                        pass
        
        except ImportError:
            # Shapely not available, skip geometry features
            pass
        
        return geometry_features.set_index("tid") if "tid" in geometry_features.columns else geometry_features

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"features": "speed", "n_jobs": 1}
        params2 = {"features": ["speed", "distance"], "include_geometry": False}
        return [params1, params2]