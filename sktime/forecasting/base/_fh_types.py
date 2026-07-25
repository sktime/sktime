from datetime import timedelta

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies


class FhIntTypeNonPandas:
    """Normalizer for integer-like forecasting horizons, numpy and python native."""

    def _is_applicable(self, fh):
        """Check if this normalizer class is applicable to the input fh.

        Parameters
        ----------
        fh : input object
            The input forecasting horizon to check for applicability.

        Returns
        -------
        bool
             True if this normalizer class can be used to normalize the input fh.
        """
        if self._is_int(fh):
            return True

        # range object
        if isinstance(fh, range):
            return True
        # numpy array
        if isinstance(fh, np.ndarray) and np.issubdtype(fh.dtype, np.integer):
            return True
        # list of integer
        if isinstance(fh, list) and all([self._is_int(value) for value in fh]):
            return True

        return False

    def _normalize(self, fh):
        """Coerce fh to 1D np.array of integers (periods) or floats (if fractional).

        Parameters
        ----------
        fh : input object
            The input object to normalize.

        Returns
        -------
        np.array, 1D array of integers or floats
            Normalized forecasting horizon as a 1D numpy array of integers or floats.
        """
        if self._is_int(fh) and fh > 0:
            return np.arange(1, fh + 1, dtype=int)
        elif self._is_int(fh):
            return np.array([fh])

        if isinstance(fh, range):
            return np.array(fh)
        else:
            return np.asarray(fh).flatten()

    @staticmethod
    def _is_int(x) -> bool:
        """Check if x is of integer type, but not boolean."""
        # boolean are subclasses of integers in Python, so explicitly exclude them
        return (
            isinstance(x, (int, np.integer))
            and not isinstance(x, bool)
            and not isinstance(x, np.timedelta64)
        )

    def _normalize_pd_index_legacy(self, fh):
        """Legacy normalizer for pandas index types, for downwards compatibility."""
        import pandas as pd

        vals = self._normalize(fh)
        # downwards compatible, tests expect length zero values to be RangeIndex
        if len(vals) == 0:
            return pd.RangeIndex(0)

        return pd.Index(vals)


class FhPandasIntType:
    """Normalizer for integer-like forecasting horizons, pandas only."""

    def _is_applicable(self, fh):
        """Check if this normalizer class is applicable to the input fh.

        Parameters
        ----------
        fh : input object
            The input forecasting horizon to check for applicability.

        Returns
        -------
        bool
             True if this normalizer class can be used to normalize the input fh.
        """
        if not _check_soft_dependencies("pandas", severity="none"):
            return False

        import pandas as pd

        if isinstance(fh, pd.RangeIndex):
            return True

        if isinstance(fh, pd.Index) and pd.api.types.is_integer_dtype(fh):
            return True

        return False

    def _normalize(self, fh):
        """Coerce fh to 1D np.array of integers (periods) or floats (if fractional).

        Parameters
        ----------
        fh : input object
            The input object to normalize.

        Returns
        -------
        np.array, 1D array of integers or floats
            Normalized forecasting horizon as a 1D numpy array of integers or floats.
        """
        return fh.to_numpy().flatten()

    def _normalize_pd_index_legacy(self, fh):
        """Legacy normalizer for pandas index types, for downwards compatibility."""
        # already a pd.Index, so return as is
        return fh


class FhDateTypeNumpyOrPandas:
    """Normalizer for date-like forecasting horizons, numpy and pandas."""

    def _is_applicable(self, fh):
        """Check if this normalizer class is applicable to the input fh.

        Parameters
        ----------
        fh : input object
            The input forecasting horizon to check for applicability.

        Returns
        -------
        bool
             True if this normalizer class can be used to normalize the input fh.
        """
        if not _check_soft_dependencies("pandas", severity="none"):
            return False

        import pandas as pd

        VALID_INDEX_TYPES = (pd.PeriodIndex, pd.DatetimeIndex, pd.TimedeltaIndex)

        if isinstance(fh, VALID_INDEX_TYPES):
            return True

        if self._is_timedelta_or_date_offset(fh):
            return True

        if isinstance(fh, (list, np.ndarray)):
            if all([self._is_timedelta_or_date_offset(value) for value in fh]):
                return True

        return False

    def _is_timedelta_or_date_offset(self, x) -> bool:
        """Check if x is of timedelta or pd.DateOffset type."""
        import pandas as pd

        # ACCEPTED_DATETIME_TYPES = np.datetime64, pd.Timestamp
        ACCEPTED_TIMEDELTA_TYPES = pd.Timedelta, timedelta, np.timedelta64
        ACCEPTED_DATEOFFSET_TYPES = pd.DateOffset

        def is_timedelta(x) -> bool:
            """Check if x is of timedelta type."""
            return isinstance(x, ACCEPTED_TIMEDELTA_TYPES)

        def is_date_offset(x) -> bool:
            """Check if x is of pd.DateOffset type."""
            return isinstance(x, ACCEPTED_DATEOFFSET_TYPES)

        return is_timedelta(x=x) or is_date_offset(x=x)

    @staticmethod
    def _to_int64(x):
        """Convert datetime-like or timedelta-like x to int64 representation."""
        import pandas as pd

        # Timestamp-like: ns since Unix epoch
        if isinstance(x, pd.Timestamp):
            return x.value

        if isinstance(x, np.datetime64):
            return x.astype("datetime64[ns]").astype(np.int64)

        # Timedelta-like: ns duration
        if isinstance(x, pd.Timedelta):
            return x.value

        if isinstance(x, timedelta):
            return pd.Timedelta(x).value

        if isinstance(x, np.timedelta64):
            return x.astype("timedelta64[ns]").astype(np.int64)

        raise TypeError(f"Unsupported type: {type(x)}")

    def _normalize(self, fh):
        """Coerce fh to 1D np.array of integers (periods) or floats (if fractional).

        Parameters
        ----------
        fh : input object
            The input object to normalize.

        Returns
        -------
        np.array, 1D array of integers or floats
            Normalized forecasting horizon as a 1D numpy array of integers or floats.
        """
        import pandas as pd

        # number of periods
        if isinstance(fh, pd.PeriodIndex):
            return fh.astype("int64")

        # nanoseconds since 1970-01-01 00:00:00 UTC, as integer
        # this is the usual convention representing datetimes internally
        if isinstance(fh, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            return fh.asi8.flatten()

        if self._is_timedelta_or_date_offset(fh):
            return np.array([self._to_int64(fh)])

        if isinstance(fh, (list, np.ndarray)):
            if all([self._is_timedelta_or_date_offset(value) for value in fh]):
                return np.array([self._to_int64(value) for value in fh]).flatten()

        raise RuntimeError("Please only pass fh that pass _is_applicable.")

    def _normalize_pd_index_legacy(self, fh):
        """Legacy normalizer for pandas index types, for downwards compatibility."""
        import pandas as pd

        VALID_INDEX_TYPES = (pd.PeriodIndex, pd.DatetimeIndex, pd.TimedeltaIndex)

        if isinstance(fh, VALID_INDEX_TYPES):
            return fh

        if self._is_timedelta_or_date_offset(fh):
            return pd.Index([fh])

        if isinstance(fh, (list, np.ndarray)):
            if all([self._is_timedelta_or_date_offset(value) for value in fh]):
                return pd.Index(fh)

        raise RuntimeError("Please only pass fh that pass _is_applicable.")


ALL_NORMALIZERS = [FhIntTypeNonPandas, FhPandasIntType, FhDateTypeNumpyOrPandas]
