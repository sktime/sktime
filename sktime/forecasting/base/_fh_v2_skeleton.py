# from earlier experimentation with new ForecastingHorizonV2 design
# only kept as a reference
# to be deleted before merging the new design into the main branch
import numpy as np


class Frequency:
    def __init__(self, frequency: str):
        self.frequency = frequency
        # Additional frequency management logic can be added here


class _ForecastingHorizonValues:
    def __init__(self, values: np.ndarray):
        self.values = values
        # Additional storage functionalities can be implemented here

    def is_contiguous(self):
        return np.all(np.diff(self.values) == 1)

    def filter_in_sample(self, start, end):
        return self.values[(self.values >= start) & (self.values <= end)]

    def filter_out_of_sample(self, start, end):
        return self.values[(self.values < start) | (self.values > end)]


class ForecastingHorizonV2:
    def __init__(self, horizon_values: _ForecastingHorizonValues, frequency: Frequency):
        self.horizon_values = horizon_values
        self.frequency = frequency
        self.validate()

    def validate(self):
        # Comprehensive validation logic
        if not self.horizon_values.is_contiguous():
            raise ValueError("Horizon values must be contiguous.")
        # Additional checks can be added here

    def to_absolute(self):
        # Convert relative horizon to absolute values
        # Placeholder for implementation
        return self.horizon_values.values

    def to_relative(self):
        # Convert absolute horizon to relative values
        # Placeholder for implementation
        return self.horizon_values.values

    def check_contiguity(self):
        return self.horizon_values.is_contiguous()

    def handle_multiindex(self, index):
        # Handle MultiIndex
        # Placeholder for implementation
        pass
