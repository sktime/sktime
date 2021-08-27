
# %%
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sktime.transformations.series.calendardummies import CalendarDummies
from sktime.forecasting.compose import ForecastingPipeline
from sktime.forecasting.naive import NaiveForecaster
from sktime.datasets import load_airline
import pandas as pd
from sktime.datasets import load_longley
import pytest
import os
os.chdir("/home/ds40/newhome/repo/sktime/")

#y, X = load_airline(),pd.DataFrame(index=load_airline().index)
y, X = load_longley()
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

fh = ForecastingHorizon(X_test.index, is_relative=False)
pipe = ForecastingPipeline(steps=[
    ("CalendarDummies", CalendarDummies(base_frequency="week", complexity=2)),
    ("forecaster", NaiveForecaster(strategy="drift"))])

pipe.steps[0][1].fit(Z=X_train)
first_test = pipe.steps[0][1].transform(Z=X_train).columns.to_list()

# %%
pipe = ForecastingPipeline(steps=[
    ("CalendarDummies", CalendarDummies(base_frequency="week", complexity=0)),
    ("forecaster", NaiveForecaster(strategy="drift"))])

pipe.steps[0][1].fit(Z=X_train)
second_test = pipe.steps[0][1].transform(Z=X_train).columns.to_list()

# %%
pipe = ForecastingPipeline(
    steps=[
        ("CalendarDummies", CalendarDummies(base_frequency="month", complexity=2)),
        ("forecaster", NaiveForecaster(strategy="drift"))])

pipe.steps[0][1].fit(Z=X_train)
third_test = pipe.steps[0][1].transform(Z=X_train).columns.to_list()
print(third_test)
# %%

pipe = ForecastingPipeline(steps=[
    ("CalendarDummies", CalendarDummies(base_frequency="month",
     complexity=2, manual_selection=["year_of_year", "second_of_minute"])),
    ("forecaster", NaiveForecaster(strategy="drift"))])

pipe.steps[0][1].fit(Z=X_train)
fourth_test = pipe.steps[0][1].transform(Z=X_train).columns.to_list()
print(fourth_test)

# %%

pipe = ForecastingPipeline(steps=[
    ("CalendarDummies", CalendarDummies(
        manual_selection=["year_of_year", "second_of_minute"])),
    ("forecaster", NaiveForecaster(strategy="drift"))])

pipe.steps[0][1].fit(Z=X_train)
fifth_test = pipe.steps[0][1].transform(Z=X_train).columns.to_list()
print(fifth_test)
# %%
y, X = load_airline(), pd.DataFrame(index=load_airline().index)
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

pipe = ForecastingPipeline(steps=[
    ("CalendarDummies", CalendarDummies(
        manual_selection=["year_of_year", "second_of_minute"])),
    ("forecaster", NaiveForecaster(strategy="drift"))])

pipe.steps[0][1].fit(Z=X_train)
sixth_test = pipe.steps[0][1].transform(Z=X_train).columns.to_list()
print(sixth_test)

# %%
test = load_airline()
test.index = test.index.view("int64")
y, X = test, pd.DataFrame(index=test.index)

pipe = ForecastingPipeline(steps=[
    ("CalendarDummies", CalendarDummies(
        manual_selection=["year_of_year", "second_of_minute"])),
    ("forecaster", NaiveForecaster(strategy="drift"))])

pipe.steps[0][1].fit(Z=X_train)
sixth_test = pipe.steps[0][1].transform(Z=X_train).columns.to_list()
print(sixth_test)
# %%
y, X = load_airline(), pd.DataFrame(index=load_airline().index)
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

pipe = ForecastingPipeline(steps=[
    ("CalendarDummies", CalendarDummies(
        manual_selection=["year_of_year", "second_of_minute"])),
    ("forecaster", NaiveForecaster(strategy="drift"))])

pipe.steps[0][1].fit(Z=X_train)
seventh_test = pipe.steps[0][1].transform(Z=X_train).columns.to_list()
print(seventh_test)

# %%


@pytest.mark.parametrize("test_input,expected", [
    (first_test, ['GNPDEFL', 'GNP', 'UNEMP', 'ARMED', 'POP', 'year', 'quarter',
     'month', 'week_of_year', 'month_of_quarter', 'week_of_quarter', 'week_of_month']),
    (second_test, ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "year", "month"]),
    (third_test, ['GNPDEFL', 'GNP', 'UNEMP', 'ARMED', 'POP',
     'year', 'quarter', 'month', 'month_of_quarter']),
    (fourth_test, ['GNPDEFL', 'GNP', 'UNEMP', 'ARMED', 'POP', 'year']),
    (fifth_test, ['GNPDEFL', 'GNP', 'UNEMP', 'ARMED', 'POP', 'year', 'second']),
    (sixth_test, ['year', 'second']),
    (seventh_test, ['year', 'second'])
    ])
def test_eval(test_input, expected):
    assert len(test_input) == len(expected)
    assert all([a == b for a, b in zip(test_input, expected)])

# %%
