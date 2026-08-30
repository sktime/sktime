from sktime.datasets import load_airline
from sktime.forecasting.naive import NaiveForecaster


def test_naive_update_advances_cutoff():
    y = load_airline()
    f = NaiveForecaster(strategy="last")
    f.fit(y[:-3])

    old_cutoff = f.cutoff
    f.update(y[-3:])

    assert f.cutoff > old_cutoff


def test_naive_update_changes_prediction_last():
    y = load_airline()
    f = NaiveForecaster(strategy="last")
    f.fit(y[:-1])

    pred_before = f.predict(fh=[1])
    f.update(y[-1:])
    pred_after = f.predict(fh=[1])

    # compare scalar values to avoid index/metadata differences
    assert pred_before.iloc[0] != pred_after.iloc[0]


def test_naive_update_update_params_false_updates_data_only():
    y = load_airline()
    f = NaiveForecaster(strategy="last")
    f.fit(y[:-1])

    # capture lightweight derived attributes if they exist
    old_window_length = getattr(f, "window_length_", None)
    old_sp = getattr(f, "sp_", None)

    f.update(y[-1:], update_params=False)

    # data should be updated (latest value changed)
    assert f._y.iloc[-1] == y.iloc[-1]

    # derived attributes should be unchanged when update_params=False
    assert getattr(f, "window_length_", None) == old_window_length
    assert getattr(f, "sp_", None) == old_sp
