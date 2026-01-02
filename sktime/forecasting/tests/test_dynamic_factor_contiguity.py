def test_dynamicfactor_predict_interval_non_contiguous_fh():
    fh = ForecastingHorizon([1, 3, 5])
    model = DynamicFactor(...)
    model.fit(y)

    y_pred_int = model.predict_interval(fh=fh)

    assert list(y_pred_int.index) == fh.to_absolute(...)
