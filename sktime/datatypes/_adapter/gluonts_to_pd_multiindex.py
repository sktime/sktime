# -*- coding: utf-8 -*-
def convert_gluonts_result_to_multiindex(gluonts_result):
    """

    Back Convert from Gluonts to sktime.

    Convert the output of Gluonts's prediction to a multiindex
    dataframe compatible with sktime.

    Parameters
    ----------
    gluonts_result: The first element of the tuple resulting
    from running `make_evaluation_predictions`.
        For example in Eg:
        forecast_it, ts_it = make_evaluation_predictions()
        gluonts_result = forecast_it

    Returns
    -------
    A MultiIndex DF mtype type compatible with sktime.

    Examples
    --------
    >>> from gluonts.dataset.util import to_pandas
    >>> from gluonts.dataset.pandas import PandasDataset
    >>> from gluonts.dataset.repository.datasets import get_dataset
    >>> from gluonts.mx.model.simple_feedforward
    ... import SimpleFeedForwardEstimator
    >>> from gluonts.mx import Trainer
    >>> import matplotlib.pyplot as plt
    >>> from sktime.datatypes._adapter import gluonts_to_pd_multiindex


    >>> dataset = get_dataset("airpassengers")
    >>> feedforward = SimpleFeedForwardEstimator(prediction_length=12,
    ... trainer=Trainer(epochs=5))
    >>> model = feedforward.train(dataset.train)

    # Make predictions
    >>> true_values = to_pandas(list(dataset.test)[0])
    >>> true_values.to_timestamp().plot(color="k")
    # Predict the last 5 months
    >>> prediction_input = PandasDataset(true_values[:-12])
    >>> predictions = model.predict(prediction_input)
    >>> result_ls = list(predictions)
    >>> convert_gluonts_result_to_multiindex(result_ls)

    """
    import pandas as pd

    from sktime.datatypes import convert_to

    instance_no = len(gluonts_result)
    global_ls = []
    per_instance_ls = []
    columns = []
    validation_no = gluonts_result[0].samples.shape[0]

    for i in range(instance_no):

        period = gluonts_result[i].samples.shape[1]
        start_date = gluonts_result[i].start_date.to_timestamp()
        freq = gluonts_result[i].freq
        ts_index = pd.date_range(start=start_date, periods=period, freq=freq)
        per_instance_ls = [
            pd.Series(data=gluonts_result[i].samples[j], index=ts_index)
            for j in range(validation_no)
        ]
        global_ls.append(per_instance_ls)

    for k in range(validation_no):
        columns.append("result_" + str(k))

    nested_univ = pd.DataFrame(global_ls, columns=columns)

    return convert_to(nested_univ, to_type="pd-multiindex")
