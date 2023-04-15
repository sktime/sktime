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

    """
    import pandas as pd

    from sktime.datatypes import convert_to

    instance_no = len(gluonts_result)
    global_ls = []
    per_instance_ls = []
    columns = []

    for i in range(instance_no):
        validation_no = gluonts_result[i].samples.shape[0]
        period = gluonts_result[i].samples.shape[1]
        start_date = pd.to_datetime(gluonts_result[i].start_date)
        freq = gluonts_result[i].freq
        ts_index = pd.date_range(start=start_date, periods=period, freq=freq)
        per_instance_ls = [
            pd.Series(data=gluonts_result[i].samples[j], index=ts_index)
            for j in range(validation_no)
        ]
        global_ls.append(per_instance_ls)

    for k in range(validation_no):
        columns.append("validation_" + str(k))

    nested_univ = pd.DataFrame(global_ls, columns=columns)

    return convert_to(nested_univ, to_type="pd-multiindex")
