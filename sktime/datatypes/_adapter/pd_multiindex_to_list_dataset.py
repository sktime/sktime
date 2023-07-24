def convert_from_multiindex_to_listdataset(trainDF, class_val_list=None):
    """Output a dataset in ListDataset format compatible with gluonts.

    Parameters
    ----------
    trainDF: Multiindex dataframe
        Input DF should be multi-index DataFrame.
        Time index must be absolute.

    class_val_list: str
        List of classes in case of classification dataset.
        If not available, class_val_list will show instance numbers
    freq: str, default="1D"
        Pandas-compatible frequency to be used.
        Only fixed frequency is supported at the moment.
    startdate: str, default = "1750-01-01"
        Custom startdate for ListDataset
    Returns
    -------
    A ListDataset mtype type to be used as input for gluonts models/estimators
    """
    import numpy as np
    import pandas as pd

    # New dependency from Gluon-ts
    from gluonts.dataset.common import ListDataset

    from sktime.datatypes import convert_to

    dimension_name = trainDF.columns
    num_dimensions = len(trainDF.columns)

    # Convert to nested_univ format
    trainDF = convert_to(trainDF, to_type="nested_univ")
    trainDF = trainDF.reset_index()
    trainDF = trainDF[dimension_name]

    # Infer frequency
    # Frequency is inferred from pd.Series's index
    # All instances must have the same freq and only fixed freq is supported
    # For a list of acceptable freq, see
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    freq = pd.infer_freq(trainDF.loc[0, dimension_name[0]].index)

    # Find start date for each instance
    start_date = [
        str(trainDF.loc[instance, dimension_name[0]].index[0].date())
        for instance, dim in trainDF.iterrows()
    ]

    if class_val_list is not None:
        feat_static_cat = class_val_list
    else:
        # If not available, class_val_list will show instance numbers
        feat_static_cat = list(np.arange(len(trainDF)))
    if num_dimensions > 1:
        one_dim_target = False
    else:
        one_dim_target = True

    all_instance_list = []
    for instance, _dim_name in trainDF.iterrows():
        one_instance_list = []
        for dim in range(num_dimensions):
            tmp = list(trainDF.loc[instance, dimension_name[dim]].to_numpy())
            one_instance_list.append(tmp)
        if one_dim_target is True:
            flatlist = [element for sublist in one_instance_list for element in sublist]
            all_instance_list.append(flatlist)
        else:
            all_instance_list.append(one_instance_list)
    train_ds = ListDataset(
        [
            {"target": target, "start": start, "fea_static_cat": [fsc]}
            for (target, start, fsc) in zip(
                all_instance_list, start_date, feat_static_cat
            )
        ],
        freq=freq,
        one_dim_target=one_dim_target,
    )
    return train_ds
