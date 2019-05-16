import pandas as pd
import numpy as np


def load_from_tsfile_to_dataframe(full_file_path_and_name, return_separate_X_and_y=True, replace_missing_vals_with='NaN'):
    data_started = False
    instance_list = []
    class_val_list = []

    has_time_stamps = False     # not supported yet
    has_class_labels = False

    uses_tuples = False

    is_first_case = True
    with open(full_file_path_and_name, 'r') as f:
        for line in f:

            if line.strip():
                if "@timestamps" in line.lower():
                    if "true" in line.lower():
                        has_time_stamps = True
                        raise Exception("Not suppoorted yet")  # we don't have any data formatted to test with yet
                    elif "false" in line.lower():
                        has_time_stamps = False
                    else:
                        raise Exception("invalid timestamp argument")

                if "@classlabel" in line.lower():
                    if "true" in line:
                        has_class_labels = True
                    elif "false" in line:
                        has_class_labels = False
                    else:
                        raise Exception("invalid classLabel argument")

                if "@data" in line.lower():
                    data_started = True
                    continue

                # if the 'data tag has been found, the header information has been cleared and now data can be loaded
                if data_started:
                    line = line.replace("?", replace_missing_vals_with)
                    dimensions = line.split(":")

                    # perhaps not the best way to do this, but on the first row, initialise stored depending on the
                    # number of dimensions that are present and determine whether data is stored in a list or tuples
                    if is_first_case:
                        num_dimensions = len(dimensions)
                        if has_class_labels:
                            num_dimensions -= 1
                        is_first_case = False
                        for dim in range(0, num_dimensions):
                            instance_list.append([])
                        if dimensions[0].startswith("("):
                            uses_tuples = True

                    this_num_dimensions = len(dimensions)
                    if has_class_labels:
                        this_num_dimensions -= 1

                    # assuming all dimensions are included for all series, even if they are empty. If this is not true
                    # it could lead to confusing dimension indices (e.g. if a case only has dimensions 0 and 2 in the
                    # file, dimension 1 should be represented, even if empty, to make sure 2 doesn't get labelled as 1)
                    if this_num_dimensions != num_dimensions:
                        raise Exception("inconsistent number of dimensions")

                    # go through each dimension that is represented in the file
                    for dim in range(0, num_dimensions):

                        # handle whether tuples or list here
                        if uses_tuples:
                            without_brackets = dimensions[dim].replace("(", "").replace(")", "").split(",")
                            without_brackets = [float(i) for i in without_brackets]

                            indices = []
                            data = []
                            i = 0
                            while i < len(without_brackets):
                                indices.append(int(without_brackets[i]))
                                data.append(without_brackets[i + 1])
                                i += 2

                            instance_list[dim].append(pd.Series(data, indices))
                        else:
                            # if the data is expressed in list form, just read into a pandas.Series
                            data_series = dimensions[dim].split(",")
                            data_series = [float(i) for i in data_series]
                            instance_list[dim].append(pd.Series(data_series))

                    if has_class_labels:
                        class_val_list.append(dimensions[num_dimensions].strip())

    # note: creating a pandas.DataFrame here, NOT an xpandas.xdataframe
    x_data = pd.DataFrame(dtype=np.float32)
    for dim in range(0, num_dimensions):
        x_data['dim_' + str(dim)] = instance_list[dim]

    if has_class_labels:
        if return_separate_X_and_y:
            return x_data, np.asarray(class_val_list)
        else:
            x_data['class_vals'] = pd.Series(class_val_list)

    return x_data


def load_from_arff_to_dataframe(full_file_path_and_name, has_class_labels=True, return_separate_X_and_y=True, replace_missing_vals_with='NaN'):

    instance_list = []
    class_val_list = []

    data_started = False
    is_multi_variate = False
    is_first_case = True

    with open(full_file_path_and_name, 'r') as f:
        for line in f:

            if line.strip():
                if is_multi_variate is False and "@attribute" in line.lower() and "relational" in line.lower():
                    is_multi_variate = True

                if "@data" in line.lower():
                    data_started = True
                    continue

                # if the 'data tag has been found, the header information has been cleared and now data can be loaded
                if data_started:
                    line = line.replace("?", replace_missing_vals_with)

                    if is_multi_variate:
                        if has_class_labels:
                            line, class_val = line.split("',")
                            class_val_list.append(class_val.strip())
                        dimensions = line.split("\\n")
                        dimensions[0] = dimensions[0].replace("'", "")

                        if is_first_case:
                            for d in range(len(dimensions)):
                                instance_list.append([])
                            is_first_case = False

                        for dim in range(len(dimensions)):
                            instance_list[dim].append(pd.Series([float(i) for i in dimensions[dim].split(",")]))

                    else:
                        if is_first_case:
                            instance_list.append([])
                            is_first_case = False

                        line_parts = line.split(",")
                        if has_class_labels:
                            instance_list[0].append(pd.Series([float(i) for i in line_parts[:len(line_parts)-1]]))
                            class_val_list.append(line_parts[-1].strip())
                        else:
                            instance_list[0].append(pd.Series([float(i) for i in line_parts[:len(line_parts)]]))

    x_data = pd.DataFrame(dtype=np.float32)
    for dim in range(len(instance_list)):
        x_data['dim_' + str(dim)] = instance_list[dim]

    if has_class_labels:
        if return_separate_X_and_y:
            return x_data, np.asarray(class_val_list)
        else:
            x_data['class_vals'] = pd.Series(class_val_list)

    return x_data


def load_from_ucr_tsv_to_dataframe(full_file_path_and_name, return_separate_X_and_y=True):
    df = pd.read_csv(full_file_path_and_name, sep="\t", header=-1)
    y = df.pop(0).values
    X = pd.DataFrame()
    X['dim_0'] = [pd.Series(df.iloc[x, :]) for x in range(len(df))]
    if return_separate_X_and_y is True:
        return X, y
    X['class_val'] = y
    return X

# assumes data is in a long table format with the following structure:
#      | case_id | dim_id | reading_id | value
# ------------------------------------------------
#   0  |   int   |  int   |    int     | double
#   1  |   int   |  int   |    int     | double
#   2  |   int   |  int   |    int     | double
#   3  |   int   |  int   |    int     | double
def from_long_to_nested(long_dataframe):

    # get distinct dimension ids
    unique_dim_ids = long_dataframe.iloc[:, 1].unique()
    num_dims = len(unique_dim_ids)

    data_by_dim = []
    indices = []

    # get number of distinct cases (note: a case may have 1 or many dimensions)
    unique_case_ids = long_dataframe.iloc[:, 0].unique()
    # assume series are indexed from 0 to m-1 (can map to non-linear indices later if needed)

    # init a list of size m for each d - to store the series data for m cases over d dimensions
    # also, data may not be in order in long format so store index data for aligning output later
    # (i.e. two stores required: one for reading id/timestamp and one for value)
    for d in range(0, num_dims):
        data_by_dim.append([])
        indices.append([])
        for c in range(0, len(unique_case_ids)):
            data_by_dim[d].append([])
            indices[d].append([])

    # go through every row in the dataframe
    for i in range(0, len(long_dataframe)):
        # extract the relevant data, catch cases where the dim id is not an int as it must be the class

        row = long_dataframe.iloc[i]
        case_id = int(row[0])
        dim_id = int(row[1])
        reading_id = int(row[2])
        value = row[3]
        data_by_dim[dim_id][case_id].append(value)
        indices[dim_id][case_id].append(reading_id)

    x_data = {}
    for d in range(0, num_dims):
        key = 'dim_' + str(d)
        dim_list = []
        for i in range(0, len(unique_case_ids)):
            temp = pd.Series(data_by_dim[d][i], indices[d][i])
            dim_list.append(temp)
        x_data[key] = pd.Series(dim_list)

    return pd.DataFrame(x_data)


# left here for now, better elsewhere later perhaps
def generate_example_long_table(num_cases=50, series_len=20, num_dims=2):

    rows_per_case = series_len*num_dims
    total_rows = num_cases*series_len*num_dims

    case_ids = np.empty(total_rows, dtype=np.int)
    idxs = np.empty(total_rows, dtype=np.int)
    dims = np.empty(total_rows, dtype=np.int)
    vals = np.random.rand(total_rows)

    for i in range(total_rows):
        case_ids[i] = int(i/rows_per_case)
        rem = i % rows_per_case
        dims[i] = int(rem/series_len)
        idxs[i] = rem % series_len

    df = pd.DataFrame()
    df['case_id'] = pd.Series(case_ids)
    df['dim_id'] = pd.Series(dims)
    df['reading_id'] = pd.Series(idxs)
    df['value'] = pd.Series(vals)
    return df
