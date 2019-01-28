from xpandas.data_container import XSeries, XDataFrame
import pandas as pd
import numpy as np
import zipfile
import urllib.request
import os


# Please note that this code is to enable development - it may not cover all cases and elements of the design could be
# improved on later.
#
# The objective is to prove code that loads univariate and multivariate time series data that may have unequal
# lengths and missing values. Code is provided to load from local files or via the web/cached files

def load_from_tsfile_to_xdataframe(file_path, file_name, replace_missing_vals_with='NaN'):
    data_started = False
    instance_list = []
    class_val_list = []

    has_time_stamps = False
    has_class_labels = False

    uses_tuples = False

    is_first_case = True
    with open(file_path + file_name, 'r') as f:
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

    # create the output XDataFame
    # start by setting up the columns for each dimension. Explicit column names could be added here later if needed,
    # currently defaults to "dim_0" to "dim_d-1" for d dimensions
    x_data = {}
    for dim in range(0, num_dimensions):
        key = 'dim_' + str(dim)
        x_data[key] = XSeries(instance_list[dim])

    # if there were class labels, return an XDataFrame containing the XSeries we just made,
    # and an XSeries for the class values
    if has_class_labels:
        return XDataFrame(x_data), XSeries(class_val_list, name="class_val")

    # otherwise just return an XDataFrame
    return XDataFrame(x_data)


# EXAMPLE NOT USING XPANDAS:
# removes the dependency on XPandas and returns a pandas.DataFrame with m rows and d columns of pandas.Series objects
# for m cases with d dimensions
def load_from_tsfile_to_dataframe(file_path, file_name, replace_missing_vals_with='NaN'):
    data_started = False
    instance_list = []
    class_val_list = []

    has_time_stamps = False
    has_class_labels = False

    uses_tuples = False

    is_first_case = True
    with open(file_path + file_name, 'r') as f:
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
    x_data = pd.DataFrame()
    for dim in range(0, num_dimensions):
        x_data['dim_' + str(dim)] = instance_list[dim]

    if has_class_labels:
        return x_data, class_val_list
    #
    # # otherwise just return an XDataFrame
    return x_data


# utility method - if the dataset exists, download it from timeseriesclassification.com and cache locally (so
# subsequent calls to this method do not each download the data). Can specify if it should load a predefined
# train (dataset_TRAIN.ts) or test (dataset_TEST.ts) partition - default is to load full data (dataset.ts)
# Argument required for where to save the data - defaults to C:/temp/sktime_temp_data/" and will save in a further
# subdir (e.g. c:/temp/sktime_temp_data/datasetName/dataset.ts)
def load_from_web_to_xdataframe(dataset_name, is_train_file=False, is_test_file=False, cache_path="C:/temp/sktime_temp_data/"):
    if is_train_file is True and is_test_file is True:
        raise ValueError("is_train_file and is_test_file should not both be True")

    suffix = ".ts"
    if is_train_file:
        suffix = "_TRAIN.ts"
    elif is_test_file:
        suffix = "_TEST.ts"

    url = "http://timeseriesclassification.com/Turing/NewFormat/" + dataset_name + "/" + dataset_name + suffix
    # cache_path = "C:/temp/sktime_temp_data/"

    if not os.path.isfile(cache_path + "/" + dataset_name + "/" + dataset_name + suffix):
        print("If you see this every time then this is going to DDoS Tony's website...")
        if not os.path.exists(cache_path + dataset_name + "/"):
            os.makedirs(cache_path + dataset_name + "/")
        if not os.path.exists(cache_path + dataset_name + "/" + dataset_name + suffix):
            urllib.request.urlretrieve(url, cache_path + dataset_name + "/" + dataset_name + suffix)

    return load_from_tsfile_to_xdataframe(cache_path + dataset_name + "/", dataset_name + suffix)


# Due to discussion in the ticket on Git functionality has also been added to load from any delimited file into "long
# format". This results in a 4 column pandas.DataFrame with [case_id, dimension_id, reading_id, value]. This can then
# be converted into a xpandas.XDataFrame through use of the later method (long_format_to_wide_format).
# Requires two delimiters - one for readings and one for dimensions. Defaults to "," for readings and ":" for
# dimensions.
# Functionality added to allow a class value to be specified as the last value (and it is recorded in the table as
# dim_id="c" for convenience. This may later be handled better by a Task, but it is up for debate as the class value
# of a case is not part of a time series so arguably should NOT be stored as part of a series at any point (because
# it can never be part of a series) so perhaps it is best to differentiate on loading (unlike forecasting problems which
# could legitimately change the target within time series data
def load_from_file_to_long_format(file_path_and_name, reading_delimiter=",", dimension_delimiter=":", last_dim_is_class_val=False):
    case_id = 0
    has_class = last_dim_is_class_val
    to_frame = []  # a list to store rows, will return by converting to pandas.DataFrame at the end
    first_case = True
    num_expected_dims = -1
    with open(file_path_and_name, 'r') as f:
        # for each line
        for line in f:
            if line.strip():  # skip if the line is empty

                if "@" in line or "%" in line or "#" in line:  # skip some of the usual comment characters
                    continue

                dimensions = line.split(dimension_delimiter)
                num_dims = len(dimensions)
                if has_class:
                    num_dims -= 1

                if first_case:
                    num_expected_dims = num_dims
                    first_case = False
                else:
                    if num_expected_dims != num_dims:
                        raise Exception("inconsistent number of dimensions")

                # for each delimited dimension
                for dim in range(0, num_dims):
                    data_series = dimensions[dim].split(reading_delimiter)
                    for reading_id in range(0, len(data_series)):
                        # try to convert each reading into a new entry for the output
                        try:
                            to_frame.append([case_id, dim, reading_id, float(data_series[reading_id])])
                        except ValueError:
                            # missing values could be included as ? - try to catch. If this is the case, we can just
                            # skip adding it to the output. Otherwise, repeat the command to raise the true Exception
                            if data_series[reading_id].strip() == '?':
                                pass  # no problem, ? explicitly indicates a missing reading
                            else:  # something else is wrong, force the original Exception
                                to_frame.append([case_id, dim, reading_id, float(data_series[reading_id])])
                if has_class:
                    to_frame.append([case_id, 'c', 'c', dimensions[num_dims].strip()])
                case_id += 1

    return pd.DataFrame(to_frame, columns=['case_id', 'dimension_id', 'reading_id', 'value'])


def long_format_to_wide_format_with_last_reading_as_class(input_long_table):
    # get unique case ids
    case_ids = input_long_table.case_id.unique()
    last_readings = {}

    # get idx of last reading for each case
    for c_id in case_ids:
        last_readings[c_id] = input_long_table.loc[input_long_table['case_id'] == c_id].reading_id.max()
        temp = input_long_table.loc[(input_long_table['case_id'] == c_id) & (input_long_table['reading_id'] == last_readings[c_id])]
        input_long_table.at[temp.index[0], 'dimension_id'] = -1
        input_long_table.at[temp.index[0], 'reading_id'] = -1

    return long_format_to_wide_format(input_long_table, class_dimension_name=-1)


# utility method - to demonstrate loading data to the long format, this method gets a Weka stle ARFF from
# timeseriesclassification.com. Once the header information is ignored, it is effectively a 1-d .csv.
def load_from_web_to_long_format(dataset_name, is_train_file=False, is_test_file=False, cache_path="C:/temp/sktime_temp_data/"):
    if is_train_file is True and is_test_file is True:
        raise ValueError("is_train_file and is_test_file should not both be True")

    url = "http://timeseriesclassification.com/Downloads/" + dataset_name + ".zip"

    if not os.path.isfile(cache_path + "/" + dataset_name + "/" + dataset_name + "_TRAIN.arff"):
        print("If you see this every time then this is going to DDoS Tony's website...")
        if not os.path.exists(cache_path + dataset_name + "/"):
            os.makedirs(cache_path + dataset_name + "/")
        if not os.path.exists("C:/temp/sktime_temp_data/" + dataset_name + "/" + dataset_name + ".zip"):
            urllib.request.urlretrieve(url, "C:/temp/sktime_temp_data/" + dataset_name + "/" + dataset_name + ".zip")
        with zipfile.ZipFile("C:/temp/sktime_temp_data/" + dataset_name + "/" + dataset_name + ".zip", "r") as zip_ref:
            zip_ref.extractall("C:/temp/sktime_temp_data/" + dataset_name + "/")
    suffix = ".arff"
    if is_train_file:
        suffix = "_TRAIN.arff"
    elif is_test_file:
        suffix = "_TEST.arff"
    return load_from_file_to_long_format(cache_path + "/" + dataset_name + "/" + dataset_name + suffix)


# convert a long pandas.DataFrame to a wide xpandas.XDataFrame. Not necessarily very efficient and it is more desirable
# to load directly from a .ts file. However, this utility function allows converting from a more standard pandas format
# to the required XPandas format.
#
# Assumes that column names are ['case_id', 'dimension_id', 'reading_id', 'value'] and functionality is added to
# extract a class value from the data and return in a separate XSeries object (consistent with the above functionality).
# To do this correctly, class values much belong to a specific dimension (default in the load from file utility method
# above is "c"). No default is given in this header however to not assume a TSC problem. As above, this could be better
# handled by loading and then using a Task object, but it is also not notionally correct to ever consider the class
# value as series data by embedding it in the data. Up for debate at a later date.

# TO-DO: support timestamps (require data to develop this with first, however!)
def long_format_to_wide_format(long_dataframe, class_dimension_name=None):
    # get distinct dimension ids
    unique_dim_ids = long_dataframe.dimension_id.unique()
    num_dims = len(unique_dim_ids)

    has_class_vals = False
    # if class dimension name has been set:
    if class_dimension_name is not None:
        has_class_vals = True
        num_dims -= 1

    data_by_dim = []
    indices = []

    # get number of distinct cases (note: a case may have 1 or many dimensions)
    unique_case_ids = long_dataframe.case_id.unique()
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

    if has_class_vals:
        class_vals = [None] * len(unique_case_ids)

    # go through every row in the dataframe
    for i in range(0, len(long_dataframe)):
        # extract the relevant data, catch cases where the dim id is not an int as it must be the class

        row = long_dataframe.iloc[i]
        case_id = int(row[0])
        try:
            dim_id = int(row[1])
            # if class dim name is numeric, handle it here
            if has_class_vals and row[1] == class_dimension_name and row[2] == class_dimension_name:
                class_vals[case_id] = str(row[3])
                continue
        except ValueError:
            # if class dim name is non-numeric, handle it here
            if has_class_vals and row[1] == class_dimension_name and row[2] == class_dimension_name:
                class_vals[case_id] = str(row[3])
                continue

        reading_id = int(row[2])  # TO-DO: support timestamps
        value = row[3]
        data_by_dim[dim_id][case_id].append(value)
        indices[dim_id][case_id].append(reading_id)

    # if there were any class values, make sure all cases have a known label. Throw Exception if not
    if has_class_vals:
        for i in range(0, len(unique_case_ids)):
            if class_vals[i] is None:
                raise ValueError("Class value for case" + i + "is None. Class values should be set for all cases")

    # at end, create a bunch of Series objects, stuff them into a xdataframe, and Bob's your teapot
    x_data = {}
    for d in range(0, num_dims):
        key = 'dim_' + str(d)
        dim_list = []
        for i in range(0, len(unique_case_ids)):
            temp = pd.Series(data_by_dim[d][i], indices[d][i])
            dim_list.append(temp)
        x_data[key] = XSeries(dim_list)

    if has_class_vals:
        return XDataFrame(x_data), XSeries(class_vals)

    return XDataFrame(x_data)


if __name__ == "__main__":
    # example of loading into a pandas (NOT XPANDAS) dataframe
    cache_path = "C:/temp/sktime_temp_data/"
    dataset_name = "Gunpoint"
    suffix = "_TRAIN.ts"
    train_x, train_y = load_from_tsfile_to_dataframe(cache_path + dataset_name + "/", dataset_name + suffix)
    print(train_x)
