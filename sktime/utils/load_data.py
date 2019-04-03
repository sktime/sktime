import pandas as pd
import numpy as np
import os

def load_from_tsfile_to_dataframe(full_file_path_and_name, replace_missing_vals_with='NaN'):
    data_started = False
    instance_list = []
    class_val_list = []

    has_time_stamps = False
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
        return x_data, np.asarray(class_val_list)
    #
    # # otherwise just return an XDataFrame
    return x_data


def load_from_arff_to_tsfile(full_file_path_and_name, file_path_and_name_to_save):
    #TODO: Implement timeStamps feature in case there are problems with these characteristics.
    #TODO: Implement in case datasets without labels.
    #TODO: Implement in case multivariate datasets.
    
    problemName = full_file_path_and_name.split("/")[-1].split("_")[0]
    
    # At this moment there are not problems with timeStamps.
    timeStamps = False
    
    patterns = []
    labels = []
    unequal_length_time_series = False
    with open(full_file_path_and_name, 'r') as f:
        for line in f:
            if not line.startswith("@") and not line.startswith("%") and not line == '\n':
                data = line.split(",")
                cadena = ''
                for i,j in enumerate(data):
                    if i == len(data)-1: # If it is the label
                        if j != '?\n': # Label
                            cadena = cadena + ':' + str(int(j))
                            labels.append(int(j))
                        else: # Missing label
                            cadena += ':?'
                    else: #If it isn't the label
                        if j != '?': # Value
                            cadena += str(np.round(float(j), 5))
                        else: # Missing data
                            if not (len(np.unique(data[i:-1])) == 1): # Working with unequal length time series
                                cadena += '?'
                        if not unequal_length_time_series and not ((i+2) == len(data)):
                            cadena += ','
                        elif unequal_length_time_series and not ((i+2) == len(data)) and not (len(np.unique(data[i:-1])) == 1) and not (len(np.unique(data[i+1:-1])) == 1): # If not last value of the pattern, i.e. not the label.
                            cadena += ','
                patterns.append(cadena)
    f.close()
    classLabel = np.unique(labels)
    
    directory = "/".join(file_path_and_name_to_save.split('/')[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(file_path_and_name_to_save, 'w+') as f:
        f.write("@problemName " + problemName + "\n@timeStamps " + str.lower(str(timeStamps)) + "\n@classLabel true " + str(classLabel).replace("[", "").replace("]", "") + "\n@data\n")
        for line in patterns:
            f.write(line)
            f.write('\n')
    f.close()