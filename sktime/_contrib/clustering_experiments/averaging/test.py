from sktime.distances.tests._utils import create_test_distance_numpy
import numpy as np
import csv

if __name__ == '__main__':
    univariate_exampels = 5
    temp_uni = []
    for i in range(univariate_exampels):
        uni = create_test_distance_numpy(n_instance=2, n_columns=1, n_timepoints=10)
        temp_uni.append(uni[0])
        temp_uni.append(uni[1])

    temp_multi = []
    multivariate_examples = 5
    for i in range(multivariate_examples):
        uni = create_test_distance_numpy(n_instance=2, n_columns=10, n_timepoints=10)
        temp_multi.append(uni[0])
        temp_multi.append(uni[1])

    temp_uni =np.asarray(temp_uni)
    temp_multi = np.asarray(temp_multi)

    with open('univariate_examples.csv', 'w+') as csv_file:
        writer = csv.writer(csv_file)
        # for row in temp_uni:
        writer.writerow(list(temp_uni))

    with open('multivariate_examples.csv', 'w+') as csv_file:
        writer = csv.writer(csv_file)
        # for row in temp_multi:
        writer.writerow(list(temp_multi))

