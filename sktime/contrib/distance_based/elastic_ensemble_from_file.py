# -*- coding: utf-8 -*-
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sktime.contrib.classification_experiments import write_results_to_uea_format


class ElasticEnsemblePostProcess:
    __author__ = "Jason Lines"

    def __init__(
        self,
        results_path,
        dataset_name,
        distance_measures="all",
        resample_id=0,
        alpha=1,
    ):
        """

        Args:
            results_path: String - path to folder storing the results to be read into the ensemble
            dataset_name: String - the name of the dataset that this ensemble will post process results for
            distance_measures: 'all' or list of Strings - default 'all'. 'all' sets classifier to use all default constituent classifiers,
                                else a list is provided of classifiers to include. Note these names must match the names of
                                the subdirs in the folder located at results_parh
            resample_id: default = 0 - to identify the deterministic seed used for resampling experiments.
                         A resampled_id of 0 demonstrates default train/test split were used to create results
            alpha: float/double - default=1.0. Used to exponentiate the confidence of constituent classifiers when making test predictions
        """
        self.results_path = results_path
        self.dataset_name = dataset_name
        self.resample_id = resample_id
        if distance_measures == "all":
            self.distance_measures = [
                "dtw",
                "ddtw",
                "wdtw",
                "wddtw",
                "lcss",
                "erp",
                "msm",
            ]
        else:
            self.distance_measures = distance_measures
        self.alpha = alpha

        # load in train information
        self.train_accs_by_classifier = np.zeros(len(self.distance_measures))
        self.train_dists_by_classifier = []
        self.test_dists_by_classifier = []
        self.ee_train_dists = None
        self.ee_test_dists = None

        self.actual_train_class_vals = None
        self.actual_test_class_vals = None
        self.classes_ = None

        num_classes = None
        num_ins = None
        class_vals = None
        # load train

        for c_id in range(len(self.distance_measures)):
            file_path = (
                self.results_path
                + self.distance_measures[c_id]
                + "/Predictions/"
                + self.dataset_name
                + "/trainFold"
                + str(self.resample_id)
                + ".csv"
            )
            with open(file_path, "r") as f:
                lines = f.readlines()
                third_line = lines[2].split(",")
                self.train_accs_by_classifier[c_id] = float(third_line[0].strip())
                this_class_vals = (
                    third_line[-1].strip().replace("[", "").replace("]", "").split(" ")
                )
                this_num_classes = len(this_class_vals)
                this_num_ins = len(lines) - 3
                if class_vals is None:
                    class_vals = this_class_vals
                    self.classes_ = np.array(this_class_vals)
                    num_classes = this_num_classes
                    num_ins = this_num_ins

                elif this_class_vals != class_vals:
                    raise ValueError(
                        "Class value mismatch when loading train file for "
                        + str(self.distance_measures[c_id])
                        + " and "
                        + self.dataset_name
                    )
                elif this_num_ins != num_ins:
                    raise ValueError(
                        "Inconsistent number of predictions in constituent training files: first spotted "
                        "in train file for "
                        + str(self.distance_measures[c_id])
                        + " and "
                        + self.dataset_name
                    )

                this_dists = np.empty((num_ins, num_classes))
                this_actual_train_class_vals = []
                for i in range(num_ins):
                    split_line = lines[i + 3].strip().split(",")
                    this_actual_train_class_vals.append(split_line[0].strip())
                    for c in range(num_classes):
                        this_dists[i][c] = (
                            np.power(float(split_line[c + 3]), self.alpha)
                            * self.train_accs_by_classifier[c_id]
                        )
                if self.actual_train_class_vals is None:
                    self.actual_train_class_vals = this_actual_train_class_vals
                elif self.actual_train_class_vals != this_actual_train_class_vals:
                    raise ValueError(
                        "Class values in files no not match for train - first spotted for "
                        + str(self.distance_measures[c_id])
                    )

            if self.ee_train_dists is None:
                self.ee_train_dists = this_dists
            else:
                self.ee_train_dists = np.add(self.ee_train_dists, this_dists)
            self.train_dists_by_classifier.append(this_dists)
        self.ee_train_dists = np.divide(
            self.ee_train_dists, sum(self.train_accs_by_classifier)
        )

        # load test
        num_test_ins = None
        for c_id in range(len(self.distance_measures)):
            file_path = (
                self.results_path
                + self.distance_measures[c_id]
                + "/Predictions/"
                + self.dataset_name
                + "/testFold"
                + str(self.resample_id)
                + ".csv"
            )
            with open(file_path, "r") as f:
                lines = f.readlines()
                third_line = lines[2].split(",")
                this_class_vals = (
                    third_line[-1].strip().replace("[", "").replace("]", "").split(" ")
                )
                this_num_ins = len(lines) - 3
                if this_class_vals != class_vals:
                    raise ValueError(
                        "Class value mismatch when loading test file for "
                        + str(self.distance_measures[c_id])
                        + " and "
                        + self.dataset_name
                    )
                if num_test_ins is None:
                    num_test_ins = this_num_ins
                elif num_test_ins != this_num_ins:
                    raise ValueError(
                        "Inconsistent number of predictions in constituent test files: first spotted "
                        "in train file for "
                        + str(self.distance_measures[c_id])
                        + " and "
                        + self.dataset_name
                    )
                this_dists = np.empty((num_ins, num_classes))
                this_actual_test_class_vals = []
                for i in range(num_ins):
                    split_line = lines[i + 3].strip().split(",")
                    this_actual_test_class_vals.append(split_line[0].strip())
                    for c in range(num_classes):
                        this_dists[i][c] = (
                            np.power(float(split_line[c + 3]), self.alpha)
                            * self.train_accs_by_classifier[c_id]
                        )
                if self.actual_test_class_vals is None:
                    self.actual_test_class_vals = this_actual_test_class_vals
                elif self.actual_test_class_vals != this_actual_test_class_vals:
                    raise ValueError(
                        "Class values in files no not match for test - first spotted for "
                        + str(self.distance_measures[c_id])
                    )

            if self.ee_test_dists is None:
                self.ee_test_dists = this_dists
            else:
                self.ee_test_dists = np.add(self.ee_test_dists, this_dists)
            self.test_dists_by_classifier.append(this_dists)

        self.ee_test_dists = np.divide(
            self.ee_test_dists, sum(self.train_accs_by_classifier)
        )

    def write_files(
        self,
        output_results_path,
        output_classifier_name="EE",
        write_train=True,
        write_test=True,
        overwrite=False,
    ):
        """

        Args:
            output_results_path: String - path to where output results will be written
            output_classifier_name: String - the name of the composite ensemble classifier in the output files
            write_train: boolean - true will write train files for the ensemble, false will skip training files
            write_test: boolean - true will write test files for the ensemble, false will skip test files
            overwrite: boolean - if true, any existing train/test files will be over-written. False prevents file overwriting


        """
        if write_train is False and write_test is False:
            print(
                "Train and test writing both set to false - method will terminate without doing anything"
            )
            return

        if not overwrite:
            if write_train:
                full_path = (
                    str(output_results_path)
                    + "/"
                    + str(output_classifier_name)
                    + "/Predictions/"
                    + str(self.dataset_name)
                    + "/trainFold"
                    + str(self.resample_id)
                    + ".csv"
                )
                if os.path.exists(full_path):
                    print(
                        full_path
                        + " already exists and overwrite set to false, not writing Train",
                        Warning,
                    )
                    write_train = False

            if write_test is True:
                full_path = (
                    str(output_results_path)
                    + "/"
                    + str(output_classifier_name)
                    + "/Predictions/"
                    + str(self.dataset_name)
                    + "/testFold"
                    + str(self.resample_id)
                    + ".csv"
                )
                if os.path.exists(full_path):
                    print(
                        full_path
                        + " already exists and overwrite set to false, not writing Test"
                    )
                    write_test = False

        if write_train is False and write_test is False:
            print(
                "Train and test files both already exist and overwrite set to false - method will terminate without doing anything"
            )
            return

        """
        file_format = None
        if os.path.exists(problem_path + self.dataset_name + '/' + self.dataset_name+ '_TRAIN.ts') and os.path.exists(problem_path + self.dataset_name + '/' + self.dataset_name+ '_TEST.ts'):
            train_x, train_y = loader.load_from_tsfile_to_dataframe(problem_path + self.dataset_name + '/' + self.dataset_name + '_TRAIN.ts')
            test_x, test_y = loader.load_from_tsfile_to_dataframe(problem_path + self.dataset_name + '/' + self.dataset_name + '_TEST.ts')
        elif os.path.exists(problem_path + self.dataset_name + '/' + self.dataset_name+ '_TRAIN.arff') and os.path.exists(problem_path + self.dataset_name + '/' + self.dataset_name+ '_TEST.arff'):
            train_x, train_y = loader.load_from_arff_to_dataframe(problem_path + self.dataset_name + '/' + self.dataset_name + '_TRAIN.ts')
            test_x, test_y = loader.load_from_arff_to_dataframe(problem_path + self.dataset_name + '/' + self.dataset_name + '_TEST.ts')
        elif os.path.exists(problem_path + self.dataset_name + '/' + self.dataset_name + '_TRAIN.tsv') and os.path.exists(problem_path + self.dataset_name + '/' + self.dataset_name + '_TEST.tsv'):
            train_x, train_y = loader.load_from_ucr_tsv_to_dataframe(problem_path + self.dataset_name + '/' + self.dataset_name + '_TRAIN.ts')
            test_x, test_y = loader.load_from_ucr_tsv_to_dataframe(problem_path + self.dataset_name + '/' + self.dataset_name + '_TEST.ts')
        else:
            raise ValueError("No dataset found for "+self.dataset_name)
        """
        if write_train:
            train_probs = self.ee_train_dists
            train_preds = self.classes_[np.argmax(train_probs, axis=1)]
            acc = accuracy_score(self.actual_train_class_vals, train_preds)
            second = str(self.distance_measures)
            third = (
                str(acc)
                + ",NA,NA,-1,-1,"
                + str(len(self.classes_))
                + ","
                + str(self.classes_)
            )
            write_results_to_uea_format(
                second_line=second,
                third_line=third,
                output_path=output_results_path,
                classifier_name=output_classifier_name,
                resample_seed=self.resample_id,
                predicted_class_vals=train_preds,
                actual_probas=train_probs,
                dataset_name=self.dataset_name,
                actual_class_vals=self.actual_train_class_vals,
                split="TRAIN",
            )

        if write_test:
            test_probs = self.ee_test_dists
            test_preds = self.classes_[np.argmax(test_probs, axis=1)]
            acc = accuracy_score(self.actual_test_class_vals, test_preds)
            second = str(self.distance_measures)
            third = (
                str(acc)
                + ",NA,NA,-1,-1,"
                + str(len(self.classes_))
                + ","
                + str(self.classes_)
            )
            write_results_to_uea_format(
                second_line=second,
                third_line=third,
                output_path=output_results_path,
                classifier_name=output_classifier_name,
                resample_seed=self.resample_id,
                predicted_class_vals=test_preds,
                actual_probas=test_probs,
                dataset_name=self.dataset_name,
                actual_class_vals=self.actual_test_class_vals,
                split="TEST",
            )
