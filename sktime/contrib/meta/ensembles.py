""" HIVE-COTE and associated implementation
NOTE: this code is very much in progress and functionality is not complete. Please see main method for example usage.
Comments and finalised functionality to follow. Content is based strongly on James Larges' newer implementation in TSML
"""

__author__ = "Jason Lines"

from time import time_ns
import numpy as np
from pathlib import Path
from sktime.classifiers.shapelet_based.stc import ShapeletTransformClassifier as STC
from sktime.classifiers.frequency_based.rise import RandomIntervalSpectralForest as RISE
from sktime.classifiers.dictionary_based.boss import BOSSEnsemble as BOSS
from sktime.classifiers.interval_based.tsf import TimeSeriesForest as TSF
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, LeaveOneOut, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score
from sktime.datasets.base import load_gunpoint

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def time_ms():
    return int(time_ns()/1000000)


class BaseTSMLEnsemble:

    def __init__(
            self,
            classifiers=None,
            classifier_names=None,
            classifier_params=None,
            seed=None,
            read_individual_results=False,
            write_individual_results=False
    ):

        self.seed = seed

        self.ensemble_name = "AbstractEnsemble"
        self.weighting_scheme = None
        self.voting_scheme = None

        self.train_estimator = None

        self.train_X = None
        self.train_y = None

        self.train_results = None  # inherited from EnhancedAbstractClassifier data generated during buildclassifier if above = true
        self.test_results = None  # data generated during testing

        # saved after building so that it can be added to our test results, even if for some reason
        # we're not building/writing train results
        self.build_time = None

        # data info
        self.num_train_ins = None
        self.num_train_atts = None
        self.num_classes = None
        self.test_ins_counter = None
        self.num_test_ins = None
        self.prev_test_ins = None

        # results file handling
        self.read_individual_results = read_individual_results
        self.write_individual_results = write_individual_results
        self.results_files_parameters_initialised = False

        # multi-threading
        self.num_threads = 1

        self.read_results_files_directories = None
        self.write_results_files_directory = None
        self.dataset_name = None

        # from enhanced
        self.able_to_estimate_own_performance = False   # whether it can
        self.estimate_own_performance = False           # whether it will

        self.seed_classifier = False
        self.random = None

        self.classes_ = None

        self.contract_ = None

        if classifiers is not None:
            if classifier_names is None:
                self.classifier_names = [None]*len(classifiers)
            else:
                self.classifier_names = classifier_names
            if classifier_params is None:
                self.classifier_params = [None]*len(classifiers)
            else:
                self.classifier_params = classifier_params

            self.modules = [EnsembleModule(classifiers[i], classifier_names[i],classifier_params[i]) for i in range(len(classifiers))]

        else:
            self.setup_default_ensemble_settings()

    def setup_default_ensemble_settings(self):
        raise Exception("Cannot instantiate an AbstractEnsemble. Please use a subclass")

    def setup_ensemble_file_writing(self, output_results_path, dataset_name):
        self.write_results_files_directory = output_results_path
        self.dataset_name = dataset_name

    def get_classifiers(self):
        return [m.classifier for m in self.modules]

    def set_classifiers(self, classifiers=None, classifier_names=None, classifier_params=None):
        if classifier_names is None:
            self.classifier_names = [None]*len(classifiers)
        else:
            self.classifier_names = classifier_names

        if classifier_params is None:
            self.classifier_params = [None]*len(classifiers)
        else:
            self.classifier_params = classifier_params

        self.modules = [EnsembleModule(classifiers[i], self.classifier_names[i], self.classifier_params[i]) for i in range(len(classifiers))]

    def set_classifier_names_for_file_read(self, classifier_names):
        self.set_classifiers(classifier_names=classifier_names)

    def set_seed(self, seed):
        self.seed_classifier = True
        self.seed = seed
        self.random = np.random.RandomState(seed=seed)

    def _initialise_modules(self):
        # i.e. loading from file
        if self.read_individual_results:
            self.load_modules()
        else:
            self.train_modules()

    def train_modules(self):
        # add parallel here later
        for module in self.modules:
            if isinstance(module.classifier, EnhancedTSMLClassifier):  # can estimate own performance
                raise Exception("Not implemented yet!")
            elif (isinstance(module.classifier, STC)):

                # a bit of a hack for now - build the transform and then cv over transformed data, rather than transforming 10 times and classifying

                le = LabelEncoder()
                y = le.fit(self.train_y)

                start_time = time_ms()
                # fit st here with all train
                # module.classifier.transform.fit_transform(self.train_X, self.train_y)
                module.classifier.fit(self.train_X, self.train_y)
                build_time = time_ms()-start_time

                # transformed data for cv
                transformed_x = module.classifier.transform.transform(self.train_X)

                start_time = time_ms()

                probas = cross_val_predict(
                    # RandomForestClassifier(n_estimators=500)
                    module.classifier.pipeline.steps[-1][1],
                    transformed_x,
                    self.train_y,
                    cv=StratifiedKFold(n_splits=10, random_state=self.seed),
                    method="predict_proba"
                )
                estimate_time = time_ms()-start_time
            else:

                le = LabelEncoder()
                y = le.fit(self.train_y)

                start_time = time_ms()
                probas = cross_val_predict(
                    module.classifier,
                    self.train_X,
                    self.train_y,
                    cv=StratifiedKFold(n_splits=10, random_state=self.seed),
                    method="predict_proba")
                estimate_time = time_ms() - start_time

                start_time = time_ms()
                module.classifier.fit(self.train_X, self.train_y)
                build_time = time_ms() - start_time

            preds = le.inverse_transform([np.argmax(dist) for dist in probas])
            acc = accuracy_score(self.train_y, preds)

            # create classifier results for train
            train_results = ClassifierResults(
                classifier_name=module.module_name,
                params=module.classifier.get_params(),
                dataset_name=self.dataset_name,
                seed=self.seed,
                split="TRAIN",
                num_classes=self.num_classes,
                build_time=build_time,
                error_estimate_time=estimate_time,
                build_plus_estimate_time=build_time+estimate_time
            )
            print("dset name")
            print(train_results.dataset_name)
            train_results.true_class_values = self.train_y
            train_results.pred_class_values = preds
            train_results.pred_distributions = probas
            train_results.acc = acc

            module.train_results = train_results

            if self.write_individual_results:
                print("howdy writing")
                self.write_results_file(module.module_name,train_results,"train")

    def load_modules(self):
        for m in range(len(self.modules)):
            module = self.modules[m]
            if type(self.read_results_files_directories) == str:
                read_results_files_directory = self.read_results_files_directories
            else:
                read_results_files_directory = self.read_results_files_directories[m]

            result_path = str(read_results_files_directory)+str(module.module_name)+"/Predictions/" + str(self.dataset_name) + "/trainFold" + str(self.seed)+".csv"
            module.train_results = ClassifierResults.init_from_file(result_path)
            result_path = str(read_results_files_directory)+str(module.module_name)+"/Predictions/" + str(self.dataset_name) + "/testFold" + str(self.seed)+".csv"
            module.test_results = ClassifierResults.init_from_file(result_path)

    def need_individual_train_preds(self):
        return self.estimate_own_performance or self.weighting_scheme.need_train_preds or self.voting_scheme.need_trainPreds

    def write_results_file(self, classifier_name, results, train_or_test):

        full_path = self.write_results_files_directory + classifier_name + "/Predictions/" + self.dataset_name
        Path(full_path).mkdir(parents=True, exist_ok=True)
        full_path += "/" + train_or_test + "Fold" + str(self.seed) + ".csv"

        print("full path:")
        print(full_path)
        output_file = open(full_path, "w")
        output_file.write(results.write_full_results_to_string())
        output_file.close()

    def write_all_module_test_results_file(self):

        for module in self.modules:
            self.write_results_file(classifier_name=module.module_name, results=module.test_results, train_or_test="test")

    def write_all_test_files(self):
        self.write_results_file(self.ensemble_name, self.test_results, "test")
        self.write_all_module_test_results_file()

    def set_results_file_location_parameters(self, individual_results_files_directory, dataset_name, seed):
        self.results_files_parameters_initialised = True
        self.read_results_files_directories = [individual_results_files_directory] if type(individual_results_files_directory) == str else individual_results_files_directory
        self.dataset_name = dataset_name
        self.seed = seed

    def set_results_file_writing_location(self, writing_results_file_directory):
        self.write_results_files_directory = writing_results_file_directory

    def set_build_individuals_from_results_file(self, true_or_false):
        self.read_individual_results = true_or_false
        if true_or_false is True:
            self.write_individual_results = False

    def set_write_individuals_train_results_file(self, true_or_false):
        self.write_individual_results = true_or_false
        if true_or_false is True:
            self.read_individual_results = False

    def estimate_ensemble_performance(self, X, y, num_classes):
        train_results = ClassifierResults(num_classes=num_classes)

        # estimate_time_start = time_ns()
        estimate_time_start = time_ms()
        probas = self.voting_scheme.distributionForTrainingInstances(X)
        # estimate_time_end = time_ns()
        estimate_time_end = time_ms()

        # different from java here as we need to do whole set of instances, not one-by-one
        preds = BaseTSMLEnsemble._probas_to_preds(probas=probas)
        train_results.add_predictions(y, probas, preds, estimate_time_end-estimate_time_start)

        train_results.dataset_name = self.dataset_name
        train_results.seed = self.seed
        train_results.split = "train"
        train_results.params = self._get_param_string()
        train_results.finalise_results()
        return train_results


    def write_individual_test_files(self, test_y, throw_exception_if_params_not_set=False):
        if self.write_individual_results is False or self.results_files_parameters_initialised is False:
            if throw_exception_if_params_not_set:
                raise Exception("to call writeIndividualTestFiles(), must have called setResultsFileLocationParameters(...) and setWriteIndividualsResultsFiles()")

        self.finalise_individual_module_test_results(test_y)

        for module in self.modules:
            self._write_results_file(module.module_name, module.parameters, module.test_results, "test")

    def get_classifier_names(self):
        return [module.module_name for module in self.modules]

    def get_individual_acc_estimates(self):
        return [module.train_results.get_acc() for module in self.modules]

    def fit(self, X, y):

        # todo: can classifier handle the data? sktime checks here instead of tsml
        # // can classifier handle the data?
        #         getCapabilities().testWithFail(data);
        self.classes_ = np.unique(y)
        if self.results_files_parameters_initialised:
            if len(self.read_results_files_directories) > 1:
                if len(self.read_results_files_directories) != len(self.modules):
                    raise Exception("Ensemble, " + str(type(self)) + ".buildClassifier: "
                                    + "more than one results path given, but number given does not align with the number of classifiers/modules.")

            if self.write_results_files_directory is None:
                self.write_results_files_directory = self.read_results_files_directories[0]

        start_time = time_ms()

        # init
        self.train_X = X
        self.train_y = y

        self.num_train_ins = len(self.train_y)
        self.num_classes = len(set(self.train_y))
        self.num_train_atts = len(self.train_X.iloc[1, :])  # TODO not sure about this for num atts

        # set up modules
        self._initialise_modules()

        # if modules' results are being read in from file, ignore the i/o overhead
        # of loading the results, we'll sum the actual buildtimes of each module as
        # reported in the files
        if self.read_individual_results:
            # start_time = time_ns()
            start_time = time_ms()

        # set up ensemble
        self.weighting_scheme.define_weightings(self.modules, self.num_classes)
        self.voting_scheme.train_voting_scheme(self.modules, self.num_classes)

        # self.build_time = time_ns() - start_time
        self.build_time = time_ms() - start_time

        if self.read_individual_results:
            # we need to sum the modules' reported build time as well as the weight
            # and voting definition time
            for module in self.modules:
                self.build_time += module.train_results.get_build_time()

                # TODO see other todo in trainModules also. Currently working under
                #  assumption that the estimate time is already accounted for in the build
                #  time of TrainAccuracyEstimators, i.e. those classifiers that will
                #  estimate their own accuracy during the normal course of training

                # TODO not sure how to do this for now. Just including
                if module.is_estimating_own_performance:
                    self.build_time += module.train_results.getErrorEstimateTime()

        self.train_results = ClassifierResults()

        self.train_results.build_time = self.build_time
        self.train_results.params = self.get_param_string()

    def predict_proba(self, X, y):

        if self.test_results is None:
            self.test_results = ClassifierResults(self.num_classes)
            self.test_results.build_time = self.build_time

            self.test_results = ClassifierResults(
                classifier_name=self.ensemble_name,
                params=self.get_param_string(),
                dataset_name=self.dataset_name,
                seed=self.seed,
                split="TEST",
                num_classes=self.num_classes,
                build_time=self.build_time,
                # error_estimate_time=estimate_time,
                # build_plus_estimate_time=build_time+estimate_time
            )

        if self.read_individual_results and self.test_ins_counter >= self.num_test_ins:  # //if no test files loaded, numTestInsts == -1
            raise Exception("Received more test instances than expected, when loading test results files, found " + str(self.num_test_ins) + " test cases")

        start_time = time_ms()

        if self.read_individual_results:  # //have results loaded from file
            # not supported yet
            raise NotImplementedError()
        else: # need to classify them normally
            dist = self.voting_scheme.distribution_for_instances(self.modules, X)

            for m in range(len(self.modules)):
                # indices = module_probas.argmax(axis=1)
                # module_preds = self.classes_[indices]
                indices = self.modules[m].test_results.pred_distributions.argmax(axis=1)
                module_preds = self.classes_[indices]
                self.modules[m].test_results.pred_class_values = module_preds
                self.modules[m].test_results.true_class_values = y
                self.modules[m].test_results.acc = accuracy_score(y, self.modules[m].test_results.pred_class_values)

            pred_time = (time_ms() - start_time)

        indices = dist.argmax(axis=1)
        preds = self.classes_[indices]
        self.test_results.test_time = pred_time

        self.test_results.true_class_values = y
        self.test_results.pred_class_values = preds
        self.test_results.pred_distributions = dist
        self.test_results.acc = accuracy_score(y, preds)

        return dist

    def predict(self, X, y):
        probas = self.predict_proba(X,y)
        indices = probas.argmax(axis=1)
        return self.classes_[indices]


class EnsembleModule:

    def __init__(self, classifier, module_name=None, parameters=None):
        self.classifier = classifier
        self.module_name = module_name
        self.parameters = parameters

        self.train_results = None
        self.test_results = None
        self.prior_weight = 1
        self.posterior_weights = None

        self.is_estimating_own_performance = False


class ClassifierResults:
    def __init__(
            self,
            classifier_name=None,
            params=None,
            dataset_name=None,
            seed=None,
            split=None,
            file_type="predictions",
            description=None,
            acc=None,
            build_time=None,
            error_estimate_method = None,
            error_estimate_time = None,
            build_plus_estimate_time = None,
            test_time=None,
            benchmark_time=None,
            memory_usage=None,
            num_classes=None,

    ):
        self.classifier_name = classifier_name

        if type(params) is dict:
            self.params = self.get_params_to_string(params)
        else:
            self.params = params
        self.dataset_name = dataset_name
        self.seed = seed
        self.split = split
        self.file_type = file_type
        self.description = description
        self.acc = acc
        self.build_time = build_time
        self.test_time = test_time
        self.benchmark_time = benchmark_time
        self.memory_usage = memory_usage
        self.num_classes = num_classes

        self.error_estimate_method = error_estimate_method
        self.error_estimate_time = error_estimate_time
        self.build_plus_estimate_time = build_plus_estimate_time

        if build_plus_estimate_time is None and build_time is not None and error_estimate_time is not None:
            self.build_plus_estimate_time = build_time+error_estimate_time

        self.true_class_values = None
        self.pred_class_values = None
        self.label_encoder = None
        self.true_class_values_encoded = None
        self.pred_class_values_encoded = None
        self.pred_distributions = None
        self.pred_times = None
        self.pred_descriptions = None
        self.finalised = False

    def add_predictions(self, true_class_values, probas, preds, pred_times=None, descriptions=None):
        if self.true_class_values is None:
            self.true_class_values = true_class_values
        else:
            self.true_class_values = np.concatenate(self.true_class_values, true_class_values)

        if self.pred_distributions is None:
            self.pred_distributions = probas
        else:
            self.pred_distributions = np.concatenate(self.pred_distributions, probas)

        if self.pred_class_values is None:
            self.pred_class_values = preds
        else:
            self.pred_class_values = np.concatenate(self.pred_class_values, preds)

        if pred_times is not None:
            if self.pred_times is None:
                self.pred_times = pred_times
            else:
                self.pred_times = np.concatenate(self.pred_times, pred_times)

        if descriptions is not None:
            if self.pred_descriptions is None:
                self.pred_descriptions = descriptions
            else:
                self.pred_times = np.concatenate(self.pred_descriptions, descriptions)

    @staticmethod
    def init_from_file(path):
        raise NotImplementedError("not implemented yet")

    def finalise_results(self):
        raise NotImplementedError("Not implemented yet")

    def get_technical_information(self):
        raise NotImplementedError()

    def get_acc(self):
        return self.acc

    def finalise_results(self):
        if self.num_classes is None or self.num_classes < 1:
            self.num_classes = len(self.pred_distributions[0])
        self.acc = np.sum([1 if self.pred_class_values[i]==self.true_class_values[i] else 0 for i in range(len(self.true_class_values))])/len(self.true_class_values)

        le = LabelEncoder()
        le.fit(self.true_class_values)
        self.true_class_values_encoded = le.transform(self.true_class_values)
        self.pred_class_values_encoded = le.transform(self.pred_class_values)

    def write_full_results_to_string(self, use_class_vals_from_0=True):
        self.finalise_results()

        return self.generate_first_line() + "\n" + \
               self.generate_second_line() + "\n" + \
               self.generate_third_line() + "\n" + \
               self.instance_predictions_to_string(use_class_vals_from_0)

    def __str__(self):
        return self.write_full_results_to_string()

    def generate_first_line(self):
        return str(self.dataset_name) + "," + \
               str(self.classifier_name) + "," +\
               str(self.split) + "," + \
               str(self.seed) + "," + \
               "MILLISECONDS," + \
               str(self.file_type) + ", " + \
               str(self.description)

    def generate_second_line(self):
        if type(self.params) is dict:
            return self.get_params_to_string(self.params)
        return self.params

    def generate_third_line(self):
        return  str(self.acc) + "," +  \
                str(self.build_time) + "," + \
                str(self.test_time) + "," + \
                str(self.benchmark_time) + "," + \
                str(self.memory_usage) + "," + \
                str(self.num_classes) + "," + \
                str(self.error_estimate_method) + "," + \
                str(self.error_estimate_time) + "," + \
                str(self.build_plus_estimate_time)

    def instance_predictions_to_string(self, use_class_vals_from_0=True):

        if use_class_vals_from_0:
            actuals_to_use = self.true_class_values_encoded
            preds_to_use = self.pred_class_values_encoded
        else:
            actuals_to_use = self.true_class_values
            preds_to_use = self.pred_class_values

        output = "";
        for i in range(len(preds_to_use)):
            output += str(actuals_to_use[i]) + "," + \
                      str(preds_to_use[i])  + ","
            probs = self.pred_distributions[i]
            for p in probs:
                output += "," + str(p)

            if self.pred_times is None or len(self.pred_times) == 0:
                output += ",,"
            else:
                output += ",," + str(self.pred_times[i])

            if self.pred_descriptions is None or len(self.pred_descriptions) == 0:
                output += ",,"
            else:
                output += ",," + str(self.pred_descriptions[i])

            if i != len(preds_to_use)-1:
                output += "\n"
        return output

    def get_params_to_string(self, get_params_dict):
        output = ""

        for key, value in get_params_dict.items():
            output += str(key) + "," + str(value) + ","

        return output


class TechnicalInformation:
    def __init__(
            self,
            contribution_type,
            author,
            title,
            journal=None,
            volume=None,
            number=None,
            pages=None,
            year=None
            ):
        self.contribution_type = contribution_type
        self.author = author
        self.title = title
        self.journal = journal,
        self.volume = volume,
        self.number = number,
        self.pages = pages,
        self.year = year


class HIVE_COTE(BaseTSMLEnsemble):

    def __init__(
            self,
            classifiers=None,
            classifier_names=None,
            classifier_params=None,
            seed=None,
            read_individual_results=False,
            write_individual_results=False):
        super().__init__(
            classifiers=classifiers,
            classifier_names=classifier_names,
            classifier_params=classifier_params,
            seed=seed,
            read_individual_results=read_individual_results,
            write_individual_results=write_individual_results
        )

    def setup_default_ensemble_settings(self):
        self.ensemble_name = "HIVE-COTE"
        self.weighting_scheme = TrainingAccWeightingScheme(power=4)
        self.voting_scheme = MajorityConfidenceModuleVotingScheme()

        classifier_names =[
            "TSF",
            "RISE",
            "BOSS",
            "STC"
        ]

        tsf = TSF(random_state=self.seed)
        rise = RISE(random_state=self.seed)
        boss = BOSS(random_state=self.seed)
        stc = STC(time_contract_in_mins=240, random_state=self.seed)

        classifiers = [
            tsf,
            rise,
            boss,
            stc
        ]

        self.set_classifiers(classifiers=classifiers, classifier_names=classifier_names)

        # just in case - probably redundant as it should be random_state by convention, but those are set above
        self.set_constituent_seeds(self.seed)

    def set_constituent_seeds(self, seed):
        super().set_seed(seed)
        for module in self.modules:
            if hasattr(module, "seed"):
                module.classifier.seed = seed

    def get_param_string(self):
        output_part_one = ""
        output_part_two = ""

        for module in self.modules:
            output_part_one += str(module.posterior_weights[0]) + "," + str(module.module_name)+ ","

            param_dict = module.classifier.get_params()
            output_part_two += ","
            for key, value in param_dict.items():
                # print(key+","+param_dict[key])
                # print(str(key) + "," + str(value))
                output_part_two +=str(key) + "," + str(value) + ","

        return output_part_one + output_part_two[:-1]

    @staticmethod
    def get_technical_information():
        return TechnicalInformation(
            contribution_type="article",
            author="J. Lines, S. Taylor and A. Bagnall",
            title="Time Series Classification with HIVE-COTE: The Hierarchical Vote Collective of Transformation-Based Ensembles",
            journal="ACM Transactions on Knowledge Discovery from Data",
            volume="12",
            number="5",
            pages="52",
            year=2018)


class CAWPE:
    pass # coming soon?


class AbstractModuleWeightingScheme:

    def __init__(
            self,
            uniform_weighting=True,
            need_train_preds=True
    ):
        self.uniform_weighting = uniform_weighting
        self.need_train_preds = need_train_preds

    def define_weightings(self, modules, num_classes):
        for module in modules: # //by default, sets weights independently for each module
            module.posterior_weights = self.define_weighting(module, num_classes);
        #  some schemes may sets weights for each moduel relative to the rest, and
        #  so will need to override this method

    def define_weighting(self, train_predictions, num_classes):
        raise NotImplementedError("Please use a subclass")

    @staticmethod
    def make_uniform_weighting(weight, num_classes):
        #  prevents all weights from being set to 0 for datasets such as Fungi.
        if weight == 0:
            weight = 1

        return [weight]*num_classes

    def __str__(self):
        return str(type(self))


class TrainingAccWeightingScheme(AbstractModuleWeightingScheme):

    def __init__(self, power=1.0):
        self.uniform_weighting = True
        self.need_train_preds = False
        self.power = power

    def define_weighting(self, module, num_classes):
        return self.make_uniform_weighting(weight=np.power(module.train_results.get_acc(), self.power), num_classes=num_classes)


class AbstractModuleVotingScheme:

    def __init__(self):
        self.num_classes = None
        self.need_train_preds = False

    def train_voting_scheme(self, modules, num_classes):
        self.num_classes = num_classes

    def distribution_for_train_instance(self, modules, train_instance_index):
        raise NotImplementedError("This class cannot be instantiated. Please use a subclass")

    def distribution_for_test_instance(self, modules, test_instance_index):
        raise NotImplementedError("This class cannot be instantiated. Please use a subclass")

    def classify_train_instance(self, modules, train_instance_index):
        return np.argmax(self.distribution_for_train_instance(modules, train_instance_index))

    def classify_test_instance(self, modules, test_instance_index):
        return np.argmax(self.distribution_for_test_instance(modules, test_instance_index))

    def distribution_for_instance(self, modules, test_instance):
        raise NotImplementedError("This class cannot be instantiated. Please use a subclass")

    def classify_instance(self, modules, test_instance):
        return np.argmax(self.distribution_for_instance(modules, test_instance))

    def normalise_dist_to_1(self, dist):
        sum_of_dist = np.sum(dist)

        if sum_of_dist == 1:
            val = 1.0/len(dist)
            dist = [val]*len(dist)
        else:
            dist = [x/sum_of_dist for x in dist]

        return dist

    def distribution_for_new_instance(self, module, instance):
        start_time = time_ms()
        dist = module.classifier.predict_proba(instance)
        pred_time = time_ms() - start_time

        self.store_module_test_result(module, dist, pred_time);

        return dist;

    def store_module_test_result(self, module, dist, pred_time):

        if module.test_results is None:
            module.test_results = ClassifierResults()
            module.test_results.set_build_time(module.train_results.get_build_time())
        module.test_results.add_prediction(dist, np.argmax(dist), pred_time, "");

    def __str__(self):
        return type(self).__name__


class MajorityConfidenceModuleVotingScheme(AbstractModuleVotingScheme):

    def __int__(self, num_classes=None):
        super()
        self.num_classes = num_classes

    def train_voting_scheme(self, modules, num_classes):
        self.num_classes = num_classes

    def distribution_for_instances(self, modules, X):
        probas = np.zeros((len(X),self.num_classes))

        for m in range(len(modules)):

            if modules[m].test_results is None:
                train_results = modules[m].train_results
                modules[m].test_results = ClassifierResults(
                    classifier_name=train_results.classifier_name,
                    params=train_results.params,
                    dataset_name=train_results.dataset_name,
                    seed=train_results.seed,
                    split="TEST",
                    num_classes=train_results.num_classes
                )
                modules[m].test_results.build_time = train_results.build_time
                modules[m].test_results.error_estimate_time = train_results.error_estimate_time
                modules[m].test_results.build_plus_estimate_time = train_results.build_plus_estimate_time

            start_time = time_ms()
            module_probas = modules[m].classifier.predict_proba(X)
            pred_time = time_ms()-start_time

            modules[m].test_results.test_time = pred_time
            modules[m].test_results.pred_distributions = module_probas

            # now adjust probas by weights for ensemble where necessary
            for c in range(self.num_classes):
                module_probas[:,c] = modules[m].prior_weight * modules[m].posterior_weights[c] * module_probas[:,c]

            probas += module_probas

            probas = np.array([self.normalise_dist_to_1(proba) for proba in probas])

        return probas



class EnhancedTSMLClassifier:
    # aka EnhancedAbstractClassifier in TSML
    pass  # for now


if __name__ == "__main__":

    hive = HIVE_COTE(
        seed=0,
        write_individual_results=True
    )

    hive.setup_ensemble_file_writing(output_results_path="temp/", dataset_name="GunPoint_cropped")

    train_x, train_y = load_gunpoint(split="TRAIN", return_X_y=True)
    train_x = train_x[0:20]
    train_y = train_y[0:20]

    test_x, test_y = load_gunpoint(split="TEST", return_X_y=True)
    test_x = test_x[0:20]
    test_y = test_y[0:20]

    hive.fit(train_x, train_y)
    preds = hive.predict(test_x, test_y)

    hive.write_all_test_files()
    print(hive.modules[0].test_results)
