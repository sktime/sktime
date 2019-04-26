import os
import sys
import sktime.contrib.hivecote.frequency_based.rise as fb
import sktime.contrib.hivecote.interval_based.tsf as ib
import sktime.contrib.hivecote.dictionary_based.boss_ensemble as db
import sktime.classifiers.ensemble as ensemble
from sktime.transformers.compose import RowwiseTransformer
from sktime.transformers.compose import Tabulariser
from sktime.transformers.series_to_series import RandomIntervalSegmenter
from sktime.pipeline import TSPipeline
from sktime.pipeline import TSFeatureUnion
from sktime.classifiers.ensemble import TimeSeriesForestClassifier
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.ar_model import AR
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#import sktime.contrib.hivecote.distance_based.elastic_ensemble as dist
import numpy as np
import time
from sklearn.metrics import accuracy_score as acc
from sktime.utils.load_data import load_from_tsfile_to_dataframe as load_ts

__author__ = "Anthony Bagnall"

""" Prototype mechanism for testing classifiers on the UCR format. This mirrors the mechanism use in Java, 
https://github.com/TonyBagnall/uea-tsc/tree/master/src/main/java/experiments
but is not yet as engineered. However, if you generate results using the method recommended here, they can be directly
and automatically compared to the results generated in java

"""


datasets = [
    "Adiac",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "Car",
    "CBF",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxTW",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
#    "ElectricDevices",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
#    "FordA",
#    "FordB",
    "GunPoint",
    "Ham",
#    "HandOutlines",
    "Haptics",
    "Herring",
    "InlineSkate",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxTW",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "Plane",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "ScreenType",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
#    "StarlightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "UWaveGestureLibraryAll",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",
]


def set_classifier(cls):
    """
    Basic way of determining the classifier to build. To differentiate settings just and another elif. So, for example, if
    you wanted tuned TSF, you just pass TuneTSF and set up the tuning mechanism in the elif.
    This may well get superceded, it is just how e have always done it
    :param cls: String indicating which classifier you want
    :return: A classifier.

    """
    if cls == 'RISE' or cls == 'rise':
        return fb.RandomIntervalSpectralForest()
    elif  cls == 'TSF' or cls == 'tsf':
        return ib.TimeSeriesForest()
    elif  cls == 'BOSS' or cls == 'boss':
        return db.BOSSEnsemble()
#    elif classifier == 'EE' or classifier == 'ElasticEnsemble':
#        return dist.ElasticEnsemble()
    elif cls == 'TSF_Markus':
        return ensemble.TimeSeriesForestClassifier()
    else:
        return 'UNKNOWN CLASSIFIER'


def run_experiment(problem_path, results_path, cls_name, dataset, resampleID=0, overwrite=False, format=".ts", train_files=False):
    """
    Method to run a basic experiment and write the results to files called testFold<resampleID>.csv and, if required,
    trainFold<resampleID>.csv.
    :param problem_path: Location of problem files, full path.
    :param results_path: Location of where to write results. Any required directories will be created
    :param cls_name: determines which classifier to use, as defined in set_classifier. This assumes predict_proba is
    implemented, to avoid predicting twice. May break some classifiers though
    :param dataset: Name of problem. Files must be  <problem_path>/<dataset>/<dataset>+"_TRAIN"+format, same for "_TEST"
    :param resampleID: Seed for resampling. If set to 0, the default train/test split from file is used. Also used in output file name.
    :param overwrite: if set to False, this will only build results if there is not a result file already present. If
    True, it will overwrite anything already there
    :param format: Valid formats are ".ts", ".arff" and ".long". For more info on format, see
    https://github.com/alan-turing-institute/sktime/blob/master/examples/Loading%20Data%20Examples.ipynb
    :param train_files: whether to generate train files or not. If true, it performs a 10xCV on the train and saves
    :return:
    """
    if not overwrite:
        full_path = str(results_path)+"/"+str(cls_name)+"/Predictions/" + str(dataset) +"/testFold"+str(resampleID)+".csv"
        if os.path.exists(full_path):
            print(full_path+" Already exists and overwrite set to false")
            return
    # TO DO: Automatically differentiate between problem types, currently only works with .ts
    trainX, trainY = load_ts(problem_path + dataset + '/' + dataset + '_TRAIN'+format)
    testX, testY = load_ts(problem_path + dataset + '/' + dataset + '_TEST'+format)
    classifier = set_classifier(cls_name)
    print(cls_name+" on "+dataset+" resample number "+str(resampleID))
    # TO DO : Use Viktors code here
    if train_files:
        print(" Generating train files not yet implemented")

    start = int(round(time.time() * 1000))
    classifier.fit(trainX,trainY)
    build_time = int(round(time.time() * 1000))-start
    start =  int(round(time.time() * 1000))
    probs = classifier.predict_proba(testX)
    preds = classifier.classes_[np.argmax(probs, axis=1)]
    test_time = int(round(time.time() * 1000))-start
    second = str(classifier.get_params())
    correct = 0
    for i in range(0, testY.__len__()):
        if testY[i] == preds[i]:
            correct += 1
        ac = correct / testY.__len__()

    third = str(ac)+","+str(build_time)+","+str(test_time)+",-1,-1,"+str(len(classifier.classes_))+ "," + str(classifier.classes_)

    write_results_to_uea_format(second_line=second, third_line=third, output_path=results_path, classifier_name=cls_name, resample_seed= resampleID,
                                predicted_class_vals=preds, actual_probas=probs, dataset_name=dataset, actual_class_vals=testY)

#        results_path+classifier_name+'/Predictions/'+dataset_name+'/testFold'+resampleID+".csv")

def write_results_to_uea_format(output_path, classifier_name, dataset_name, actual_class_vals,
                                predicted_class_vals, split='TEST', resample_seed=0, actual_probas=None, second_line="No Parameter Info",third_line="N/A",class_labels=None):
    """
    This is very alpha and I will probably completely change the structure once train fold is sorted, as that internally
    does all this I think!
    Output mirrors that produced by this Java
    https://github.com/TonyBagnall/uea-tsc/blob/master/src/main/java/experiments/Experiments.java
    :param output_path:
    :param classifier_name:
    :param dataset_name:
    :param actual_class_vals:
    :param predicted_class_vals:
    :param split:
    :param resample_seed:
    :param actual_probas:
    :param second_line:
    :param third_line:
    :param class_labels:
    :return:
    """
    if len(actual_class_vals) != len(predicted_class_vals):
        raise IndexError("The number of predicted class values is not the same as the number of actual class values")

    try:
        os.makedirs(str(output_path)+"/"+str(classifier_name)+"/Predictions/" + str(dataset_name) + "/")
    except os.error:
        pass  # raises os.error if path already exists

    if split == 'TRAIN' or split == 'train':
        train_or_test = "train"
    elif split == 'TEST' or split == 'test':
        train_or_test = "test"
    else:
        raise ValueError("Unknown 'split' value - should be TRAIN/train or TEST/test")

    file = open(str(output_path)+"/"+str(classifier_name)+"/Predictions/" + str(dataset_name) +
                "/"+str(train_or_test)+"Fold"+str(resample_seed)+".csv", "w")

    print(classifier_name+" on "+dataset_name+" for resample "+str(resample_seed)+" Has line three "+third_line)
    # the first line of the output file is in the form of:
    # <classifierName>,<datasetName>,<train/test>,<Class Labels>
    file.write(str(dataset_name) + ","+str(classifier_name)+"," + str(train_or_test)+","+str(resample_seed)+",MILLISECONDS,PREDICTIONS, Generated by experiments.py")
    file.write("\n")

    # the second line of the output is free form and classifier-specific; usually this will record info
    # such as build time, parameter options used, any constituent model names for ensembles, etc.
    file.write(str(second_line)+"\n")

    # the third line of the file is the accuracy (should be between 0 and 1 inclusive). If this is a train
    # output file then it will be a training estimate of the classifier on the training data only (e.g.
    # 10-fold cv, leave-one-out cv, etc.). If this is a test output file, it should be the output
    # of the estimator on the test data (likely trained on the training data for a-priori parameter optimisation)
    file.write(str(third_line))
    file.write("\n")


    # from line 4 onwards each line should include the actual and predicted class labels (comma-separated). If
    # present, for each case, the probabilities of predicting every class value for this case should also be
    # appended to the line (a space is also included between the predicted value and the predict_proba). E.g.:
    #
    # if predict_proba data IS provided for case i:
    #   actual_class_val[i], predicted_class_val[i],,prob_class_0[i],prob_class_1[i],...,prob_class_c[i]
    #
    # if predict_proba data IS NOT provided for case i:
    #   actual_class_val[i], predicted_class_val[i]
    for i in range(0, len(predicted_class_vals)):
        file.write(str(actual_class_vals[i]) + "," + str(predicted_class_vals[i]))
        if actual_probas is not None:
            file.write(",")
            for j in actual_probas[i]:
                file.write("," + str(j))
            file.write("\n")

    file.close()




if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing
    """
#Input args -dp=${dataDir} -rp=${resultsDir} -cn=${classifier} -dn=${dataset} -f=\$LSB_JOBINDEX
    if sys.argv.__len__() > 1: #cluster run, this is fragile
        data_dir = sys.argv[1]
        results_dir = sys.argv[2]
        classifier =  sys.argv[3]
        dataset = sys.argv[4]
        resample = sys.argv[5]
        run_experiment(problem_path=data_dir, results_path=results_dir, cls_name=classifier, dataset=dataset,
                       resampleID=resample)
    else : #Local run
        data_dir = "E:/Data/sktimeFormat/TSCProblems/"
        results_dir = "E:/Results/UCR Debug/Python/"
        classifier = "TSF"
        resample = 0
 #       for i in range(3, len(datasets)):
 #           dataset = datasets[i]
        dataset = "ItalyPowerDemand"
        run_experiment(problem_path=data_dir, results_path=results_dir, cls_name=classifier, dataset=dataset, resampleID=resample)

