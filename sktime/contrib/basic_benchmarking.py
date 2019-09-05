import os

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

from sktime.utils.load_data import load_from_tsfile_to_dataframe as load_ts
import sktime.classifiers.interval_based.tsf as ib
import sktime.contrib.experiments as exp
import numpy as np

from sktime.transformers.compose import RowwiseTransformer
from sktime.transformers.segment import RandomIntervalSegmenter

from sktime.pipeline import Pipeline
from sktime.pipeline import FeatureUnion

from sktime.classifiers.compose import TimeSeriesForestClassifier
from sktime.utils.time_series import time_series_slope

from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

#method 1



benchmark_datasets = [
    "ACSF1",
    "Adiac",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "BME",
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
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "EthanolLevel",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "Fungi",
    "Ham",
    "Haptics",
    "Herring",
    "InlineSkate",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
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
    "MixedShapesRegularTrain",
    "MixedShapesSmallTrain",
    "MoteStrain",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "PickupGestureWiimoteZ",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "PLAID",
    "Plane",
    "PowerCons",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "Rock",
    "ScreenType",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ",
    "ShapeletSim",
    "SmallKitchenAppliances",
    "SmoothSubspace",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UMD",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",
]

data_dir = "Z:/ArchiveData/Univariate_ts/"
results_dir="Z:/Benchmarking/"

def tsf_benchmarking():
    for i in range(0, len(benchmark_datasets)):
        dataset = benchmark_datasets[i]
        print(str(i)+" problem = "+dataset)
        tsf = ib.TimeSeriesForest(n_trees=100)
        exp.run_experiment(overwrite=False, problem_path=data_dir, results_path=results_dir, cls_name="PythonTSF",
                           classifier=tsf,dataset=dataset, train_file=False)
        steps = [
            ('segment', RandomIntervalSegmenter(n_intervals='sqrt')),
            ('transform', FeatureUnion([
                ('mean', RowwiseTransformer(FunctionTransformer(func=np.mean, validate=False))),
                ('std', RowwiseTransformer(FunctionTransformer(func=np.std, validate=False))),
                ('slope', RowwiseTransformer(FunctionTransformer(func=time_series_slope, validate=False)))
            ])),
            ('clf', DecisionTreeClassifier())
        ]
        base_estimator = Pipeline(steps)
        tsf = TimeSeriesForestClassifier(base_estimator=base_estimator,
                                         n_estimators=100)
        exp.run_experiment(overwrite=True, problem_path=data_dir, results_path=results_dir, cls_name="PythonTSFComposite",
                       classifier=tsf, dataset=dataset, train_file=False)


def rise_benchmarking():
    for i in range(0, len(benchmark_datasets)):
        dataset = benchmark_datasets[i]
        print(str(i)+" problem = "+dataset)
        tsf = ib.TimeSeriesForest(n_trees=100)
        exp.run_experiment(overwrite=False, problem_path=data_dir, results_path=results_dir, cls_name="PythonTSF",
                           classifier=tsf,dataset=dataset, train_file=False)
        steps = [
            ('segment', RandomIntervalSegmenter(n_intervals='sqrt')),
            ('transform', FeatureUnion([
                ('mean', RowwiseTransformer(FunctionTransformer(func=np.mean, validate=False))),
                ('std', RowwiseTransformer(FunctionTransformer(func=np.std, validate=False))),
                ('slope', RowwiseTransformer(FunctionTransformer(func=time_series_slope, validate=False)))
            ])),
            ('clf', DecisionTreeClassifier())
        ]
        base_estimator = Pipeline(steps)
        tsf = TimeSeriesForestClassifier(base_estimator=base_estimator,
                                         n_estimators=100)
        exp.run_experiment(overwrite=False, problem_path=data_dir, results_path=results_dir, cls_name="PythonTSFComposite",
                       classifier=tsf, dataset=dataset, train_file=False)



if __name__ == "__main__":
    tsf_benchmarking()
