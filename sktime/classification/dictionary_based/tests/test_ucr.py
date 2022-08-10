# -*- coding: utf-8 -*-
"""UCR test."""
import os
import sys
import time
from warnings import simplefilter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import zscore
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from sktime.classification.dictionary_based import WEASEL, WEASEL_STEROIDS
from sktime.transformations.panel.rocket import MiniRocket, Rocket

sys.path.append("../../..")


def load_from_ucr_tsv_to_dataframe_plain(full_file_path_and_name):
    """Load UCR datasets."""
    df = pd.read_csv(
        full_file_path_and_name,
        sep=r"\s+|\t+|\s+\t+|\t+\s+",
        engine="python",
        header=None,
    )
    y = df.pop(0).values
    df.columns -= 1
    return df, y


dataset_names_excerpt = [
    # 'ACSF1',
    # 'Adiac',
    # 'AllGestureWiimoteX',
    # 'AllGestureWiimoteY',
    # 'AllGestureWiimoteZ',
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    # 'BME',
    "Car",
    "CBF",
    # 'Chinatown',
    # 'ChlorineConcentration',
    # 'CinCECGTorso',
    "Coffee",
    # 'Computers',
    # 'CricketX',
    # 'CricketY',
    # 'CricketZ',
    # 'Crop',
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    # 'DodgerLoopDay',
    # 'DodgerLoopGame',
    # 'DodgerLoopWeekend',
    # 'Earthquakes',
    "ECG200",
    # 'ECG5000',
    "ECGFiveDays",
    # 'ElectricDevices',
    # 'EOGHorizontalSignal',
    # 'EOGVerticalSignal',
    # 'EthanolLevel',
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    # 'FiftyWords',
    # 'Fish',
    # 'FordA',
    # 'FordB',
    # 'FreezerRegularTrain',
    # 'FreezerSmallTrain',
    # 'Fungi',
    # 'GestureMidAirD1',
    # 'GestureMidAirD2',
    # 'GestureMidAirD3',
    # 'GesturePebbleZ1',
    # 'GesturePebbleZ2',
    "Gun_Point",
    # 'GunPointAgeSpan',
    # 'GunPointMaleVersusFemale',
    # 'GunPointOldVersusYoung',
    # 'Ham',
    # 'HandOutlines',
    # 'Haptics',
    # 'Herring',
    # 'HouseTwenty',
    # 'InlineSkate',
    # 'InsectEPGRegularTrain',
    # 'InsectEPGSmallTrain',
    # 'InsectWingbeatSound',
    "ItalyPowerDemand",
    # 'LargeKitchenAppliances',
    # 'Lightning2',
    # 'Lightning7',
    # 'Mallat',
    # 'Meat',
    # 'MedicalImages',
    # 'MelbournePedestrian',
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
    # 'Missing_value_and_variable_length_datasets_adjusted',
    # 'MixedShapesRegularTrain',
    # 'MixedShapesSmallTrain',
    # 'MoteStrain',
    # 'NonInvasiveFetalECGThorax1',
    # 'NonInvasiveFetalECGThorax2',
    "OliveOil",
    # 'OSULeaf',
    # 'PhalangesOutlinesCorrect',
    # 'Phoneme',
    # 'PickupGestureWiimoteZ',
    # 'PigAirwayPressure',
    # 'PigArtPressure',
    # 'PigCVP',
    # 'PLAID',
    "Plane",
    # 'PowerCons',
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    # 'RefrigerationDevices',
    # 'Rock',
    # 'ScreenType',
    # 'SemgHandGenderCh2',
    # 'SemgHandMovementCh2',
    # 'SemgHandSubjectCh2',
    # 'ShakeGestureWiimoteZ',
    # 'ShapeletSim',
    # 'ShapesAll',
    # 'SmallKitchenAppliances',
    # 'SmoothSubspace',
    "SonyAIBORobot Surface",
    "SonyAIBORobot SurfaceII",
    # 'StarLightCurves',
    # 'Strawberry',
    # 'SwedishLeaf',
    # 'Symbols',
    "synthetic_control",
    # 'ToeSegmentation1',
    # 'ToeSegmentation2',
    # 'Trace',
    "TwoLeadECG",
    # 'TwoPatterns',
    # 'UMD',
    # 'UWaveGestureLibraryAll',
    # 'UWaveGestureLibraryX',
    # 'UWaveGestureLibraryY',
    # 'UWaveGestureLibraryZ',
    # 'Wafer',
    "Wine",
    # 'WordSynonyms',
    # 'Worms',
    # 'WormsTwoClass',
    # 'Yoga'
]


others = []

DATA_PATH = "/Users/bzcschae/workspace/similarity/datasets/classification/"
parallel_jobs = 4


if __name__ == "__main__":

    def _parallel_fit(dataset_name):
        csv_scores = []
        sum_scores = {}

        # ignore all future warnings
        simplefilter(action="ignore", category=FutureWarning)

        X_train, y_train = load_from_ucr_tsv_to_dataframe_plain(
            os.path.join(DATA_PATH, dataset_name, dataset_name + "_TRAIN")
        )
        X_test, y_test = load_from_ucr_tsv_to_dataframe_plain(
            os.path.join(DATA_PATH, dataset_name, dataset_name + "_TEST")
        )

        clfs = {
            "WEASEL": WEASEL(random_state=1379, n_jobs=4),
            "WEASEL ST": WEASEL_STEROIDS(
                random_state=1379,
                binning_strategies=["equi-depth", "equi-width"],
                variance=True,
                ensemble_size=50,
                n_jobs=4,
            ),
            "Rocket": make_pipeline(
                Rocket(random_state=1379),
                RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True),
            ),
            "MiniRocket": make_pipeline(
                MiniRocket(random_state=1379),
                RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True),
            ),
        }
        for name, _ in clfs.items():
            sum_scores[name] = {
                "dataset": [],
                "all_scores": [],
                "fit_time": 0.0,
                "pred_time": 0.0,
            }

        # z-norm training/test data
        X_train = zscore(X_train, axis=1)
        X_test = zscore(X_test, axis=1)
        X_train = np.reshape(np.array(X_train), (len(X_train), 1, -1))
        X_test = np.reshape(np.array(X_test), (len(X_test), 1, -1))

        print(
            f"Running Dataset={dataset_name}, "
            f"Train-Size={np.shape(X_train)}, "
            f"Test-Size={np.shape(X_test)}"
        )

        for name, clf in clfs.items():
            # try:
            fit_time = time.time()
            clf.fit(X_train, y_train)
            fit_time = np.round(time.time() - fit_time, 5)

            pred_time = time.time()
            acc = clf.score(X_test, y_test)
            pred_time = np.round(time.time() - pred_time, 5)

            print(
                f"Dataset={dataset_name}"
                + f"\n\tclassifier={name}"
                + f"\n\ttime (fit, predict)="
                f"{np.round(fit_time, 3), np.round(pred_time, 3)}"
                + f"\n\taccuracy={np.round(acc, 4)}"
            )

            sum_scores[name]["dataset"].append(dataset_name)
            sum_scores[name]["all_scores"].append(acc)
            sum_scores[name]["fit_time"] += sum_scores[name]["fit_time"] + fit_time
            sum_scores[name]["pred_time"] += sum_scores[name]["pred_time"] + pred_time

            csv_scores.append((name, clf, dataset_name, acc, fit_time, pred_time))

        # except Exception as e:
        #    print("An exception occurred: {}".format(e))
        #    print("\tFailed: ", dataset_name, name)
        #    sum_scores[name]["dataset"].append(dataset_name)
        #    sum_scores[name]["all_scores"].append(0)
        #    sum_scores[name]["fit_time"] += sum_scores[name]["fit_time"] + 0
        #    sum_scores[name]["pred_time"] += sum_scores[name]["pred_time"] + 0
        #    csv_scores.append((name, clf, dataset_name, 0, 0, 0))
        print("-----------------")

        return sum_scores  # , csv_scores

    parallel_res = Parallel(n_jobs=parallel_jobs)(
        delayed(_parallel_fit)(dataset) for dataset in dataset_names_excerpt
    )

    sum_scores = {}
    for result in parallel_res:
        if not sum_scores:
            sum_scores = result
        else:
            for name, data in result.items():
                for key, value in data.items():
                    sum_scores[name][key] += value

    print("\n\n---- Final results -----")

    for name, _ in sum_scores.items():
        print("---- Name", name, "-----")
        print(
            "Total mean-accuracy:", np.round(np.mean(sum_scores[name]["all_scores"]), 3)
        )
        print(
            "Total std-accuracy:", np.round(np.std(sum_scores[name]["all_scores"]), 3)
        )
        print(
            "Total median-accuracy:",
            np.round(np.median(sum_scores[name]["all_scores"]), 2),
        )
        print("Total fit_time:", np.round(sum_scores[name]["fit_time"], 2))
        print("Total pred_time:", np.round(sum_scores[name]["pred_time"], 2))
        print("-----------------")

    """
    pd.DataFrame.from_records(
        scores,
        columns=[
            "Classifier",
            "Dataset",
            "Accuracy",
            "Fit-Time",
            "Predict-Time",
        ],
    ).to_csv("scores.csv", index=None)"""
