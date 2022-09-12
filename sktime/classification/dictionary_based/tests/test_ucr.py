# -*- coding: utf-8 -*-
"""UCR test."""
import os

os.environ["KMP_WARNINGS"] = "off"

import itertools
import sys

sys.path.insert(0, "./")

import time
from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=UserWarning)

import numpy as np
import pandas as pd
import torch
from convst.classifiers import R_DST_Ridge
from joblib import Parallel, delayed, parallel_backend
from scipy.stats import zscore
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from sktime.classification.dictionary_based import (
    WEASEL,
    WEASEL_STEROIDS,
    BOSSEnsemble,
    ContractableBOSS,
    Hydra,
    TemporalDictionaryEnsemble,
)
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


dataset_names_full = [
    "ACSF1",
    "Adiac",
    # "AllGestureWiimoteX",
    # "AllGestureWiimoteY",
    # "AllGestureWiimoteZ",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "BME",
    "Car",
    "CBF",
    "Chinatown",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "Crop",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    # "DodgerLoopDay",
    # "DodgerLoopGame",
    # "DodgerLoopWeekend",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "ElectricDevices",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "EthanolLevel",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    # "Fungi",
    # "GestureMidAirD1",
    # "GestureMidAirD2",
    # "GestureMidAirD3",
    # "GesturePebbleZ1",
    # "GesturePebbleZ2",
    "GunPoint",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "HouseTwenty",
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
    # "MelbournePedestrian",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
    "MixedShapesRegularTrain",
    "MixedShapesSmallTrain",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "PickupGestureWiimoteZ",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    # "PLAID",
    "Plane",
    "PowerCons",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "Rock",
    "ScreenType",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SmoothSubspace",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarLightCurves",  # 5 Mins
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
    "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",  # error???
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",
]

dataset_names_excerpt = [
    # "FreezerSmallTrain",
    # "GestureMidAirD3",
    # "InlineSkate",
    # "MelbournePedestrian",
    # "PickupGestureWiimoteZ",
    # "PowerCons",
    # "SemgHandMovementCh2",
    # "SmoothSubspace",
    # "Worms",
    # "WormsTwoClass",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "Car",
    "CBF",
    "Coffee",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "ECG200",
    "ECGFiveDays",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "GunPoint",
    "ItalyPowerDemand",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
    "OliveOil",
    "Plane",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "SyntheticControl",
    "TwoLeadECG",
    "Wine",
]


def get_classifiers(threads_to_use):
    """Obtain the benchmark classifiers."""
    clfs = {
        # "WEASEL": WEASEL(random_state=1379, n_jobs=threads_to_use),
        # "BOSS": BOSSEnsemble(random_state=1379, n_jobs=threads_to_use),
        # "cBOSS": ContractableBOSS(random_state=1379, n_jobs=threads_to_use),
        # "TDE": TemporalDictionaryEnsemble(random_state=1379, n_jobs=threads_to_use),
        "WEASEL 2a": WEASEL_STEROIDS(
            random_state=1379,
            binning_strategies=["equi-depth"],
            alphabet_sizes=[2],
            lower_bounding=False,
            min_window=4,
            max_window=24,
            max_feature_count=10_000,
            word_lengths=[8],
            norm_options=[False],
            variance=True,
            ensemble_size=50,
            use_first_differences=[True, False],
            feature_selection="none",
            # sections=2,
            # remove_repeat_words=True,
            n_jobs=threads_to_use,
        ),
        # "WEASEL 2b": WEASEL_STEROIDS(
        #     random_state=1379,
        #     binning_strategies=["equi-depth"],
        #     alphabet_sizes=[2],
        #     lower_bounding=False,
        #     min_window=4,
        #     max_window=24,
        #     max_feature_count=10_000,
        #     word_lengths=[8],
        #     norm_options=[False],
        #     variance=True,
        #     ensemble_size=50,
        #     use_first_differences=[True, False],
        #     feature_selection="none",
        #     sections=4,
        #     # remove_repeat_words=True,
        #     n_jobs=threads_to_use,
        # ),
        # "WEASEL 2c": WEASEL_STEROIDS(
        #     random_state=1379,
        #     binning_strategies=["equi-depth"],
        #     alphabet_sizes=[2],
        #     lower_bounding=False,
        #     min_window=4,
        #     max_window=24,
        #     max_feature_count=10_000,
        #     word_lengths=[8],
        #     norm_options=[False],
        #     variance=True,
        #     ensemble_size=50,
        #     use_first_differences=[True, False],
        #     feature_selection="none",
        #     sections=8,
        #     # remove_repeat_words=True,
        #     n_jobs=threads_to_use,
        # ),
        # "WEASEL 2d": WEASEL_STEROIDS(
        #     random_state=1379,
        #     binning_strategies=["equi-depth"],
        #     alphabet_sizes=[2],
        #     lower_bounding=False,
        #     min_window=4,
        #     max_window=24,
        #     max_feature_count=10_000,
        #     word_lengths=[8],
        #     norm_options=[False],
        #     variance=True,
        #     ensemble_size=50,
        #     use_first_differences=[True, False],
        #     feature_selection="none",
        #     sections=16,
        #     # remove_repeat_words=True,
        #     n_jobs=threads_to_use,
        # ),
        # "WEASEL 2b": WEASEL_STEROIDS(
        #     random_state=1379,
        #     binning_strategies=["equi-depth", "equi-width"],
        #     alphabet_sizes=[2],
        #     lower_bounding=True,
        #     min_window=4,
        #     max_window=40,
        #     max_feature_count=10_000,
        #     word_lengths=[8],
        #     norm_options=[False],
        #     variance=True,
        #     ensemble_size=50,
        #     use_first_differences=[True, False],
        #     feature_selection="none",
        #     sections=sections,
        #     # remove_repeat_words=True,
        #     n_jobs=threads_to_use,
        # ),
        # "WEASEL 2c": WEASEL_STEROIDS(
        #     random_state=1379,
        #     binning_strategies=["equi-depth", "equi-width"],
        #     alphabet_sizes=[2],
        #     lower_bounding=True,
        #     min_window=8,
        #     max_window=32,
        #     # remove_repeat_words=True,
        #     max_feature_count=10_000,
        #     word_lengths=[8],
        #     norm_options=[False],
        #     variance=True,
        #     ensemble_size=50,
        #     use_first_differences=[True, False],
        #     feature_selection="none",
        #     sections=sections,
        #     n_jobs=threads_to_use,
        # ),
        # "Hydra": [],  # see below
        # "R_DST": R_DST_Ridge(random_state=1379),
        # "Rocket": make_pipeline(
        #     Rocket(random_state=1379, n_jobs=threads_to_use),
        #     RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True),
        # ),
        # "MiniRocket": make_pipeline(
        #     MiniRocket(random_state=1379, n_jobs=threads_to_use),
        #     RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True),
        # ),
    }
    return clfs


DATA_PATH = "/Users/bzcschae/workspace/UCRArchive_2018/"
parallel_jobs = 1
threads_to_use = 4
server = False

# local
if os.path.exists(DATA_PATH):
    DATA_PATH = "/Users/bzcschae/workspace/UCRArchive_2018/"
    used_dataset = dataset_names_excerpt
# server
else:
    DATA_PATH = "/vol/fob-wbib-vol2/wbi/schaefpa/sktime/datasets/UCRArchive_2018"
    parallel_jobs = 80
    threads_to_use = 1
    server = True
    used_dataset = dataset_names_full

if __name__ == "__main__":

    def _parallel_fit(dataset_name, clf_name):
        # ignore all future warnings
        simplefilter(action="ignore", category=FutureWarning)
        simplefilter(action="ignore", category=UserWarning)

        X_train, y_train = load_from_ucr_tsv_to_dataframe_plain(
            os.path.join(DATA_PATH, dataset_name, dataset_name + "_TRAIN.tsv")
        )
        X_test, y_test = load_from_ucr_tsv_to_dataframe_plain(
            os.path.join(DATA_PATH, dataset_name, dataset_name + "_TEST.tsv")
        )

        X_train.fillna(0, inplace=True)
        X_test.fillna(0, inplace=True)

        sum_scores = {
            clf_name: {
                "dataset": [],
                "all_scores": [],
                "all_fit": [],
                "all_pred": [],
                "fit_time": 0.0,
                "pred_time": 0.0,
            }
        }

        # try:

        X_train = np.reshape(np.array(X_train), (len(X_train), 1, -1))
        X_test = np.reshape(np.array(X_test), (len(X_test), 1, -1))

        if clf_name == "Hydra":
            fit_time = time.perf_counter()
            transform = Hydra(X_train.shape[-1])
            X_training_transform = transform(torch.tensor(X_train).float())

            clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
            clf.fit(X_training_transform, y_train)
            fit_time = np.round(time.perf_counter() - fit_time, 5)

            pred_time = time.perf_counter()
            X_test_transform = transform(torch.tensor(X_test).float())
            acc = clf.score(X_test_transform, y_test)
            pred_time = np.round(time.perf_counter() - pred_time, 5)
        else:
            clf = get_classifiers(threads_to_use)[clf_name]
            fit_time = time.perf_counter()
            clf.fit(X_train, y_train)
            fit_time = np.round(time.perf_counter() - fit_time, 5)

            pred_time = time.perf_counter()
            acc = clf.score(X_test, y_test)
            pred_time = np.round(time.perf_counter() - pred_time, 5)

        print(
            f"Dataset={dataset_name}, "
            + (
                f"Feature Count={clf.total_features_count}, "
                if hasattr(clf, "total_features_count")
                else f""
            )
            + f"Train-Size={np.shape(X_train)}, "
            + f"Test-Size={np.shape(X_test)}"
            + f"\n\tclassifier={clf_name}"
            + f"\n\ttime (fit, predict)="
            f"{np.round(fit_time, 2), np.round(pred_time, 2)}"
            + f"\n\taccuracy={np.round(acc, 3)}"
        )

        sum_scores[clf_name]["dataset"].append(dataset_name)
        sum_scores[clf_name]["all_scores"].append(acc)
        sum_scores[clf_name]["all_fit"].append(fit_time)
        sum_scores[clf_name]["all_pred"].append(pred_time)

        sum_scores[clf_name]["fit_time"] += sum_scores[clf_name]["fit_time"] + fit_time
        sum_scores[clf_name]["pred_time"] += (
            sum_scores[clf_name]["pred_time"] + pred_time
        )

        # except Exception as e:
        #    print("An exception occurred: {}".format(e))
        #    print("\tFailed: ", dataset_name, clf_name)
        #    print(e)

        print("-----------------")

        return sum_scores

    # with parallel_backend("threading", n_jobs=-1):
    parallel_res = Parallel(n_jobs=parallel_jobs, timeout=9999999, batch_size=1)(
        delayed(_parallel_fit)(dataset, clf_name)
        for dataset, clf_name in itertools.product(
            used_dataset, get_classifiers(threads_to_use)
        )
    )

    sum_scores = {}
    for result in parallel_res:
        if not sum_scores:
            sum_scores = result
        else:
            for name, data in result.items():
                if name not in sum_scores:
                    sum_scores[name] = {}
                for key, value in data.items():
                    if key not in sum_scores[name]:
                        if type(value) == list:
                            sum_scores[name][key] = []
                        else:
                            sum_scores[name][key] = 0
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

    csv_timings = []
    csv_scores = []
    for name, _ in sum_scores.items():
        all_accs = sum_scores[name]["all_scores"]
        all_datasets = sum_scores[name]["dataset"]
        for acc, dataset_name in zip(all_accs, all_datasets):
            csv_scores.append((name, dataset_name, acc))

        all_fit = np.round(sum_scores[name]["all_fit"], 2)
        all_pred = np.round(sum_scores[name]["all_pred"], 2)
        for fit, pred, dataset_name in zip(all_fit, all_pred, all_datasets):
            csv_timings.append((name, dataset_name, fit, pred))

    if server:
        pd.DataFrame.from_records(
            csv_scores,
            columns=[
                "Classifier",
                "Dataset",
                "Accuracy",
                # "Fit-Time",
                # "Predict-Time",
            ],
        ).to_csv("ucr-112-accuracy-12-09-22.csv", index=None)

        pd.DataFrame.from_records(
            csv_timings,
            columns=[
                "Classifier",
                "Dataset",
                "Fit-Time",
                "Predict-Time",
            ],
        ).to_csv("ucr-112-runtime-12-09-22.csv", index=None)
