# -*- coding: utf-8 -*-
"""UCR test."""
import os

os.environ["KMP_WARNINGS"] = "off"

import sys
import time
from warnings import simplefilter

import numpy as np
import pandas as pd
import torch

# from convst.classifiers import R_DST_Ridge
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from sktime.classification.dictionary_based import WEASEL, WEASEL_STEROIDS, Hydra
from sktime.transformations.panel.rocket import MiniRocket, Rocket

# from scipy.stats import zscore


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
    "AllGestureWiimoteX",
    "AllGestureWiimoteY",
    "AllGestureWiimoteZ",
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
    "DodgerLoopDay",
    "DodgerLoopGame",
    "DodgerLoopWeekend",
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
    "Fungi",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "GesturePebbleZ1",
    "GesturePebbleZ2",
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
    "MelbournePedestrian",
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
    "PLAID",
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
    "StarLightCurves",
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
    "UWaveGestureLibraryZ",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",
]

dataset_names_excerpt = [
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


simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=UserWarning)

others = []

# DATA_PATH = "/Users/bzcschae/workspace/UCRArchive_2018/"
# parallel_jobs = 1

DATA_PATH = "/vol/fob-wbib-vol2/wbi/schaefpa/sktime/datasets/UCRArchive_2018"
parallel_jobs = 40

if __name__ == "__main__":

    def _parallel_fit(dataset_name):
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

        threads_to_use = 4
        clfs = {
            # "WEASEL": WEASEL(random_state=1379, n_jobs=threads_to_use),
            "WEASEL-ST (2nd Best)": WEASEL_STEROIDS(
                random_state=1379,
                alphabet_sizes=[2],
                binning_strategies=["equi-depth", "equi-width"],
                min_window=4,
                max_window=24,
                max_feature_count=10_000,
                word_lengths=[8],
                norm_options=[False],
                variance=True,
                ensemble_size=50,
                use_first_differences=[True, False],
                n_jobs=threads_to_use,
            ),
            "WEASEL_ST (Best)": WEASEL_STEROIDS(
                random_state=1379,
                binning_strategies=["equi-depth"],
                alphabet_sizes=[4],
                min_window=4,
                max_window=24,
                max_feature_count=10_000,
                word_lengths=[8],  # test only 6 or 8?
                norm_options=[False],  # p[True]=0.8
                variance=True,
                ensemble_size=50,
                use_first_differences=[True, False],
                n_jobs=threads_to_use,
            ),
            "WEASEL (Bench)": WEASEL_STEROIDS(
                random_state=1379,
                # alphabet_sizes=[2],
                binning_strategies=["equi-depth"],  # "kmeans"
                word_lengths=[6, 8],  # test only 6 or 8?
                norm_options=[True, True, True, True, False],  # p[True]=0.8
                variance=True,
                max_feature_count=10_000,
                ensemble_size=50,
                use_first_differences=[True, False],
                n_jobs=threads_to_use,
            ),
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

        sum_scores = {}
        for name, _ in clfs.items():
            sum_scores[name] = {
                "dataset": [],
                "all_scores": [],
                "fit_time": 0.0,
                "pred_time": 0.0,
            }

        # z-norm training/test data
        # X_train = zscore(X_train, axis=1)
        # X_test = zscore(X_test, axis=1)
        X_train = np.reshape(np.array(X_train), (len(X_train), 1, -1))
        X_test = np.reshape(np.array(X_test), (len(X_test), 1, -1))

        # print(
        #    f"Running Dataset={dataset_name}, "
        #    f"Train-Size={np.shape(X_train)}, "
        #    f"Test-Size={np.shape(X_test)}"
        # )

        for name, clf in clfs.items():
            if name == "Hydra":
                transform = Hydra(X_train.shape[-1])
                X_training_transform = transform(torch.tensor(X_train).float())
                X_test_transform = transform(torch.tensor(X_test).float())

                clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
                fit_time = time.process_time()
                clf.fit(X_training_transform, y_train)
                fit_time = np.round(time.process_time() - fit_time, 5)

                pred_time = time.process_time()
                acc = clf.score(X_test_transform, y_test)
                pred_time = np.round(time.process_time() - pred_time, 5)
            else:
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
                + f"\n\tclassifier={name}"
                + f"\n\ttime (fit, predict)="
                f"{np.round(fit_time, 2), np.round(pred_time, 2)}"
                + f"\n\taccuracy={np.round(acc, 3)}"
            )

            sum_scores[name]["dataset"].append(dataset_name)
            sum_scores[name]["all_scores"].append(acc)
            sum_scores[name]["fit_time"] += sum_scores[name]["fit_time"] + fit_time
            sum_scores[name]["pred_time"] += sum_scores[name]["pred_time"] + pred_time

            # print("DFT:", SFA_NEW.time_dft)
            # print("MCB:", SFA_NEW.time_mcb)
        print("-----------------")

        return sum_scores

    parallel_res = Parallel(n_jobs=parallel_jobs, timeout=99999)(
        delayed(_parallel_fit)(dataset) for dataset in dataset_names_full
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

    csv_scores = []
    for name, _ in sum_scores.items():
        all_accs = sum_scores[name]["all_scores"]
        # total_fit_time in sum_scores[name]["all_scores"]
        # total_predict_time in sum_scores[name]["all_scores"]
        for acc, dataset_name in zip(all_accs, dataset_names_excerpt):
            csv_scores.append((name, dataset_name, acc))

    pd.DataFrame.from_records(
        csv_scores,
        columns=[
            "Classifier",
            "Dataset",
            "Accuracy",
            # "Fit-Time",
            # "Predict-Time",
        ],
    ).to_csv("full_run_classifier_all_scores.csv", index=None)
