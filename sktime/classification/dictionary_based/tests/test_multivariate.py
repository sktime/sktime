# -*- coding: utf-8 -*-
"""UCR test."""
import os

from pandas.errors import PerformanceWarning

from sktime.datasets import load_UCR_UEA_dataset

os.environ["KMP_WARNINGS"] = "off"

import itertools
import sys
import time
from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=UserWarning)
simplefilter(action="ignore", category=PerformanceWarning)


import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sktime.classification.dictionary_based import MUSE, MUSE_NEW

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
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    ## "CharacterTrajectories",  # variable length
    "Cricket",
    "DuckDuckGeese",  # Muse old is slow
    "EigenWorms",
    "Epilepsy",
    "ERing",
    "EthanolConcentration",
    "FaceDetection",  # muse a=4 has problems
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    ## "InsectWingbeat",  # variable length
    ## "JapaneseVowels",  # variable length
    "Libras",
    "LSST",
    "MotorImagery",
    "NATOPS",
    "PEMS-SF",  # PEMS-SF, MUSE old, Feature Count=31930999, 128 Stunden
    "PenDigits",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    ## "SpokenArabicDigits",  # variable length
    "StandWalkJump",
    "UWaveGestureLibrary",
]

dataset_names_excerpt = [
    "AtrialFibrillation",
    "BasicMotions",
    "Epilepsy",
    "ERing",
    "FingerMovements",
    "Handwriting",
    "Libras",
    "NATOPS",
    "RacketSports",
    "UWaveGestureLibrary",
]


def get_classifiers(threads_to_use):
    """Obtain the benchmark classifiers."""
    clfs = {
        # "MUSE (old)" : MUSE(random_state=1379, n_jobs=threads_to_use),
        "MUSE 2a (default +46 +variance)": MUSE_NEW(
            random_state=1379,
            alphabet_size=2,
            variance=True,
            anova=False,
            n_jobs=threads_to_use,
        ),
        "MUSE 2b (default +46)": MUSE_NEW(
            random_state=1379, alphabet_size=2, n_jobs=threads_to_use
        ),
        # "MUSE 2c (default +46 -bigrams)": MUSE_NEW(
        #     random_state=1379, alphabet_size=2, bigrams=False, n_jobs=threads_to_use
        # ),
        # "MUSE 2d (default +46 -bigrams +variance)": MUSE_NEW(
        #     random_state=1379,
        #     alphabet_size=2,
        #     bigrams=False,
        #     variance=True,
        #     anova=False,
        #     n_jobs=threads_to_use,
        # ),
    }
    return clfs


DATA_PATH = "/Users/bzcschae/Downloads/Multivariate_ts"
parallel_jobs = 1
threads_to_use = 4
server = False

# local
if os.path.exists(DATA_PATH):
    DATA_PATH = "/Users/bzcschae/Downloads/Multivariate_ts"
    used_dataset = dataset_names_excerpt
# server
else:
    DATA_PATH = "/vol/fob-wbib-vol2/wbi/schaefpa/sktime/datasets/Multivariate_ts"
    parallel_jobs = 80
    threads_to_use = 1
    server = True
    used_dataset = dataset_names_full

if __name__ == "__main__":

    def _parallel_fit(dataset_name, clf_name):
        # ignore all future warnings
        simplefilter(action="ignore", category=FutureWarning)
        simplefilter(action="ignore", category=UserWarning)
        simplefilter(action="ignore", category=PerformanceWarning)

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

        X_train, y_train = load_UCR_UEA_dataset(
            dataset_name,
            split="train",
            extract_path=DATA_PATH,
            return_type="numpy3D",
        )
        X_test, y_test = load_UCR_UEA_dataset(
            dataset_name,
            split="test",
            extract_path=DATA_PATH,
            return_type="numpy3D",
        )

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

    parallel_res = Parallel(
        n_jobs=parallel_jobs,
        # backend="threading",
        timeout=9999999,
        batch_size=1,
    )(
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
            ],
        ).to_csv("classifier_all_scores_mv_06-09-22.csv", index=None)

        pd.DataFrame.from_records(
            csv_timings,
            columns=[
                "Classifier",
                "Dataset",
                "Fit-Time",
                "Predict-Time",
            ],
        ).to_csv("classifier_all_runtimes_mv_06-09-22.csv", index=None)
