import numpy as np
import pandas as pd
import sys
import time
import os

from sklearn.metrics import accuracy_score
from scipy.stats import zscore
from sktime.classification.dictionary_based import WEASEL

sys.path.append("../../..")


def load_from_ucr_tsv_to_dataframe_plain(full_file_path_and_name):
    df = pd.read_csv(full_file_path_and_name, sep=r'\s+|\t+|\s+\t+|\t+\s+',
                     engine="python", header=None)
    y = df.pop(0).values
    df.columns -= 1
    return df, y


dataset_names_excerpt = [
    # 'ACSF1',
    # 'Adiac',
    # 'AllGestureWiimoteX',
    # 'AllGestureWiimoteY',
    # 'AllGestureWiimoteZ',

    'ArrowHead',
    'Beef',
    'BeetleFly',
    'BirdChicken',
    # 'BME',

    'Car',
    'CBF',
    # 'Chinatown',
    # 'ChlorineConcentration',
    # 'CinCECGTorso',
    'Coffee',
    # 'Computers',
    # 'CricketX',
    # 'CricketY',
    # 'CricketZ',
    # 'Crop',
    'DiatomSizeReduction',
    # 'DistalPhalanxOutlineAgeGroup',
    # 'DistalPhalanxOutlineCorrect',
    # 'DistalPhalanxTW',

    # 'DodgerLoopDay',
    # 'DodgerLoopGame',
    # 'DodgerLoopWeekend',
    # 'Earthquakes',
    'ECG200',
    # 'ECG5000',
    'ECGFiveDays',
    # 'ElectricDevices',
    # 'EOGHorizontalSignal',
    # 'EOGVerticalSignal',
    # 'EthanolLevel',
    # 'FaceAll',
    'FaceFour',
    # 'FacesUCR',
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
    'Gun_Point',
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
    'ItalyPowerDemand',
    # 'LargeKitchenAppliances',
    # 'Lightning2',
    # 'Lightning7',
    # 'Mallat',
    # 'Meat',
    # 'MedicalImages',
    # 'MelbournePedestrian',
    # 'MiddlePhalanxOutlineAgeGroup',
    # 'MiddlePhalanxOutlineCorrect',
    # 'MiddlePhalanxTW',
    # 'Missing_value_and_variable_length_datasets_adjusted',
    # 'MixedShapesRegularTrain',
    # 'MixedShapesSmallTrain',
    # 'MoteStrain',
    # 'NonInvasiveFetalECGThorax1',
    # 'NonInvasiveFetalECGThorax2',
    'OliveOil',
    # 'OSULeaf',
    # 'PhalangesOutlinesCorrect',
    # 'Phoneme',
    # 'PickupGestureWiimoteZ',
    # 'PigAirwayPressure',
    # 'PigArtPressure',
    # 'PigCVP',
    # 'PLAID',
    'Plane',
    # 'PowerCons',
    'ProximalPhalanxOutlineAgeGroup',
    'ProximalPhalanxOutlineCorrect',
    # 'ProximalPhalanxTW',
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
    'SonyAIBORobot Surface',
    'SonyAIBORobot SurfaceII',
    # 'StarLightCurves',
    # 'Strawberry',
    # 'SwedishLeaf',
    # 'Symbols',
    'synthetic_control',
    # 'ToeSegmentation1',
    # 'ToeSegmentation2',
    # 'Trace',
    'TwoLeadECG',
    # 'TwoPatterns',
    # 'UMD',
    # 'UWaveGestureLibraryAll',
    # 'UWaveGestureLibraryX',
    # 'UWaveGestureLibraryY',
    # 'UWaveGestureLibraryZ',
    # 'Wafer',
    'Wine',
    # 'WordSynonyms',
    # 'Worms',
    # 'WormsTwoClass',
    # 'Yoga'
]

DATA_PATH = "/Users/bzcschae/workspace/similarity/datasets/classification/"

if __name__ == '__main__':
    scores = []

    for dataset_name in dataset_names_excerpt:

        X_train, y_train = load_from_ucr_tsv_to_dataframe_plain(
            os.path.join(DATA_PATH, dataset_name, dataset_name + "_TRAIN"))
        X_test, y_test = load_from_ucr_tsv_to_dataframe_plain(
            os.path.join(DATA_PATH, dataset_name, dataset_name + "_TEST"))

        # z-norm training/test data
        X_train = zscore(X_train, axis=1).tolist()
        X_test = zscore(X_test, axis=1).tolist()

        df_train = pd.DataFrame()
        df_train['dim_0'] = (X_train)

        df_test = pd.DataFrame()
        df_test['dim_0'] = (X_test)

        clf = WEASEL(random_state=1379, window_inc=4)

        print(f"\nDataset: {dataset_name}")

        fit_time = time.process_time()
        clf.fit(df_train, y_train)
        fit_time = np.round(time.process_time() - fit_time, 5)

        pred_time = time.process_time()
        y_pred = clf.predict(df_test)
        pred_time = np.round(time.process_time() - pred_time, 5)

        acc = np.round(accuracy_score(y_test, y_pred), 5)
        print(f"accuracy_score={acc}")

        scores.append((dataset_name, acc, fit_time, pred_time))
        pd.DataFrame.from_records(scores,
                                  columns=['Dataset', 'Accuracy',
                                           'Fit-Time', 'Predict-Time'])\
            .to_csv("scores.csv", index=None)
