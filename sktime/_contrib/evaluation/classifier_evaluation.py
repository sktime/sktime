# -*- coding: utf-8 -*-
"""Working code that will be moved into its own package."""
from sktime.datasets import load_UCR_UEA_dataset

""" 39 UEA multivariate time series classification problems, 2022 version"""
multivariate = [
    "ArticularyWordRecognition",
    "AsphaltObstaclesCoordinates",
    "AsphaltPavementTypeCoordinates",
    "AsphaltRegularityCoordinates",
    "AtrialFibrillation",
    "BasicMotions",
    "CharacterTrajectories",
    "Cricket",
    "DuckDuckGeese",
    "EigenWorms",
    "EMOPain",
    "Epilepsy",
    "ERing",
    "EthanolConcentration",
    "EyesOpenShut",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "InsectWingbeat",
    "JapaneseVowels",
    "Libras",
    "LSST",
    "MindReading",
    "MotorImagery",
    "MotionSenseHAR",
    "NATOPS",
    "PenDigits",
    "PEMS-SF",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "Siemens",
    "SpokenArabicDigits",
    "StandWalkJump",
    "Tiselac",
    "UWaveGestureLibrary",
]
temp = [
    "PEMS-SF"
]

univariate = [
    "ACSF1",
    "Adiac",
    "AllGestureWiimoteX",
    "AllGestureWiimoteY",
    "AllGestureWiimoteZ",
    "ArrowHead",
    "AsphaltObstacles",
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
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxOutlineAgeGroup",
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
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxOutlineAgeGroup",
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
]

testy = ["AsphaltObstaclesCoordinates"]


def run_experiments(
    classifiers,
    datasets,
    extract_path=None,
    overwrite=True,
    results_path="../local_results/",
    resample=0,
    build_train=False,
):
    """Run experiments."""
    for cls_name in classifiers:
        for dataset in datasets:
            X_train, y_train = load_UCR_UEA_dataset(
                problem, split="TRAIN", extract_path=extract_path
            )
            X_test, y_test = load_UCR_UEA_dataset(
                problem, split="TEST", extract_path=extract_path
            )
            classifier = set_classifier(cls_name)
            load_and_run_classification_experiment(
                overwrite=overwrite,
                problem_path=extract_path,
                results_path=results_path,
                cls_name=cls_name,
                classifier=classifier,
                dataset=dataset,
                resample_id=resample,
                build_train=build_train
                #                predefined_resample=predefined_resample,
            )


if __name__ == "__main__":
    """Main test"""
    print("Length of univariate  = ", len(univariate))
    for problem in univariate:
        try:
            X, y = load_UCR_UEA_dataset(problem)
            print(
                " Shape of combined train test for ",
                problem,
                " is X = ",
                X.shape,
                " Y = ",
                y.shape,
            )
        except:
            print("Failed to load ", problem)

    print("Length of multivariate  = ", len(multivariate))
    for problem in multivariate:
        try:
            X, y = load_UCR_UEA_dataset(problem)
            print(
                " Shape of combined train test for ",
                problem,
                " is X = ",
                X.shape,
                " Y = ",
                y.shape,
            )
        except Exception as e:
            print("Failed to load ", problem, " exception = ", e)
