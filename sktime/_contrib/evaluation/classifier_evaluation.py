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


if __name__ == "__main__":
    """Main test"""
    print("Length of multivariate  = ", len(multivariate))
    for problem in multivariate:
        X, y = load_UCR_UEA_dataset(problem)
        print(
            " Shape of combined train test for ",
            problem,
            " is X = ",
            X.shape,
            " Y = ",
            y.shape,
        )
