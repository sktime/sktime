"""TSC Bake off 2017 catalogue."""

from sklearn.metrics import accuracy_score

from sktime.catalogues.base import BaseCatalogue


class BakeOffCatalogue(BaseCatalogue):
    """Catalogue of Bake Off datasets, classifiers, metrics, and cv."""

    _tags = {
        "authors": "jgyasu",
        "maintainers": "jgyasu",
        "python_dependencies": ["tslearn"],
        "object_type": "catalogue",
        "catalogue_type": "mixed",
        "n_items": 98,
        "n_datasets": 85,
        "n_classifiers": 12,
        "n_metrics": 1,
        "n_cv_splitters": 0,
        "info:name": "The great time series classification bake off",
        "info:source": "https://doi.org/10.1007/s10618-016-0483-9",
    }

    def _get(self):
        """Return a dict of items (datasets, forecasters, metrics)."""
        datasets = [
            "MiddlePhalanxTW",
            "TwoLeadECG",
            "WormsTwoClass",
            "Meat",
            "MiddlePhalanxOutlineCorrect",
            "Beef",
            "ArrowHead",
            "ECG5000",
            "ScreenType",
            "Wine",
            "Adiac",
            "Coffee",
            "DistalPhalanxOutlineCorrect",
            "OliveOil",
            "Symbols",
            "Lightning2",
            "SonyAIBORobotSurface2",
            "Computers",
            "Haptics",
            "UWaveGestureLibraryY",
            "FacesUCR",
            "SwedishLeaf",
            "WordSynonyms",
            "ProximalPhalanxTW",
            "ProximalPhalanxOutlineAgeGroup",
            "ShapeletSim",
            "Herring",
            "Car",
            "CricketY",
            "UWaveGestureLibraryZ",
            "Phoneme",
            "PhalangesOutlinesCorrect",
            "HandOutlines",
            "InlineSkate",
            "SyntheticControl",
            "Yoga",
            "ProximalPhalanxOutlineCorrect",
            "ToeSegmentation2",
            "Ham",
            "CricketZ",
            "Trace",
            "Worms",
            "UWaveGestureLibraryAll",
            "CricketX",
            "DiatomSizeReduction",
            "FordA",
            "CinCECGTorso",
            "MedicalImages",
            "FaceAll",
            "Plane",
            "ToeSegmentation1",
            "RefrigerationDevices",
            "OSULeaf",
            "MoteStrain",
            "ShapesAll",
            "CBF",
            "ECG200",
            "Lightning7",
            "SonyAIBORobotSurface1",
            "SmallKitchenAppliances",
            "FordB",
            "ECGFiveDays",
            "TwoPatterns",
            "DistalPhalanxOutlineAgeGroup",
            "LargeKitchenAppliances",
            "NonInvasiveFetalECGThorax2",
            "Mallat",
            "DistalPhalanxTW",
            "ItalyPowerDemand",
            "GunPoint",
            "NonInvasiveFetalECGThorax1",
            "Earthquakes",
            "ChlorineConcentration",
            "Wafer",
            "Fish",
            "ElectricDevices",
            "StarLightCurves",
            "InsectWingbeatSound",
            "FiftyWords",
            "BirdChicken",
            "MiddlePhalanxOutlineAgeGroup",
            "Strawberry",
            "BeetleFly",
            "UWaveGestureLibraryX",
            "FaceFour",
        ]

        classifiers = [
            # Dictionary based
            "BOSSEnsemble()",
            # Shapelet based
            "ShapeletLearningClassifierTslearn()",
            # Interval based
            "TimeSeriesForestClassifier()",
            # KNN based
            "KNeighborsTimeSeriesClassifier(distance='dtw')",
            "KNeighborsTimeSeriesClassifier(distance='euclidean')",
            "KNeighborsTimeSeriesClassifier(distance='wdtw')",
            "KNeighborsTimeSeriesClassifier(distance='wddtw')",
            "KNeighborsTimeSeriesClassifier(distance='lcss')",
            "KNeighborsTimeSeriesClassifier(distance='erp')",
            "KNeighborsTimeSeriesClassifier(distance='msm')",
            # Feature based?
            "ProximityForest()",
            "ElasticEnsemble()",
        ]

        metrics = [accuracy_score]

        cv_splitters = []

        all_objects = {
            "dataset": [f"UCRUEADataset('{dataset}')" for dataset in datasets],
            "classifier": classifiers,
            "metric": metrics,
            "cv_splitter": cv_splitters,
        }

        return all_objects
