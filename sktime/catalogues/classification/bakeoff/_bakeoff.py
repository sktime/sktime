"""TSC Bake off 2017 catalogue."""

from sklearn.metrics import accuracy_score

from sktime.catalogues.base import BaseCatalogue


class BakeOffCatalogue(BaseCatalogue):
    """Catalogue of Bake Off datasets, classifiers, metrics, and cv."""

    _tags = {
        "authors": "jgyasu",
        "maintainers": "jgyasu",
        "object_type": "catalogue",
        "catalogue_type": "mixed",
        "n_items": 87,
        "n_datasets": 85,
        "n_classifiers": 1,
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
            "StarlightCurves",
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
            "ShapeletLearningClassifierTslearn()",
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
