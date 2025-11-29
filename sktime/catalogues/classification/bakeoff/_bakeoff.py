"""TSC Bake off 2017 catalogue."""

from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit

from sktime.catalogues.base import BaseCatalogue


class BakeOffCatalogue(BaseCatalogue):
    """TSC Bake off 2017 catalogue.

    Catalogue of datasets, classifiers, metrics, and CV splitters used in the
    2017 Time Series Classification (TSC) Bake Off.

    Notes
    -----
    The original bake off used fixed, predefined train/test splits supplied as
    separate files for each dataset and did not use cross-validation. These
    fixed splits are intentionally not included here, as repeated reuse of a
    single partition can lead to overfitting on that specific split.

    Instead, this catalogue provides a lightweight ``ShuffleSplit`` with
    ``n_splits=1``. With a given ``random_state``, it offers a repeatable
    single train/test split in the spirit of the original protocol, without
    hard-coding the historical partitions.

    Examples
    --------
    >>> from sktime.catalogues.classification import BakeOffCatalogue
    >>> from sktime.benchmarking.classification import ClassificationBenchmark
    >>> catalogue = BakeOffCatalogue(random_state=42)
    >>> benchmark = ClassificationBenchmark()
    >>> benchmark.add(catalogue) # doctest: +SKIP
    >>> benchmark.run() # doctest: +SKIP
    """

    _tags = {
        "authors": "jgyasu",
        "maintainers": "jgyasu",
        "python_dependencies": ["tslearn"],
        "object_type": "catalogue",
        "catalogue_type": "mixed",
        "n_items": 100,
        "n_datasets": 85,
        "n_classifiers": 13,
        "n_metrics": 1,
        "n_cv_splitters": 1,
        "info:name": "The great time series classification bake off",
        "info:source": "https://doi.org/10.1007/s10618-016-0483-9",
    }

    def __init__(self, random_state=None):
        super().__init__()
        self.random_state = random_state

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
            "RotationForest()",
        ]

        metrics = [accuracy_score]

        cv_splitters = [ShuffleSplit(n_splits=1, random_state=self.random_state)]

        all_objects = {
            "dataset": [f"UCRUEADataset('{dataset}')" for dataset in datasets],
            "classifier": classifiers,
            "metric": metrics,
            "cv_splitter": cv_splitters,
        }

        return all_objects
