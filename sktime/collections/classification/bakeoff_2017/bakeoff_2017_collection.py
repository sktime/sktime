"""TSC Bakeoff 2017 Collection."""

from sktime.collections.base import BaseCollection


class TSCBakeOff2017(BaseCollection):
    """Collection of datasets, estimators, and metrics from TSC Bake Off 2017.

    Examples
    --------
    >>> from sktime.collections import TSCBakeOff2017
    >>> collection = TSCBakeOff2017()
    >>> available_categories = collection.available_categories()
    >>> all_items = collection.get("all")
    >>> dataset_loaders = collection.get("dataset_loaders")
    >>> classifiers = collection.get("estimators")
    """

    _tags = {
        "authors": "jgyasu",
        "maintainers": "jgyasu",
        "collection_type": "mixed",
        "info:name": "The Great Time Series Classification Bake Off",
        "info:source": "https://arxiv.org/pdf/1602.01711",
    }

    def _get(self):
        """Return a dict of items (datasets, estimators, metrics)."""
        return items


dataset_loaders = [
    "LargeKitchenAppliances",
    "SmallKitchenAppliances",
    "OSULeaf",
    "TwoPatterns",
    "FaceFour",
    "Wafer",
    "Plane",
    "RefrigerationDevices",
    "ProximalPhalanxTW",
    "TwoLeadECG",
    "ProximalPhalanxOutlineCorrect",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineCorrect",
    "Beef",
    "WormsTwoClass",
    "OliveOil",
    "MoteStrain",
    "ToeSegmentation1",
    "MiddlePhalanxOutlineAgeGroup",
    "Worms",
    "ArrowHead",
    "Lightning2",
    "ShapeletSim",
    "ElectricDevices",
    "Adiac",
    "ProximalPhalanxOutlineAgeGroup",
    "ECGFiveDays",
    "Trace",
    "ItalyPowerDemand",
    "UWaveGestureLibraryX",
    "Computers",
    "Ham",
    "DistalPhalanxTW",
    "Wine",
    "FiftyWords",
    "BeetleFly",
    "CricketZ",
    "FordA",
    "FordB",
    "CricketY",
    "GunPoint",
    "Coffee",
    "FacesUCR",
    "UWaveGestureLibraryZ",
    "Strawberry",
    "UWaveGestureLibraryAll",
    "ToeSegmentation2",
    "DistalPhalanxOutlineAgeGroup",
    "Earthquakes",
    "FaceAll",
    "NonInvasiveFetalECGThorax2",
    "MiddlePhalanxOutlineCorrect",
    "CricketX",
    "CBF",
    "SonyAIBORobotSurface2",
    "PhalangesOutlinesCorrect",
    "Lightning7",
    "ShapesAll",
    "ScreenType",
    "ChlorineConcentration",
    "SyntheticControl",
    "InlineSkate",
    "ECG200",
    "InsectWingbeatSound",
    "Car",
    "WordSynonyms",
    "Fish",
    "HandOutlines",
    "NonInvasiveFetalECGThorax1",
    "SonyAIBORobotSurface1",
    "Meat",
    "UWaveGestureLibraryY",
    "Herring",
    "Symbols",
    "MedicalImages",
    "SwedishLeaf",
    "Phoneme",
    "Yoga",
    "ECG5000",
    "MiddlePhalanxTW",
    "StarLightCurves",
    "Mallat",
    "CinCECGTorso",
    "Haptics",
    "BirdChicken",
]

estimators = [
    "KNeighborsTimeSeriesClassifier",
]

items = {
    "dataset_loaders": [f"UCRUEADataset('{ds}')" for ds in dataset_loaders],
    "estimators": estimators,
}
