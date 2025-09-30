"""TSC Bakeoff 2017 catalogue."""

from sktime.catalogues.base import BaseCatalogue


class TSCBakeOff2017(BaseCatalogue):
    """catalogue of datasets, estimators, and metrics from TSC Bake Off 2017.

    Examples
    --------
    >>> from sktime.catalogues import TSCBakeOff2017
    >>> catalogue = TSCBakeOff2017()
    >>> available_categories = catalogue.available_categories()
    >>> all_items = catalogue.get("all")
    >>> datasets = catalogue.get("dataset")
    >>> classifiers = catalogue.get("classifier")
    """

    _tags = {
        "authors": "jgyasu",
        "maintainers": "jgyasu",
        "catalogue_type": "mixed",
        "info:name": "The Great Time Series Classification Bake Off",
        "info:source": "https://doi.org/10.1007/s10618-016-0483-9",
    }

    def _get(self):
        """Return a dict of items (datasets, estimators, metrics)."""
        return items


datasets = [
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

classifiers = [
    "KNeighborsTimeSeriesClassifier()",
]

items = {
    "dataset": [f"UCRUEADataset('{dataset}')" for dataset in datasets],
    "classifier": classifiers,
}
