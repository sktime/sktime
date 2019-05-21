import os

import sktime.utils.load_data

def convert(src_dir, datasets_dir_name, dataset_names = None, dest_dir = None):
    if dataset_names == None:
        dataset_names = src_dir + '/' + datasets_dir_name
    if not isinstance(dataset_names, list):
        if os.path.isdir(dataset_names):
            dataset_names = os.listdir(dataset_names)
        else:
            # todo read names from file here
            raise NotImplementedError()
    if dest_dir == None:
        dest_dir = src_dir
    datasets_dir = src_dir + '/' + datasets_dir_name
    for dataset_name in dataset_names:
        try:
            os.makedirs(dest_dir + '/' + dataset_name)
        except:
            pass
        train_file_path_src = datasets_dir + '/' + dataset_name + '/' + dataset_name + '_TRAIN'
        test_file_path_src = datasets_dir + '/' + dataset_name + '/' + dataset_name + '_TEST'
        train_file_path_dest = dest_dir + '/' + dataset_name + '/' + dataset_name + '_TRAIN'
        test_file_path_dest = dest_dir + '/' + dataset_name + '/' + dataset_name + '_TEST'
        sktime.utils.load_data.arff_to_ts(train_file_path_src, train_file_path_dest)
        sktime.utils.load_data.arff_to_ts(test_file_path_src, test_file_path_dest)

if __name__ == '__main__':
    dataset_names = [
            # "GunPoint",
            # "ItalyPowerDemand",
            # "ArrowHead",
            # "Coffee",
            # "Adiac",
            # "Beef",
            # "BeetleFly",
            # "BirdChicken",
            # "Car",
            # "CBF",
            # "ChlorineConcentration",
            # "CinCECGTorso",
            # "Computers",
            # "CricketX",
            # "CricketY",
            # "CricketZ",
            # "DiatomSizeReduction",
            # "DistalPhalanxOutlineCorrect",
            # "DistalPhalanxOutlineAgeGroup",
            # "DistalPhalanxTW",
            # "Earthquakes",
            # "ECG200",
            # "ECG5000",
            # "ECGFiveDays",
            #    "ElectricDevices",
            # "FaceAll",
            # "FaceFour",
            # "FacesUCR",
            # "FiftyWords",
            # "Fish",
            #    "FordA",
            #    "FordB",
            # "Ham",
            #    "HandOutlines",
            # "Haptics",
            # "Herring",
            # "InlineSkate",
            # "InsectWingbeatSound",
            # "LargeKitchenAppliances",
            # "Lightning2",
            # "Lightning7",
            # "Mallat",
            # "Meat",
            # "MedicalImages",
            # "MiddlePhalanxOutlineCorrect",
            # "MiddlePhalanxOutlineAgeGroup",
            # "MiddlePhalanxTW",
            # "MoteStrain",
            # "NonInvasiveFetalECGThorax1",
            # "NonInvasiveFetalECGThorax2",
            # "OliveOil",
            # "OSULeaf",
            # "PhalangesOutlinesCorrect",
            # "Phoneme",
            # "Plane",
            # "ProximalPhalanxOutlineCorrect",
            # "ProximalPhalanxOutlineAgeGroup",
            # "ProximalPhalanxTW",
            # "RefrigerationDevices",
            # "ScreenType",
            # "ShapeletSim",
            # "ShapesAll",
            # "SmallKitchenAppliances",
            # "SonyAIBORobotSurface1",
            # "SonyAIBORobotSurface2",
            #    "StarlightCurves",
            # "Strawberry",
            # "SwedishLeaf",
            # "Symbols",
            # "SyntheticControl",
            # "ToeSegmentation1",
            # "ToeSegmentation2",
            # "Trace",
            # "TwoLeadECG",
            # "TwoPatterns",
            # "UWaveGestureLibraryX",
            # "UWaveGestureLibraryY",
            # "UWaveGestureLibraryZ",
            # "UWaveGestureLibraryAll",
            # "Wafer",
            # "Wine",
            # "WordSynonyms",
            # "Worms",
            # "WormsTwoClass",
            # "Yoga",
            ]
    convert('/scratch', 'mv_datasets', dest_dir = '/scratch/mv_datasets_ts')

