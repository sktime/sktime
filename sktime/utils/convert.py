import os

import joblib

import sktime.utils.load_data

def convert_dataset(datasets_dir_path, dataset_name, dest_dir_path):
    dataset_dir_path = datasets_dir_path + '/' + dataset_name
    file_names = os.listdir(dataset_dir_path)
    for file_name in file_names:
        if file_name.endswith('.arff'):
            file_name = file_name[:-5]  # trim extension
            src_path = dataset_dir_path + '/' + file_name
            dest_path = dest_dir_path + '/' + dataset_name + '/' + file_name
            try:
                os.makedirs(dest_dir_path + '/' + dataset_name)
            except:
                pass
            sktime.utils.load_data.arff_to_ts(src_path, dest_path)
            loaded_data = sktime.utils.load_data.load_from_tsfile_to_dataframe(dest_path + '.ts')

def convert(src_dir, datasets_dir_name, dataset_names = None, dest_dir_path = None):
    if dataset_names == None:
        dataset_names = src_dir + '/' + datasets_dir_name
    if not isinstance(dataset_names, list):
        if os.path.isdir(dataset_names):
            dataset_names = os.listdir(dataset_names)
        else:
            # todo read names from file here
            raise NotImplementedError()
    if dest_dir_path == None:
        dest_dir_path = src_dir
    datasets_dir_path = src_dir + '/' + datasets_dir_name
    par = joblib.Parallel(n_jobs = -1)
    par(joblib.delayed(convert_dataset)(datasets_dir_path, dataset_name, dest_dir_path) for dataset_name in dataset_names)

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
    convert('/scratch', 'mv_datasets', dest_dir_path = '/scratch/mv_datasets_ts'
            # , dataset_names = ['BasicMotions']
            # , dataset_names = ['InsectWingbeat']
            )

