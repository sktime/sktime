import os
import shutil

import sktime.contrib.experiments

def arff_to_ts(file_path):
    print('converting ' + file_path + ' from arff to ts')
    source = open(file_path + '.arff', 'r')
    destination = open(file_path + '.ts', 'w')
    relation_tag = '@relation'
    attribute_tag = '@attribute'
    data_tag = '@data'
    end_tag = '@end'
    last_attribute = None
    data_begun = False
    for line in source:
        if data_begun:
            line = line.replace('\n', '')
            parts = line.split(',')
            next = parts[0]
            for part in parts[1:-1]:
                temp = next
                next = part
                part = temp
                if part == "'":
                    pass
                elif next == "'":
                    destination.write(part)
                    destination.write(':')
                else:
                    destination.write(part)
                    destination.write(',')
            if next != "'":
                destination.write(next)
                destination.write(':')
            destination.write(parts[-1])
            destination.write('\n')
        else:
            line_lower = line.lower()
            if line_lower.startswith(relation_tag):
                line = line[(len(relation_tag)):]
                destination.write('@problemName')
                line = line.replace('\n', '')
                destination.write(line)
                destination.write('\n')
                destination.write('@timeStamps false\n')
            elif line_lower.startswith(attribute_tag):
                parts = line.split()
                if parts[-1].lower() == 'relational':
                    pass
                else:
                    last_attribute = line[(len(attribute_tag)):]
            elif line_lower.startswith(data_tag):
                destination.write('@classLabel true ')
                parts = last_attribute.split()
                class_labels_str = parts[-1]
                class_labels_str = class_labels_str[1:-1]
                class_labels_str = class_labels_str.replace(', ', ' ')
                class_labels_str = class_labels_str.replace(',', ' ')
                destination.write(class_labels_str)
                destination.write('\n')
                data_begun = True
                destination.write('@data\n')
            elif line_lower.startswith(end_tag):
                pass
            else:
                if line.startswith('%'):
                    line = '#' + line[1:]
                destination.write(line)
    source.close()
    destination.close()

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
    dataset_names_file = '/scratch/datasets'
    if os.path.isdir(dataset_names_file):
        dataset_names = os.listdir(dataset_names_file)
    else:
        # read names from file here
        pass
    datasets_dir = '/scratch/datasets'
    for dataset_name in dataset_names:
        train_file_path = datasets_dir + '/' + dataset_name + '/' + dataset_name + '_TRAIN'
        test_file_path = datasets_dir + '/' + dataset_name + '/' + dataset_name + '_TEST'
        arff_to_ts(train_file_path)
        arff_to_ts(test_file_path)
        try:
            os.makedirs('/scratch/ts_datasets/' + dataset_name)
        except:
            pass
        os.system('cp ' + train_file_path + '.ts' + ' /scratch/ts_datasets/' +
                        dataset_name + '/' + dataset_name + '_TRAIN.ts')
        os.system('cp ' + test_file_path + '.ts' + ' /scratch/ts_datasets/' +
                        dataset_name + '/' + dataset_name + '_TEST.ts')
