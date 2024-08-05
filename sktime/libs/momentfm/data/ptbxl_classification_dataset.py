import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import wfdb
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import ast
from operator import itemgetter
from itertools import compress
from joblib import Parallel, delayed
import pickle
import pdb
import json 
from scipy import interpolate

def is_directory(path):
    extensions = ['.pth', '.txt', '.json', '.yaml']

    for ext in extensions:
        if ext in path:
            return False
    return True

def make_dir_if_not_exists(path, verbose=True):
    if not is_directory(path):
        path = path.split('.')[0]
    if not os.path.exists(path=path):
        os.makedirs(path)
        if verbose: print(f'Making directory: {path}...')    
    return True

def load_paths(basepath, fs):
    """ File Names to Load """
    folders1 = os.listdir(basepath)
    pth_to_folders = [folder for folder in folders1 if 'records%i' % fs in folder] #records100 contains 100Hz data
    pth_to_folders = [os.path.join(pth_to_folder,fldr) for pth_to_folder in pth_to_folders for fldr in os.listdir(os.path.join(basepath,pth_to_folder)) if not fldr.endswith('index.html')]
    files = [os.path.join(pth_to_folder,fldr) for pth_to_folder in pth_to_folders for fldr in os.listdir(os.path.join(basepath,pth_to_folder))]
    paths_to_files = [os.path.join(basepath,file.split('.hea')[0]) for file in files if '.hea' in file]
    return paths_to_files
    
def modify_df(basepath, output_type='single'):
    """Creates a dataframe with patient information, reports, data file paths and labels
    """
    """ Database with Patient-Specific Info """
    df = pd.read_csv(os.path.join(basepath,'ptbxl_database.csv'),index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    """ Database with Label Information """
    codes_df = pd.read_csv(os.path.join(basepath,'scp_statements.csv'),index_col=0)
        
    if output_type == 'single':
        encoder = LabelEncoder()
    elif output_type == 'multi':
        encoder = MultiLabelBinarizer()
    
    def aggregate_diagnostic(y_dic):
        """ Map Label To Diffeent Categories """
        tmp = []
        for key in y_dic.keys():
            if key in diag_agg_df.index:
                c = diag_agg_df.loc[key].diagnostic_class
                if str(c) != 'nan':
                    tmp.append(c)
        return list(set(tmp))
    
    aggregation_df = codes_df
    diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
    
    """ Obtain Superdiagnostic Label(s) """
    df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
    """ Obtain Number of Superdiagnostic Label(s) Per Recording """
    df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))
    
    """ Return Histogram of Each Label """
    min_samples = 0
    counts = pd.Series(np.concatenate(df.superdiagnostic.values)).value_counts()
    counts = counts[counts > min_samples]
    
    """ Obtain Encoded Label as New Column """
    if output_type == 'single':
        df = df[df.superdiagnostic_len == 1] # ==1 OR > 0
        df.superdiagnostic = df.superdiagnostic.apply(lambda entry:entry[0])
        encoder.fit(df.superdiagnostic.values)
        df['superdiagnostic_label'] = encoder.transform(df.superdiagnostic.values)
    elif output_type == 'multi':
        df = df[df.superdiagnostic_len > 0]
        encoder.fit(list(map(lambda entry: [entry],counts.index.values)))
        multi_hot_encoding_df = pd.DataFrame(encoder.transform(df.superdiagnostic.values),index=df.index,columns=encoder.classes_.tolist()) 
        df = pd.merge(df,multi_hot_encoding_df,on=df.index).drop(['key_0'],axis=1)
    
    return df

def obtain_phase_to_paths_dict(df,paths_to_files, phase_to_pids=None):
    """Creates a mapping between the phase (train, val, test) 
    and data files belonging to that phase
    """
    train_fold = np.arange(0,8)
    val_fold = [9]
    test_fold = [10]
    phases = ['train','val','test']
    folds = [train_fold,val_fold,test_fold]
    
    """ Obtain Patient IDs """
    if phase_to_pids is None:
        phase_to_pids = dict()
        for phase,fold in zip(phases,folds):
            current_ecgid = df[df.strat_fold.isin(fold)].index.tolist() #index is ecg_id by default when loading csv
            current_ecgid = list(map(lambda entry:int(entry),current_ecgid))
            phase_to_pids[phase] = current_ecgid
    else:
        phase_to_pids = phase_to_pids
    
    paths_to_ids = list(map(lambda path:int(path.split('/')[-1].split('_')[0]),paths_to_files))
    paths_to_ids_df = pd.Series(paths_to_ids)
    """ Obtain Paths For Each Phase """
    phase_to_paths = dict()
            
    for phase,pids in phase_to_pids.items():
        """ Obtain Paths For All Leads """
        paths = list(compress(paths_to_files,paths_to_ids_df.isin(pids).tolist()))
        # """ Assign Paths and Leads Labels """
        phase_to_paths[phase] = paths
    #pdb.set_trace()
    return phase_to_paths

def load_ptbxl_data(phase_to_paths, df, phase, output_type='single'):
    """ Load PTB-XL Data """
    paths = phase_to_paths[phase] #this is the most essential part
    
    """ Obtain IDs """
    ecg_ids = list(map(lambda path:int(path.split('/')[-1].split('_')[0]),paths))
    print("[INFO] Obtaining Labels")

    """ Obtain Labels """
    if output_type == 'single':
        labels = np.asarray([df[df.index == id_entry].superdiagnostic_label.iloc[0] for id_entry in tqdm(ecg_ids)])
    elif output_type == 'multi':
        labels = np.asarray([df[df.index == id_entry].iloc[0][-5:].tolist() for id_entry in tqdm(ecg_ids)])

    print("[INFO] Obtaining Text Reports")   
    """ Obtain Report Text For Each Entry """
    text_reports = np.asarray([df[df.index == id_entry].report.iloc[0] for id_entry in tqdm(ecg_ids)])
    
    return text_reports, labels, ecg_ids
  
class PTBXL_dataset(Dataset):
    def __init__(self, config, phase):
        self.phase = phase
        self.bands = list(range(12))
        self.fs = config.fs
        self.output_type = config.output_type
                 
        print("[INFO] Loading PTB-XL")

        #if folder not exist create 
        if not os.path.exists(os.path.join(config.cache_dir, "ptbxl")):
            os.mkdir(os.path.join(config.cache_dir, "ptbxl"))

        cache_path = os.path.join(config.cache_dir, f"ptbxl/{phase}.pkl")
        make_dir_if_not_exists(config.cache_dir)
        if os.path.exists(cache_path) and config.load_cache:
            print("[INFO] Cached file exists")
            with open(cache_path, "rb") as fp:
                self._timeseries, self._texts, self._labels, self._ecg_ids = pickle.load(fp)
        else:
            # Load path to the data and get dataframe with patient attributes, 
            # data path, labels and expert interpretations
            paths_to_files = load_paths(config.basepath, fs=self.fs)
            self.df = modify_df(config.basepath, output_type=self.output_type)

            print("[INFO] Modified database")
            # Create a mapping between data files and the train, test, val split they belong to
            phase_to_pids = json.load(open(config.path_to_pids))
            self.phase_to_paths = obtain_phase_to_paths_dict(self.df, paths_to_files, phase_to_pids=phase_to_pids)

            print("[INFO] Extracted paths")
            self._texts, self._labels, self._ecg_ids = load_ptbxl_data(self.phase_to_paths, self.df, self.phase, self.output_type)

            print("[INFO] Loading ECG Signals")        
            ecg_input_paths = self.phase_to_paths[self.phase]
            self._timeseries = []

            self._timeseries = Parallel(n_jobs=1)(delayed(self.get_timeseries)(ecg_input_path) for ecg_input_path in tqdm(ecg_input_paths))
            self._timeseries = np.stack(self._timeseries, axis=0)

            print("[INFO] Saving cache file")
            with open(cache_path, "wb") as fp:
                pickle.dump((self._timeseries, self._texts, self._labels, self._ecg_ids), fp)

        self._timeseries = np.stack(self._timeseries, axis=0)
        self._length = len(self._timeseries)
        self.n_classes = len(np.unique(self._labels))
    
        # Pre-process the timeseries
        timeseries = self._timeseries
        n_samples, n_leads, timesteps = timeseries.shape
        timeseries = timeseries.reshape(n_samples * n_leads, timesteps)

        f = interpolate.interp1d(np.linspace(0, 1, timesteps), 
                                 timeseries)
        self._timeseries = f(np.linspace(0, 1, config.seq_len))        
        
        # Normalize
        mean = np.mean(self._timeseries, axis=-1, keepdims=True)
        std = np.std(self._timeseries, axis=-1, keepdims=True)
        self._timeseries = (self._timeseries - mean) / std
        
        self._timeseries = self._timeseries.reshape(n_samples, n_leads, config.seq_len)

    def __len__(self):
        return self._length
    
    def get_timeseries(self, ecg_input_path):
        ecg_signal, _ = wfdb.rdsamp(ecg_input_path, channels=self.bands)
        return ecg_signal.T

    @property
    def timeseries(self):
        """Timeseries has shape # Number of data points x # features/leads x # time steps  
        """
        return self._timeseries
        
    @property
    def labels(self):
        return self._labels
    
    @property
    def texts(self):
        return self._texts
    
    @property
    def answers(self):
        return self._texts 
    
    @property 
    def questions(self):
        return np.asarray(['Caption the given ECG signal. \nAnswer: '] * len(self._texts))
    
    def ecg_ids(self):
        return self._ecg_ids

    def repr(self):
        return "A dataset object for the PTB-XL dataset."
    
    def __getitem__(self, idx):
        return self._timeseries[idx], self._labels[idx]