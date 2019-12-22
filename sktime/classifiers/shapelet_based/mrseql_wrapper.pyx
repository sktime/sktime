from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sktime.transformers.dictionary_based.SFA import SFA
from sktime.classifiers.base import BaseClassifier

######################### SAX and SFA #########################

cdef extern from "mrseql_cpp/sax_converter.h":
    cdef cppclass SAX:
        SAX(int, int, int)
        #string timeseries2SAX(string, string)
        vector[string] timeseries2SAX(vector[double])
        vector[double] map_weighted_patterns(vector[double], vector[string], vector[double])

cdef class PySAX:
    cdef SAX *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, int N, int w, int a):
        self.thisptr = new SAX(N, w, a)
    def __dealloc__(self):
        del self.thisptr
    def timeseries2SAX(self, ts):
        return self.thisptr.timeseries2SAX(ts)
        #if isinstance(obj, basestring):
        #    return self.thisptr.timeseries2SAX(ts, delimiter)
    def timeseries2SAXseq(self, ts):
        words = self.thisptr.timeseries2SAX(ts)
        seq = b''
        #print(words)
        for w in words:
            seq = seq + b' ' + w
        if seq: # remove extra space
            seq = seq[1:]
        return seq
    def map_weighted_patterns(self, ts, sequences, weights):
        return self.thisptr.map_weighted_patterns(ts, sequences, weights)

class AdaptedSFA:
    def __init__(self, int N, int w, int a):
        self.sfa = SFA(w,a,N,norm=True,remove_repeat_words=True)

    def fit(self, train_x):
        self.sfa.fit(train_x)

    def timeseries2SFAseq(self, ts):
        dfts = self.sfa.MFT(ts)
        sfa_str = b''
        for window in range(dfts.shape[0]):
            if sfa_str:
                sfa_str += b' '
            dft = dfts[window]
            first_char = ord(b'A')
            for i in range(self.sfa.word_length):
                for bp in range(self.sfa.alphabet_size):
                    if dft[i] <= self.sfa.breakpoints[i][bp]:
                        sfa_str += bytes([first_char + bp])
                        #print(chr(first_char + bp))
                        break
                first_char += self.sfa.alphabet_size
        return sfa_str

###########################################################################


#########################SEQL wrapper#########################


cdef extern from "mrseql_cpp/seql.h":
    cdef cppclass SEQL:
        SEQL()
        void learn(vector[string] &, vector[double] &)
        double brute_classify(string , double)
        void print_model(int)
        vector[string] get_sequence_features(bool)
        vector[double] get_coefficients(bool)


cdef class PySEQL:
    cdef SEQL *thisptr

    def __cinit__(self):
        self.thisptr = new SEQL()
    def __dealloc__(self):
        del self.thisptr

    def learn(self, vector[string] sequences, vector[double] labels):
        self.thisptr.learn(sequences, labels)

    def classify(self, string sequence):
        return self.thisptr.brute_classify(sequence, 0.0)

    def print_model(self):
        self.thisptr.print_model(100)

    def get_sequence_features(self, bool only_positive):
        return self.thisptr.get_sequence_features(only_positive)

    def get_coefficients(self, bool only_positive):
        return self.thisptr.get_coefficients(only_positive)


###########################################################################


######################### Mr-SEQL (main class) #########################
''' Time Series Classification with multiple symbolic representations and SEQL (Mr-SEQL)

 @article{mrseql,
 author = {Nguyen, Thach and Gsponer, Severin and Ilie, Iulia and O'reilly, Martin and Ifrim, Georgiana},
 title = {Interpretable Time Series Classification Using Linear Models and Multi-resolution Multi-domain Symbolic Representations},
 journal = {Data Mining and Knowledge Discovery},
 volume = {33},
 number = {4},
 year = {2019},
 }


Overview: Mr-SEQL is a time series classifier that learn from multiple symbolic representations of multiple resolutions and multiple domains.
Currently, Mr-SEQL supports both SAX and SFA representations.

Parameters
----------
seql_mode       : str, either 'clf' or 'fs'. In the 'clf', Mr-SEQL mode trains an ensemble of SEQL models while in the 'fs' mode it uses SEQL to select features for training a logistic regression model.
symrep          : list or tuple, should contains only 'sax' or 'sfa' or both. The symbolic representations to be used to transform the input time series.
symrep_config   : dict, customized parameters for the symbolic transformation. If defined, symrep will be ignored.

'''
class MrSEQLClassifier(BaseClassifier):


    def __init__(self, seql_mode='clf', symrep=['sax'], symrepconfig=None):

        self.symbolic_methods = symrep

        if seql_mode in ('fs','clf'):
            self.seql_mode = seql_mode
        else:
            raise ValueError('seql_mode should be either clf or fs.')

        if symrepconfig is None:
            self.config = [] # http://effbot.org/zone/default-values.htm
        else:
            self.config = symrepconfig

        #self.label_dict = {} # for translating labels since seql only accept [1.-1] as labels
        self.seql_models = [] # seql models

        # all the unique labels in the data
        # in case of binary data the first one is always the negative class
        self.classes_ = []



        self.clf = None # scikit-learn model

        # store fitted sfa for later transformation
        self.sfas = {}


    def __is_multiclass(self):
        return len(self.classes_) > 2

    # change arbitrary binary labels to -1, 1 labels as SEQL can only work with -1, 1
    def __to_tmp_labels(self, y):
        return [1 if l == self.classes_[1] else -1 for l in y]


    def __transform_time_series(self, ts_x):
        multi_tssr = []

        # generate configuration if not predefined
        if not self.config:
            min_ws = 16
            max_ws = ts_x.shape[1]
            pars = [[w, 16, 4] for w in range(min_ws, max_ws, int(np.sqrt(max_ws)))]

            if 'sax' in self.symbolic_methods:
                for p in pars:
                    self.config.append({'method':'sax','window':p[0],'word':p[1],'alphabet':p[2]})

            if 'sfa' in self.symbolic_methods:
                for p in pars:
                    self.config.append({'method':'sfa','window':p[0],'word':8,'alphabet':p[2]})


        for cfg in self.config:

            tssr = []

            if cfg['method'] == 'sax': # convert time series to SAX
                ps = PySAX(cfg['window'],cfg['word'],cfg['alphabet'])
                for ts in ts_x:
                    sr = ps.timeseries2SAXseq(ts)
                    tssr.append(sr)

            if cfg['method'] == 'sfa':  # convert time series to SFA
                if (cfg['window'],cfg['word'],cfg['alphabet']) not in self.sfas:
                    sfa = AdaptedSFA(cfg['window'],cfg['word'],cfg['alphabet'])
                    sfa.fit(ts_x)
                    self.sfas[(cfg['window'],cfg['word'],cfg['alphabet'])] = sfa
                for ts in ts_x:
                    sr = self.sfas[(cfg['window'],cfg['word'],cfg['alphabet'])].timeseries2SFAseq(ts)
                    tssr.append(sr)

            multi_tssr.append(tssr)


        return multi_tssr


    def __fit_binary_problem(self, mr_seqs, labels):
        models = []
        for rep in mr_seqs:
            m = PySEQL()
            m.learn(rep, labels)
            models.append(m)
        return models


    def __fit_multiclass_problem(self, mr_seqs, labels):
        # one versus all
        ova_models = []
        for l in self.classes_:
            tmp_labels = [1 if x == l else -1 for x in labels]
            models = self.__fit_binary_problem(mr_seqs, tmp_labels)
            ova_models.append(models)
        return ova_models

    # represent data (in multiple reps form) in feature space
    def __to_feature_space(self, mr_seqs):
        full_fm = []

        if self.__is_multiclass():
            for ova_models in self.seql_models:
                for rep, model in zip(mr_seqs, ova_models):
                    seq_features = model.get_sequence_features(True)
                    # print(seq_features)
                    fm = np.zeros((len(rep), len(seq_features)))
                    for i,s in enumerate(rep):
                        for j,f in enumerate(seq_features):
                            if f in s:
                                fm[i,j] = 1
                    full_fm.append(fm)

        else:
            for rep, model in zip(mr_seqs, self.seql_models):
                seq_features = model.get_sequence_features(False)
                # print(seq_features)
                fm = np.zeros((len(rep), len(seq_features)))
                for i,s in enumerate(rep):
                    for j,f in enumerate(seq_features):
                        if f in s:
                            fm[i,j] = 1
                full_fm.append(fm)
        full_fm = np.hstack(full_fm)
        return full_fm

    '''
    Check if X input is correct.
    From dictionary_based/boss.py
    '''
    def __X_check(self,X):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] > 1:
                raise TypeError("Mr-SEQL cannot handle multivariate problems yet")
            elif isinstance(X.iloc[0, 0], pd.Series):
                X = np.asarray([a.values for a in X.iloc[:, 0]])
            else:
                raise TypeError("Input should either be a 2d numpy array, or a pandas dataframe with a single column of Series objects.")
        return X

    def fit(self, X, y, input_checks=True):

        X = self.__X_check(X)

        # transform time series to multiple symbolic representations
        mr_seqs = self.__transform_time_series(X)

        self.classes_ = np.unique(y) #because sklearn also uses np.unique

        if self.__is_multiclass(): #one versus all
            self.seql_models = self.__fit_multiclass_problem(mr_seqs, y)
        else:
            temp_labels = self.__to_tmp_labels(y)
            self.seql_models = self.__fit_binary_problem(mr_seqs, temp_labels)

        # if seql is being used to select features
        # first computing the feature vectors
        # then fit the new data to a logistic regression model
        if self.seql_mode == 'fs':
            train_x = self.__to_feature_space(mr_seqs)
            self.clf = LogisticRegression(solver='newton-cg',multi_class = 'multinomial', class_weight='balanced').fit(train_x, y)
            self.classes_ = self.clf.classes_ # shouldn't matter

    def __compute_proba(self, score):
        # if score < -8000:
        #    return 0
        # else:
        return 1.0 / (1.0 + np.exp(-score))

    def predict_proba(self, X, input_checks=True):
        if input_checks:
            X = self.__X_check(X)
        mr_seqs = self.__transform_time_series(X)

        if self.seql_mode == 'fs':
            test_x = self.__to_feature_space(mr_seqs)
            return self.clf.predict_proba(test_x) # TODO: Check if sklearn clf store labels in the same order



        if self.__is_multiclass():
            scores = np.zeros((len(X), len(self.classes_)))

            for li, ova_models in enumerate(self.seql_models):
                for rep, model in zip(mr_seqs, ova_models):
                    for c,seq in enumerate(rep):
                        scores[c,li] += model.classify(seq)

            proba = self.__compute_proba(scores)
            proba /= proba.sum(axis=1).reshape((proba.shape[0], -1)) # https://github.com/scikit-learn/scikit-learn/blob/bf24c7e3d6d768dddbfad3c26bb3f23bc82c0a18/sklearn/linear_model/_base.py#L300

            return proba




        else: # binary
            scores = np.zeros(len(X))
            for rep, model in zip(mr_seqs, self.seql_models):
                for c,seq in enumerate(rep):
                    scores[c] = scores[c] + model.classify(seq)
            positive_proba = self.__compute_proba(scores)
            return np.vstack([1 - positive_proba, positive_proba]).T # https://github.com/scikit-learn/scikit-learn/blob/bf24c7e3d6d768dddbfad3c26bb3f23bc82c0a18/sklearn/linear_model/_base.py#L300






    def predict(self, X, input_checks=True):
        if input_checks:
            X = self.__X_check(X)
        proba = self.predict_proba(X, False)
        return np.array([self.classes_[np.argmax(prob)] for prob in proba])


    def get_configuration(self):
        return self.config

    # map sax features on time series
    # return a time series with the same length
    # each value represents the importanceness
    def map_sax_model(self, ts):
        weighted_ts = np.zeros(len(ts))
        if self.__is_multiclass():
            print('Do nothing')
        else:
            for cfg,m in zip(self.config, self.seql_models):
                if cfg['method'] == 'sax':
                    features = m.get_sequence_features(False)
                    weights = m.get_coefficients(False)
                    ps = PySAX(cfg['window'],cfg['word'],cfg['alphabet'])
                    weighted_ts += ps.map_weighted_patterns(ts,features,weights)
        return weighted_ts




    def summary(self):
        print('Symbolic methods: ' + ', '.join(self.symbolic_methods))
        if not self.config:
            print('No symbolic parameters found. To be generated later.')

        if self.seql_mode == 'fs':
            print('Classification Method: SEQL as feature selection')
        elif self.seql_mode == 'clf':
            print('Classification Method: Ensemble SEQL')
