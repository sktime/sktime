STUFF = "Hi"

from sktime.utils.validation.series_as_features import check_X, check_y, _enforce_X_univariate
from sktime.classification.base import BaseClassifier
from sktime.transformers.series_as_features.dictionary_based import SFA
from sktime.utils.data_container import detabularize
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.string cimport string

__author__ = ["Thach Le Nguyen"]

######################### SAX and SFA #########################

cdef extern from "sax_converter.h":
    cdef cppclass SAX:
        SAX(int, int, int)        
        vector[string] timeseries2SAX(vector[double])
        vector[double] map_weighted_patterns(vector[double], vector[string], vector[double])

cdef class PySAX:
    '''
    Wrapper of SAX C++ implementation.
    '''
    cdef SAX * thisptr      # hold a C++ instance which we're wrapping

    def __cinit__(self, int N, int w, int a):
        self.thisptr = new SAX(N, w, a)

    def __dealloc__(self):
        del self.thisptr

    def timeseries2SAX(self, ts):
        return self.thisptr.timeseries2SAX(ts)
        

    def timeseries2SAXseq(self, ts):
        words = self.thisptr.timeseries2SAX(ts)
        seq = b''
        
        for w in words:
            seq = seq + b' ' + w
        if seq:  # remove extra space
            seq = seq[1:]
        return seq

    def map_weighted_patterns(self, ts, sequences, weights):
        return self.thisptr.map_weighted_patterns(ts, sequences, weights)


class AdaptedSFA:
    '''
    SFA adaptation for Mr-SEQL. This code uses a different alphabet for each Fourier coefficient in the output of SFA.
    '''

    def __init__(self, int N, int w, int a):
        self.sfa = SFA(w, a, N, norm=True, remove_repeat_words=True)

    def fit(self, train_x):
        train_x = detabularize(pd.DataFrame(train_x))
        self.sfa.fit(train_x)

    def timeseries2SFAseq(self, ts):
        dfts = self.sfa._mft(ts)
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
                        break
                first_char += self.sfa.alphabet_size
        return sfa_str

###########################################################################


#########################SEQL wrapper#########################


cdef extern from "seql.h":
    cdef cppclass SEQL:
        SEQL()
        void learn(vector[string] & , vector[double] & )
        double brute_classify(string, double)
        void print_model(int)
        vector[string] get_sequence_features(bool)
        vector[double] get_coefficients(bool)


cdef class PySEQL:
    '''
    Wrapper of SEQL C++ implementation.
    '''

    cdef SEQL * thisptr

    def __cinit__(self):
        self.thisptr = new SEQL()

    def __dealloc__(self):
        del self.thisptr

    def learn(self, vector[string] sequences, vector[double] labels):
        self.thisptr.learn(sequences, labels)

    def classify(self, string sequence):
        scr = self.thisptr.brute_classify(sequence, 0.0)
        return np.array([-scr, scr])  # keep consistent with multiclass case

    def print_model(self):
        self.thisptr.print_model(100)

    def get_sequence_features(self, bool only_positive=False):
        return self.thisptr.get_sequence_features(only_positive)

    def get_coefficients(self, bool only_positive=False):
        return self.thisptr.get_coefficients(only_positive)


class OVASEQL:
    '''
    SEQL in one versus all scenario for multiclass problem.
    '''

    def __init__(self, unique_labels):
        self.labels_ = unique_labels
        self.models = []

    def learn(self, sequences, labels):
        for l in self.labels_:
            tmp_labels = [1 if c == l else - 1 for c in labels]
            m = PySEQL()
            m.learn(sequences, tmp_labels)
            self.models.append(m)

    def classify(self, string sequence):
        scr = []
        for m in self.models:
            scr.append(m.classify(sequence)[1])
        return np.array(scr)

    def get_sequence_features(self, bool only_positive=False):
        sqs = []
        for m in self.models:
            sqs.extend(m.get_sequence_features(True))
        return sqs

    def get_coefficients(self, bool only_positive=False):
        coefs = []
        for m in self.models:
            coefs.extend(m.get_coefficients(True))
        return coefs


###########################################################################


######################### Mr-SEQL (main class) #########################

class MrSEQLClassifier(BaseClassifier):
    ''' Time Series Classification with multiple symbolic representations and SEQL (Mr-SEQL)

    @article{mrseql,
    author = {Le Nguyen, Thach and Gsponer, Severin and Ilie, Iulia and O'reilly, Martin and Ifrim, Georgiana},
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

    def __init__(self, seql_mode='clf', symrep=['sax'], symrepconfig=None):

        self.symbolic_methods = symrep

        if seql_mode in ('fs', 'clf'):
            self.seql_mode = seql_mode
        else:
            raise ValueError('seql_mode should be either clf or fs.')

        if symrepconfig is None:
            self.config = []  # http://effbot.org/zone/default-values.htm
        else:
            self.config = symrepconfig

       
        self.seql_models = []  # seql models

        # all the unique labels in the data
        # in case of binary data the first one is always the negative class
        self.classes_ = []

        self.clf = None  # scikit-learn model

        # store fitted sfa for later transformation
        self.sfas = {}

    def _is_multiclass(self):
        return len(self.classes_) > 2

    
    def _to_tmp_labels(self, y):
        # change arbitrary binary labels to -1, 1 labels as SEQL can only work with -1, 1
        return [1 if l == self.classes_[1] else -1 for l in y]

    def _transform_time_series(self, ts_x):
        multi_tssr = []

        # generate configuration if not predefined
        if not self.config:
            min_ws = 16
            max_ws = ts_x.shape[1]
            pars = [[w, 16, 4]
                    for w in range(min_ws, max_ws, int(np.sqrt(max_ws)))]

            if 'sax' in self.symbolic_methods:
                for p in pars:
                    self.config.append(
                        {'method': 'sax', 'window': p[0], 'word': p[1], 'alphabet': p[2]})

            if 'sfa' in self.symbolic_methods:
                for p in pars:
                    self.config.append(
                        {'method': 'sfa', 'window': p[0], 'word': 8, 'alphabet': p[2]})

        for cfg in self.config:

            tssr = []

            if cfg['method'] == 'sax':  # convert time series to SAX
                ps = PySAX(cfg['window'], cfg['word'], cfg['alphabet'])
                for ts in ts_x:
                    sr = ps.timeseries2SAXseq(ts)
                    tssr.append(sr)

            if cfg['method'] == 'sfa':  # convert time series to SFA
                if (cfg['window'], cfg['word'], cfg['alphabet']) not in self.sfas:
                    sfa = AdaptedSFA(
                        cfg['window'], cfg['word'], cfg['alphabet'])
                    sfa.fit(ts_x)
                    self.sfas[(cfg['window'], cfg['word'],
                               cfg['alphabet'])] = sfa
                for ts in ts_x:
                    sr = self.sfas[(cfg['window'], cfg['word'],
                                    cfg['alphabet'])].timeseries2SFAseq(ts)
                    tssr.append(sr)

            multi_tssr.append(tssr)

        return multi_tssr

    def _fit_binary_problem(self, mr_seqs, labels):
        models = []
        for rep in mr_seqs:
            m = PySEQL()
            m.learn(rep, labels)
            models.append(m)
        return models

    def _fit_multiclass_problem(self, mr_seqs, labels):
        models = []
        for rep in mr_seqs:
            m = OVASEQL(self.classes_)
            m.learn(rep, labels)
            models.append(m)
        return models

    

    def _to_feature_space(self, mr_seqs):
        # compute feature vectors
        full_fm = []

        for rep, model in zip(mr_seqs, self.seql_models):
            seq_features = model.get_sequence_features(False)            
            fm = np.zeros((len(rep), len(seq_features)))

            for i, s in enumerate(rep):
                for j, f in enumerate(seq_features):
                    if f in s:
                        fm[i, j] = 1
            full_fm.append(fm)

        full_fm = np.hstack(full_fm)
        return full_fm

    def _X_check(self, X):
        '''
        Check if X input is correct. Convert X to 2d numpy array.        
        '''
        check_X(X)
        _enforce_X_univariate(X)

        return np.asarray([a.values for a in X.iloc[:, 0]])

    def fit(self, X, y, input_checks=True):
        """
        Fit the model according to the given training time series data.
        Parameters
        ----------
        X : Time series data.
        y : Target vector relative to X.

        Returns
        -------
        self
            Fitted estimator.        
        """
        if input_checks:
            X = self._X_check(X)
            check_y(y)

        # transform time series to multiple symbolic representations
        mr_seqs = self._transform_time_series(X)

        self.classes_ = np.unique(y)  # because sklearn also uses np.unique

        if self._is_multiclass():  # one versus all
            self.seql_models = self._fit_multiclass_problem(mr_seqs, y)
        else:
            temp_labels = self._to_tmp_labels(y)
            self.seql_models = self._fit_binary_problem(mr_seqs, temp_labels)

        # if seql is being used to select features
        # first computing the feature vectors
        # then fit the new data to a logistic regression model
        if self.seql_mode == 'fs':
            train_x = self._to_feature_space(mr_seqs)
            self.clf = LogisticRegression(
                solver='newton-cg', multi_class='multinomial', class_weight='balanced').fit(train_x, y)
            self.classes_ = self.clf.classes_  # shouldn't matter

        return self

    def _compute_proba(self, score):
        return 1.0 / (1.0 + np.exp(-score))

    def predict_proba(self, X, input_checks=True):
        """ 
        If seql_mode is set to 'fs', it returns the estimation by sklearn logistic regression model.
        Otherwise (seql_mode == 'clf'), it returns normalized probability estimated with one-versus-all method.

        Parameters
        ----------
        X : time series data.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        if input_checks:
            X = self._X_check(X)
        mr_seqs = self._transform_time_series(X)

        if self.seql_mode == 'fs':
            test_x = self._to_feature_space(mr_seqs)            
            return self.clf.predict_proba(test_x)

        else:
            scores = np.zeros((len(X), len(self.classes_)))
            for rep, model in zip(mr_seqs, self.seql_models):
                for c, seq in enumerate(rep):
                    scores[c] = scores[c] + model.classify(seq)

            proba = self._compute_proba(scores)
            # https://github.com/scikit-learn/scikit-learn/blob/bf24c7e3d6d768dddbfad3c26bb3f23bc82c0a18/sklearn/linear_model/_base.py#L300
            proba /= proba.sum(axis=1).reshape((proba.shape[0], -1))

            return proba

    def predict(self, X, input_checks=True):
        """
        Predict class labels for samples in X.
        Parameters
        ----------
        X : time series data.
        Returns
        -------
        C : array
            Predicted class label per sample.
        """
        if input_checks:
            X = self._X_check(X)
        proba = self.predict_proba(X, False)
        return np.array([self.classes_[np.argmax(prob)] for prob in proba])

    def map_sax_model(self, ts):
        """    For interpretation.
        Returns vectors of weights with the same length of the input time series.
        The weight of each point implies its contribution in the classification decision regarding the class.

        Parameters
        ----------
        ts : A single time series.

        Returns
        -------
        weighted_ts: ndarray of (number of classes, length of time series)

        Note
        -------
        Only supports SAX features.
        """

        if len(self.symbolic_methods) == 1 and self.symbolic_methods[0] == 'sax' and self.seql_mode == 'fs':

            weighted_ts = np.zeros((len(self.classes_), len(ts)))

            fi = 0
            for cfg, m in zip(self.config, self.seql_models):
                features = m.get_sequence_features()
                if cfg['method'] == 'sax':
                    ps = PySAX(cfg['window'], cfg['word'], cfg['alphabet'])
                    if self._is_multiclass():
                        for ci, cl in enumerate(self.classes_):
                            weighted_ts[ci, :] += ps.map_weighted_patterns(
                                ts, features, self.clf.coef_[ci, fi:(fi+len(features))])
                    else:
                        weighted_ts[0, :] += ps.map_weighted_patterns(
                            ts, features, self.clf.coef_[0, fi:(fi+len(features))])

                fi += len(features)
            if not self._is_multiclass():
                weighted_ts[1, :] = -weighted_ts[0, :]
            return weighted_ts
        else:
            print('The mapping only works on fs mode. In addition, only sax features will be mapped to the time series.')
            return None

    def summary(self):
        """
        Print description.
        """
        print('Symbolic methods: ' + ', '.join(self.symbolic_methods))
        if not self.config:
            print('No symbolic parameters found. To be generated later.')

        if self.seql_mode == 'fs':
            print('Classification Method: SEQL as feature selection')
        elif self.seql_mode == 'clf':
            print('Classification Method: Ensemble SEQL')

    def get_all_sequences(self):
        """
        Get all the subsequences used as features.

        Returns
        -------
        An array of string.

        """
        rt = []
        for m in self.seql_models:
            rt.append([s.decode('ascii') for s in m.get_sequence_features()])
        return rt
