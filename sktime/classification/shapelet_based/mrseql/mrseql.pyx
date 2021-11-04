# -*- coding: utf-8 -*-
"""MrSEQL Classifier.
"""

# TODO remove in v0.10.0
# the functionality in this file is depreciated and to be replaced with a version
# based on numba.
STUFF = "Hi"  # https://stackoverflow.com/questions/8024805/cython-compiled-c-extension-importerror-dynamic-module-does-not-define-init-fu

import numpy as np

from libcpp.string cimport string
from libcpp.vector cimport vector

from deprecated.sphinx import deprecated
from sklearn.linear_model import LogisticRegression

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.dictionary_based import SFA
from sktime.utils.validation.panel import check_X, check_X_y

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


@deprecated(
    version="0.8.1",
    reason="AdaptedSFA will be removed in v0.10.0",
    category=FutureWarning,
)
class AdaptedSFA:
    """SFA adaptation for Mr-SEQL. This code uses a different alphabet for each
    Fourier coefficient in the output of SFA."""

    def __init__(self, int N, int w, int a):
        self.sfa = SFA(w, a, N, norm=True, remove_repeat_words=True)

    def fit(self, train_x):
        self.sfa.fit(train_x)

    def timeseries2SFAseq(self, ts):
        """Convert time series to SFA sequence."""
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
    """Wrapper of SEQL C++ implementation."""

    cdef SEQL * thisptr

    def __cinit__(self):
        self.thisptr = new SEQL()

    def __dealloc__(self):
        del self.thisptr

    def learn(self, vector[string] sequences, vector[double] labels):
        self.thisptr.learn(sequences, labels)
        return self.thisptr.get_sequence_features(False), self.thisptr.get_coefficients(False)

# TODO: remove in v0.10.0
@deprecated(
    version="0.8.1",
    reason="SEQLCLF will be removed in v0.10.0",
    category=FutureWarning,
)
class SEQLCLF:
    '''
    SEQL with multiple symbolic representations of time series.
    '''

    def __init__(self):
        self.features = []
        self.coefficients = []

    def is_binary(self):
        if len(self.classes_) > 2:
            return False
        return True

    def _fit_binary(self, mr_seqs, labels):
        # labels have to be 1 and -1
        features = []
        coefficients = []
        for rep in mr_seqs:
            m = PySEQL()
            f,c = m.learn(rep, labels)
            features.append(f)
            coefficients.append(c)

        return features, coefficients

    def _reverse_coef(self,coefs):
        return [[-f for f in fs] for fs in coefs]

    def fit(self, mr_seqs, labels):
        self.classes_ = np.unique(labels)
        for ul in self.classes_:
            tmp_labels = [1 if l == ul else -1 for l in labels]
            f,c = self._fit_binary(mr_seqs, tmp_labels)
            self.features.append(f)
            self.coefficients.append(c)
            if self.is_binary():
                self.features.append(f)
                self.coefficients.append(self._reverse_coef(c))
                break


    def predict_proba(self, mr_seqs):
        nrow = len(mr_seqs[0])
        ncol = len(self.classes_)
        scores = np.zeros((nrow, ncol))
        for i in range(0,ncol):
            for rep,fs,cs in zip(mr_seqs,self.features[i], self.coefficients[i]):
                for j in range(0,nrow):
                    for f,c in zip(fs,cs):
                        if f in rep[j]:
                            scores[j,i] += c
            if self.is_binary():
                scores[:,1] = -scores[:,0]
                break
        proba = 1.0 / (1.0 + np.exp(-scores))
        # https://github.com/scikit-learn/scikit-learn/blob/bf24c7e3d6d768dddbfad3c26bb3f23bc82c0a18/sklearn/linear_model/_base.py#L300
        proba /= proba.sum(axis=1).reshape((proba.shape[0], -1))

        return proba

    def get_sequence_features(self):
        if self.is_binary():
            return self.features[0]
        else:
            # select only positive features from each set
            ret_set = []
            for i in range(0,len(self.features[0])):
                pos_f = []
                for j in range(0,len(self.classes_)):
                    for f,c in zip(self.features[j][i], self.coefficients[j][i]):
                        if c > 0:
                            pos_f.append(f)
                ret_set.append(pos_f)
            return ret_set



###########################################################################


######################### Mr-SEQL (main class) #########################

# TODO: remove in v0.10.0
@deprecated(
    version="0.8.1",
    reason="MrSEQLClassifier will be removed in v0.10.0. It will be replaced with an "
           "implementation based on Numba.",
    category=FutureWarning,
)
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

    seql_mode       : str, either 'clf' or 'fs'. In the 'clf' mode, Mr-SEQL is an ensemble of SEQL models while in the 'fs' mode Mr-SEQL trains a logistic regression model with features extracted by SEQL from symbolic representations of time series.

    symrep          : list or tuple, should contains only 'sax' or 'sfa' or both. The symbolic representations to be used to transform the input time series.

    custom_config   : dict, customized parameters for the symbolic transformation. If defined, symrep will be ignored.

    '''

    # Capability tags
    capabilities = {
        "multivariate": True,
        "unequal_length": False,
        "missing_values": False,
        "train_estimate": False,
        "contractable": False,
    }

    def __init__(self, seql_mode='fs', symrep=('sax'), custom_config=None):
        if 'sax' in symrep or 'sfa' in symrep:
            self.symrep = symrep
        else:
            raise ValueError('symrep: only sax and sfa supported.')

        if seql_mode in ('fs', 'clf'):
            self.seql_mode = seql_mode
        else:
            raise ValueError('seql_mode should be either clf or fs.')


        self.custom_config = custom_config
        self.config = self.custom_config
        self.seql_clf = SEQLCLF()  # seql model
        self.ots_clf = None  # scikit-learn model
        # store fitted sfa for later transformation
        self.sfas = {}
        self._is_fitted = False

    def _transform_time_series(self, ts_x):
        multi_tssr = []

        # generate configuration if not predefined
        if not self.config:
            self.config = []
            min_ws = 16
            min_len = max_len = ts_x.shape[2]
            for a in ts_x[:, 0, :]:
                min_len = min(min_len, len(a))
                max_len = max(max_len, len(a))
            max_ws = (min_len + max_len)//2

            if min_ws < max_ws:
                pars = [[w, 16, 4] for w in range(min_ws, max_ws, int(np.sqrt(max_ws)))]
            else:
                pars = [[max_ws, 16, 4]]

            if 'sax' in self.symrep:
                for p in pars:
                    self.config.append(
                        {'method': 'sax', 'window': p[0], 'word': p[1], 'alphabet': p[2]})

            if 'sfa' in self.symrep:
                for p in pars:
                    self.config.append(
                        {'method': 'sfa', 'window': p[0], 'word': 8, 'alphabet': p[2]})


        for cfg in self.config:
            for i in range(ts_x.shape[1]):
                tssr = []

                if cfg['method'] == 'sax':  # convert time series to SAX
                    ps = PySAX(cfg['window'], cfg['word'], cfg['alphabet'])
                    for ts in ts_x[:, i, :]:
                        sr = ps.timeseries2SAXseq(ts)
                        tssr.append(sr)

                if cfg['method'] == 'sfa':  # convert time series to SFA
                    if (cfg['window'], cfg['word'], cfg['alphabet']) not in self.sfas:
                        sfa = AdaptedSFA(
                            cfg['window'], cfg['word'], cfg['alphabet'])
                        sfa.fit(ts_x[:, [i], :])
                        self.sfas[(cfg['window'], cfg['word'],
                                cfg['alphabet'])] = sfa
                    for ts in ts_x[:, i]:
                        sr = self.sfas[(cfg['window'], cfg['word'],
                                        cfg['alphabet'])].timeseries2SFAseq(ts)
                        tssr.append(sr)

                multi_tssr.append(tssr)

        return multi_tssr

    def _to_feature_space(self, mr_seqs):
        # compute feature vectors
        full_fm = []

        for rep, seq_features in zip(mr_seqs, self.sequences):

            fm = np.zeros((len(rep), len(seq_features)))

            for i, s in enumerate(rep):
                for j, f in enumerate(seq_features):
                    if f in s:
                        fm[i, j] = 1
            full_fm.append(fm)

        full_fm = np.hstack(full_fm)
        return full_fm

    def fit(self, X, y):
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
        X, y = check_X_y(X,y, coerce_to_numpy=True)

        # transform time series to multiple symbolic representations
        mr_seqs = self._transform_time_series(X)

        self.seql_clf.fit(mr_seqs,y)
        self.classes_ = self.seql_clf.classes_
        self.sequences = self.seql_clf.get_sequence_features()

        # if seql is being used to select features
        # first computing the feature vectors
        # then fit the new data to a logistic regression model
        if self.seql_mode == 'fs':
            train_x = self._to_feature_space(mr_seqs)
            self.ots_clf = LogisticRegression(
                solver='newton-cg', multi_class='multinomial', class_weight='balanced').fit(train_x, y)
            self.classes_ = self.ots_clf.classes_

        self._is_fitted = True
        return self



    def predict_proba(self, X):
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
        self.check_is_fitted()
        X = check_X(X, coerce_to_numpy=True)
        mr_seqs = self._transform_time_series(X)

        if self.seql_mode == 'fs':
            test_x = self._to_feature_space(mr_seqs)
            return self.ots_clf.predict_proba(test_x)
        else:
            return self.seql_clf.predict_proba(mr_seqs)

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : time series data.

        Returns
        -------
        C : array
            Predicted class label per sample.
        """
        proba = self.predict_proba(X)
        return np.array([self.classes_[np.argmax(prob)] for prob in proba])

    def map_sax_model(self, ts):
        """For interpretation.

        Returns vectors of weights with the same length of the input time series.
        The weight of each point implies its contribution in the classification decision regarding the class.

        Parameters
        ----------
        ts : A single time series.

        Returns
        -------
        weighted_ts: ndarray of (number of classes, length of time series)

        Notes
        -----
        Only supports univariate time series and SAX features.
        """
        self.check_is_fitted()

        is_multiclass = len(self.classes_) > 2

        if self.seql_mode == 'fs':

            weighted_ts = np.zeros((len(self.classes_), len(ts)))

            fi = 0
            for cfg, features in zip(self.config, self.sequences):
                if cfg['method'] == 'sax':
                    ps = PySAX(cfg['window'], cfg['word'], cfg['alphabet'])
                    if is_multiclass:
                        for ci, cl in enumerate(self.classes_):
                            weighted_ts[ci, :] += ps.map_weighted_patterns(
                                ts, features, self.ots_clf.coef_[ci, fi:(fi+len(features))])
                    else:
                        weighted_ts[0, :] += ps.map_weighted_patterns(
                            ts, features, self.ots_clf.coef_[0, fi:(fi+len(features))])

                fi += len(features)
            if not is_multiclass:
                weighted_ts[1, :] = -weighted_ts[0, :]
            return weighted_ts
        else:
            print('The mapping only works on fs mode. In addition, only sax features will be mapped to the time series.')
            return None
