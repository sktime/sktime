#utilities
import numpy as np
import pandas as pd
import math
from sktime.utils.validation.series_as_features import check_X,check_X_y
from sktime.utils.load_data import load_from_tsfile_to_dataframe as load_ts
from sktime.utils.data_container import tabularize

#Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

#Transforms
from sktime.contrib.shape_dtw.transformers.SubsequenceTransformer import SubsequenceTransformer
from sktime.contrib.shape_dtw.transformers._paa_multivariate import PAA_Multivariate
from sktime.contrib.shape_dtw.transformers.DWT import DWT
from sktime.contrib.shape_dtw.transformers.Slope import Slope
from sktime.contrib.shape_dtw.transformers.Derivative import Derivative
from sktime.contrib.shape_dtw.transformers.HOG1D import HOG1D

#Classifiers
from sktime.classification.base import BaseClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

#Write ShapeDTW as a classifier (that implements BaseClassifier).
#Wrapper for KNeighborsTimeSeriesClassifier that uses DTW as a distance metric

class ShapeDTW(BaseClassifier):

    """
    Parameters
    ----------
    n_neighbours                : int, int, set k for knn (default =1).
    subsequence_length          : int, defines the length of the subsequences (default=sqrt(num_atts)).
    shape_descriptor_function   : string, defines the function to describe the set of subsequences (default = 'raw').
    
    
    The possible shape descriptor functions are as follows:
    
        - 'raw'                 : use the raw subsequence as the shape descriptor function.
                                - params = None
                                
        - 'paa'                 : use PAA as the shape descriptor function.
                                - params = num_intervals_paa (default=8)
        
        - 'dwt'                 : use DWT (Discrete Wavelet Transform) as the shape descriptor function.
                                - params = num_levels_dwt (default=3)
        
        - 'slope'               : use the gradient of each subsequence fitted by a total least squares regression as the shape descriptor function.
                                - params = num_intervals_slope (default=8)
        
        - 'derivative'          : use the derivative of each subsequence as the shape descriptor function.
                                - params = None
        
        - 'hog1d'               : use a histogram of gradients in one dimension as the shape desciptor function.
                                - params = num_intervals_hog1d (defualt=2)
                                         = num_bins_hod1d (default=8)
                                         = scaling_factor_hog1d (default=0.1)
        
        - 'compound'            : use a combination of two shape descriptors simultaneously.
                                - params = weighting_factor (default=None)
                                           Defines how to adjust values of a shape descriptor such that the final vectors become (shape_desc1,weighting_factor*shape_desc2)
                                           If a value is not given, then the appropriate weighting_factor value is tuned by 10-fold cross-validation
                                           on the training data.
                                
        
    shape_descriptor_functions  : string list, only applicable when the shape_descriptor_function is set to 'compound'. Use a list of
                                  shape descriptor functions at the same time.
                                  
    metric_params               : dictionary for metric parameters (default = None).
    """
    def __init__(self,n_neighbors=1,subsequence_length=None,shape_descriptor_function='raw',shape_descriptor_functions=None,metric_params=None):
        self.n_neighbors=n_neighbors
        self.subsequence_length=subsequence_length
        self.shape_descriptor_function=shape_descriptor_function
        self.shape_descriptor_functions=shape_descriptor_functions
        self.metric_params=metric_params
        super(ShapeDTW, self).__init__()
        
    """
    Parameters
    ----------
    X - training data.
    y - list of class labels.
    
    Returns
    -------
    self : object
    """
    def fit(self, X, y):
        #Perform preprocessing on params.
        self.shape_descriptor_function=self.shape_descriptor_function.lower()
        #Convert all strings in list to lowercase.
        if self.shape_descriptor_functions is not None:
            self.shape_descriptor_functions=[x.lower() for x in self.shape_descriptor_functions]
        else:
            self.shape_descriptor_functions = None
        
        X,y= check_X_y(X,y,enforce_univariate=False)
        
        #If the shape descriptor is 'compound', calculate the appropriate weighting_factor
        if self.shape_descriptor_function == "compound":
            self.calculateWeightingFactorValue(X,y)
        
        self.trainData = X
        self.trainDataClasses = y
        
        #get the number of attributes and instances
        num_atts = self.trainData.iloc[0,0].shape[0]
        num_insts = self.trainData.shape[0]
        
        #If a subsequence length is not given, then set it to sqrt(num_atts)
        if self.subsequence_length is None:
            self.subsequence_length=math.floor(math.sqrt(num_atts))
            
        #Convert training data into a list of subsequences
        st = SubsequenceTransformer(self.subsequence_length)
        st.fit(self.trainData)
        self.sequences = st.transform(self.trainData)
        
        self.trainData = self.sequences
        
        #Create the training data by finding the shape descriptors
        self.trainData = self.generateShapeDescriptors(self.sequences,num_insts,num_atts)
        
        #Fit the kNN classifier
        self.knn = KNeighborsTimeSeriesClassifier(self.n_neighbors)
        self.knn.fit(self.trainData,self.trainDataClasses)
        self.classes_ = self.knn.classes_
        
        return self 
        
    """
    Helper function for calculating the appropriate weighting_factor for the compound shape descriptor.
    If a value is given, the weighting_factor is set as the given value. If not, its tuned via
    a 10-fold cross-validation on the training data.
    """
    def calculateWeightingFactorValue(self,X,y):
    
        if self.metric_params is None:
            self.metric_params={}
        
        self.metric_params = {k.lower(): v for k, v in self.metric_params.items()}
            
        #Get the weighting_factor if one is provided
        if self.metric_params.get("weighting_factor") is not None:
            self.weighting_factor=self.metric_params.get("weighting_factor")
        else:
            #Tune it otherwise
            self._param_matrix = {
                'metric_params': [{'weighting_factor': 0.1},
                                  {'weighting_factor': 0.125},
                                  {'weighting_factor': (1/6)},
                                  {'weighting_factor': 0.25},
                                  {'weighting_factor': 0.5},
                                  {'weighting_factor': 1},
                                  {'weighting_factor': 2},
                                  {'weighting_factor': 4},
                                  {'weighting_factor': 6},
                                  {'weighting_factor': 8},
                                  {'weighting_factor': 10}]}
                                  
            grid = GridSearchCV(
                estimator=ShapeDTW(n_neighbors=self.n_neighbors,
                                   subsequence_length=self.subsequence_length,
                                   shape_descriptor_function=self.shape_descriptor_function,
                                   shape_descriptor_functions=self.shape_descriptor_functions,
                                   metric_params=self.metric_params),
                param_grid=self._param_matrix,
                cv=KFold(n_splits=10,shuffle=True),
                scoring='accuracy'
            )
            grid.fit(X,y)
            self.weighting_factor = grid.best_params_['metric_params']['weighting_factor']
            
        

    """
    Find probability estimates for each class for all cases in X.
    
    Parameters
    ----------
    X : The testing input samples. array-like or sparse matrix of shape = [n_test_instances, series_length]
        If a Pandas data frame is passed (sktime format) a check is performed that it only has one column.
        If not, an exception is thrown, since this classifier does not yet have
        multivariate capability.

    Returns
    -------
    output : numpy array of shape = [n_test_instances, num_classes] of probabilities
    """
    def predict_proba(self, X):
        X= check_X(X,enforce_univariate=False)
        
        self.testData = X
        
        #get the number of attributes and instances
        num_atts = self.testData.shape[1]
        num_insts = self.testData.shape[0]
        
        #Convert testing data into a list of subsequences
        st = SubsequenceTransformer(self.subsequence_length)
        st.fit(self.testData)
        self.sequences = st.transform(self.testData)
        
        self.testData = self.sequences
        
        #Create the testing data by finding the shape descriptors
        self.testData = self.generateShapeDescriptors(self.sequences,num_insts,num_atts)
        
        #Classify the test data
        return self.knn.predict_proba(self.testData)


    """
    Find predictions for all cases in X. Could do a wrap function for predict_proba, but this will do for now.
    ----------
    X : The testing input samples. array-like or pandas data frame.
    If a Pandas data frame is passed, a check is performed that it only has one column.
    If not, an exception is thrown, since this classifier does not yet have
    multivariate capability.

    Returns
    -------
    output : numpy array of shape = [n_test_instances]
    """
    def predict(self, X):
        X = check_X(X,enforce_univariate=False)
        self.testData = X
        
        #get the number of attributes and instances
        num_atts = self.testData.shape[1]
        num_insts = self.testData.shape[0]
        
        #Convert testing data into a list of subsequences
        st = SubsequenceTransformer(self.subsequence_length)
        st.fit(self.testData)
        self.sequences = st.transform(self.testData)
        
        self.testData = self.sequences
        
        #Create the testing data by finding the shape descriptors
        self.testData = self.generateShapeDescriptors(self.sequences,num_insts,num_atts)

        #Classify the test data
        return self.knn.predict(self.testData)
        
    """
    This function is used to convert a list of subsequences into a list of shape descriptors to be used for classification.
    """
    def generateShapeDescriptors(self,data,num_insts,num_atts):
    
        #Get the appropriate transformer objects
        if self.shape_descriptor_function != "compound":
            self.transformer = [self.getTransformer(self.shape_descriptor_function)]
        else:
            self.transformer = []
            for x in self.shape_descriptor_functions:
                self.transformer.append(self.getTransformer(x))
            #Compound only supports 2 shape descriptor functions 
            if not (len(self.transformer)==2):
                raise ValueError("When using 'compound', shape_descriptor_functions must be a string array of length 2.")
        
        #To hold the result of each transformer
        dataFrames = []
        col_names = [x for x in range(len(data.columns))]
        
        #Apply each transformer on the set of subsequences
        for t in self.transformer:
            if t is None:
                #Do no transformations
                dataFrames.append(data)
            else:
                #Do the transformation and extract the resulting data frame.
                t.fit(data)
                newData = t.transform(data)
                dataFrames.append(newData)
                
        #Combine the arrays into one dataframe
        if self.shape_descriptor_function == "compound":
            result = combineDataFrames(dataFrames,self.weighting_factor,col_names)
        else:    
            result = dataFrames[0]
            result.columns=col_names
            
        return result
        
        
    """
    Function to extract the appropriate transformer
    
    Returns
    -------
    output : Base Transformer object corresponding to the class (or classes if its a compound transformer) of the
             required transformer. The transformer is configured with the parameters given in self.metric_params.
             
    throws : ValueError if a shape descriptor doesn't exist.
    """
    def getTransformer(self,tName):
        parameters = self.metric_params
    
        if tName == "raw":
            return None
        elif tName == "paa":
            num_intervals = parameters.get("num_intervals_paa")
            if num_intervals is None:
                return PAA_Multivariate()
            return PAA_Multivariate(num_intervals)
        elif tName == "dwt":
            num_levels = parameters.get("num_levels_dwt")
            if num_levels is None:
                return DWT()
            return DWT(num_levels)
        elif tName == "slope":
            num_intervals = parameters.get("num_intervals_slope")
            if num_intervals is None:
                return Slope()
            return Slope(num_intervals)
        elif tName == "derivative":
            return Derivative()
        elif tName == "hog1d":
            num_intervals = parameters.get("num_intervals_hog1d")
            num_bins = parameters.get("num_bins_hog1d")
            scaling_factor = parameters.get("scaling_factor_hog1d")
            
            #All 3 paramaters are None
            if num_intervals is None and num_bins is None and scaling_factor is None:
                return HOG1D()
                
            #2 parameters are None
            if num_intervals is None and num_bins is None:
                return HOG1D(scaling_factor=scaling_factor)
            if num_intervals is None and scaling_factor is None:
                return HOG1D(num_bins=num_bins)
            if num_bins is None and scaling_factor is None:
                return HOG1D(num_intervals=num_intervals)
                
            #1 parameter is None
            if num_intervals is None:
                return HOG1D(scaling_factor=scaling_factor,num_bins=num_bins)
            if scaling_factor is None:
                return HOG1D(num_intervals=num_intervals,num_bins=num_bins)
            if num_bins is None:
                return HOG1D(scaling_factor=scaling_factor,num_intervals=num_intervals)
                
            #All parameters are given
            return HOG1D(num_intervals=num_intervals,num_bins=num_bins,scaling_factor=scaling_factor)
        else:
            raise ValueError("Invalid shape desciptor function.")
    
"""
Helper function for the shape_dtw class to combine two dataframes together into a single dataframe.
Used when the shape_descriptor_function is set to "compound".
"""
def combineDataFrames(dataFrames,weighting_factor,col_names):
    first_desc=dataFrames[0]
    second_desc=dataFrames[1]
    
    first_desc_array = []
    second_desc_array = []
    
    # Convert the dataframes into arrays
    for x in first_desc.columns:
        first_desc_array.append(tabularize(first_desc[x],return_array=True))
        
    for x in second_desc.columns:
        second_desc_array.append(tabularize(second_desc[x],return_array=True))
    
    # Concatenate the arrays together
    res = []
    for x in range(len(first_desc_array)):
        dim1 = []
        for y in range(len(first_desc_array[x])):
            dim2 = []
            dim2.extend(first_desc_array[x][y])
            dim2.extend(second_desc_array[x][y]*weighting_factor)
            dim1.append(dim2)
        res.append(dim1)
        
    res = np.asarray(res)
        
    # Convert to pandas dataframe
    df = pd.DataFrame()
    
    for col in col_names:
        colToAdd = []
        for row in range(len(res[col])):
            inst = res[col][row]
            colToAdd.append(pd.Series(inst)) 
        df[col] = colToAdd
        
    return df
 
if __name__ == "__main__":
    #trainPath="C:\\Users\\Vince\\Documents\\Dissertation Repositories\\datasets\\Test\\Test\\Test_TRAIN.ts"
    #testPath="C:\\Users\\Vince\\Documents\\Dissertation Repositories\\datasets\\Test\\Test\\Test_TEST.ts"
    
    trainPath="C:\\Users\\Vince\\Documents\\Dissertation Repositories\\datasets\\Univariate2018_ts\\Chinatown\\Chinatown_TRAIN.ts"
    testPath="C:\\Users\\Vince\\Documents\\Dissertation Repositories\\datasets\\Univariate2018_ts\\Chinatown\\Chinatown_TEST.ts"
    
    #trainPath="C:\\Users\\Vince\\Documents\\Dissertation Repositories\\datasets\\Multivariate2018_ts\\AtrialFibrillation\\AtrialFibrillation_TRAIN.ts"
    #testPath="C:\\Users\\Vince\\Documents\\Dissertation Repositories\\datasets\\Multivariate2018_ts\\AtrialFibrillation\\AtrialFibrillation_TEST.ts"
    
    trainData,trainDataClasses =  load_ts(trainPath)
    testData,testDataClasses =  load_ts(testPath)
    
    shp = ShapeDTW(n_neighbors=1,subsequence_length=10,shape_descriptor_function="compound",shape_descriptor_functions=["raw","hog1d"],metric_params={"num_intervals_hog1d":2,"num_bins_hog1d":8,"scaling_factor_hog1d":0.1,"num_levels_dwt":3,"weighting_factor":None})
    shp.fit(trainData,trainDataClasses)
    print(shp.score(testData,testDataClasses))