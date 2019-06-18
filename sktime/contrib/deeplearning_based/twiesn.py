# model TWIESN
import numpy as np
from scipy import sparse
#scipy
from scipy.sparse import linalg as slinalg

########   sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd

from sktime.contrib.deeplearning_based.basenetwork import BaseDeepLearner
from sktime.contrib.deeplearning_based.basenetwork import networkTests

#class Classifier_TWIESN:
class TWIESN(BaseDeepLearner):
	def __init__(self, 
					output_directory=None,
					verbose=False,
					dim_to_use=0): 
					
					
		self.output_directory = output_directory
		
		self.verbose = verbose
		
		self.dim_to_use = 0

		# hyperparameters 
		first_config = {'N_x':250,'connect':0.5,'scaleW_in':1.0,'lamda':0.0}
		second_config = {'N_x':250,'connect':0.5,'scaleW_in':2.0,'lamda':0.05}
		third_config = {'N_x':500,'connect':0.1,'scaleW_in':2.0,'lamda':0.05}
		fourth_config = {'N_x':800,'connect':0.1,'scaleW_in':2.0,'lamda':0.05}
		self.configs = [first_config,second_config,third_config,fourth_config]
		self.rho_s = [0.55,0.9,2.0,5.0]
		self.alpha = 0.1 # leaky rate


	def build_model(self, input_shape, nb_classes, **kwargs):
		self.init_matrices()
	
		# construct the riger classifier model
		self.ridge_classifier = Ridge(alpha=self.lamda)
		
		
		
	def evaluate_paramset(self, X, y, val_X, val_y, rho, config):

		#param setting is correct.
		self.rho = rho
		self.N_x = config['N_x']
		self.connect = config['connect']
		self.scaleW_in = config['scaleW_in']
		self.lamda = config['lamda']

		#init transformer based on paras.
		self.init_matrices()
		
		#transformed X
		x_transformed = self.transform_to_feature_space(X)

		new_train_labels = np.repeat(y,self.T,axis=0)

		ridge_classifier = Ridge(alpha=self.lamda)
		ridge_classifier.fit(x_transformed, new_train_labels)


		#transform Validation and labels
		x_val_transformed = self.transform_to_feature_space(val_X)
		new_val_labels = np.repeat(val_y,self.T,axis=0)

		val_preds = ridge_classifier.predict(x_val_transformed)
		
		y_pred_val = self.reshape_prediction(val_preds,val_X.shape[0],self.T)

		#calculate validation accuracy
		#argmax the val_y because it is in onehot encoding.
		return accuracy_score(np.argmax(val_y, axis=1),y_pred_val)
	
	def fit(self, X, y, input_checks = True, **kwargs):
		#check and convert input to a univariate Numpy array
		if isinstance(X, pd.DataFrame):
			if isinstance(X.iloc[0, self.dim_to_use], pd.Series):
				X = np.asarray([a.values for a in X.iloc[:, 0]])
			else:
				raise TypeError(
					"Input should either be a 2d numpy array, or a pandas dataframe containing Series objects")
		if len(X.shape) == 2:
			# add a dimension to make it multivariate with one dimension
			X = X.reshape((X.shape[0], X.shape[1], 1))

		onehot_y = self.convert_y(y)


		self.num_dim = X.shape[2]
		self.T = X.shape[1]
		
		#FINE TUNE MODEL PARAMS
		# split train to validation set to choose best hyper parameters 
		x_train, x_val, y_train, y_val = train_test_split(X,onehot_y, test_size=0.2)
		self.N = x_train.shape[0]

		# limit the hyperparameter search if dataset is too big 
		if x_train.shape[0] > 1000:
			for config in self.configs: 
				config['N_x'] = 100
			self.configs = [self.configs[0],self.configs[1],self.configs[2]]
		
		#search for best hyper parameters
		best_train_acc = -1
		best_rho = -1
		best_config = None
		for idx_config in range(len(self.configs)): 
			for rho in self.rho_s:
				train_acc = self.evaluate_paramset(x_train,
										y_train,
										x_val,
										y_val,
										rho, 
										self.configs[idx_config])

				print(train_acc)
				if best_train_acc < train_acc:
					best_train_acc = train_acc
					best_rho = rho
					best_config = self.configs[idx_config]
		
		self.rho = best_rho
		self.N_x = best_config['N_x']
		self.connect = best_config['connect']
		self.scaleW_in = best_config['scaleW_in']
		self.lamda = best_config['lamda']
		
		#init transformer based on paras.
		self.init_matrices()
		
		#transformed X
		x_transformed = self.transform_to_feature_space(X)

		# transform the corresponding labels
		new_labels = np.repeat(y,X.shape[1],axis=0)

		#create and fit the tuned ridge classifier.
		self.model = Ridge(alpha=self.lamda)
		self.model.fit(x_transformed,new_labels)
		
		
	def predict_proba(self, X, input_checks = True, **kwargs):
		#check input is univariate etc.
		if isinstance(X, pd.DataFrame):
			if isinstance(X.iloc[0, self.dim_to_use], pd.Series):
				X = np.asarray([a.values for a in X.iloc[:, 0]])
			else:
				raise TypeError(
					"Input should either be a 2d numpy array, or a pandas dataframe containing Series objects")

		if len(X.shape) == 2:
			# add a dimension to make it multivariate with one dimension
			X = X.reshape((X.shape[0], X.shape[1], 1))


		#transform and predict prodba on the ridge classifier.
		new_x_test = self.transform_to_feature_space(X)

		#TODO: need to get the probabilities. this is not correct #need to convert but not argmax.
		probas = self.model.predict_proba(new_x_test)

		# reshape so the first axis has the number of instances
		new_y_pred = y_pred.reshape(X.shape[0], X.shape[1], -1)
		# average the predictions of instances
		return np.average(new_y_pred, axis=1)


	def init_matrices(self):
		self.W_in = (2.0*np.random.rand(self.N_x,self.num_dim)-1.0)/(2.0*self.scaleW_in)

		converged = False

		i =0 

		# repeat because could not converge to find eigenvalues 
		while(not converged):
			i+=1

			# generate sparse, uniformly distributed weights
			self.W = sparse.rand(self.N_x,self.N_x,density=self.connect).todense()

			# ensure that the non-zero values are uniformly distributed 
			self.W[np.where(self.W>0)] -= 0.5

			try:
				# get the largest eigenvalue 
				eig, _ = slinalg.eigs(self.W,k=1,which='LM')
				converged = True
			except: 
				print('not converged ',i)
				continue

		# adjust the spectral radius
		self.W /= np.abs(eig)/self.rho

	def compute_state_matrix(self, x_in):
		# number of instances 
		n = x_in.shape[0]
		# the state matrix to be computed
		X_t = np.zeros((n, self.T, self.N_x),dtype=np.float64) 
		# previous state matrix
		X_t_1 = np.zeros((n, self.N_x),dtype=np.float64) 
		# loop through each time step 
		for t in range(self.T):
			# get all the time series data points for the time step t  
			curr_in = x_in[:,t,:]
			# calculate the linear activation 
			curr_state = np.tanh(self.W_in.dot(curr_in.T)+self.W.dot(X_t_1.T)).T
			# apply leakage 
			curr_state = (1-self.alpha)*X_t_1 + self.alpha*curr_state
			# save in previous state 
			X_t_1 = curr_state
			# save in state matrix 
			X_t[:,t,:] = curr_state
				
		return X_t
		
		
	def transform_to_feature_space(self, X):
		# compute the state matrices which is the new feature space  
		state_matrix = self.compute_state_matrix(X)
		# add the input to form the new feature space and transform to 
		# the new feature space to be feeded to the classifier 
		return np.concatenate((X,state_matrix), axis=2).reshape(
			X.shape[0] * self.T , self.num_dim+self.N_x)
			
			
	
	def reshape_prediction(self,y_pred, num_instances,length_series): 
		# reshape so the first axis has the number of instances
		new_y_pred = y_pred.reshape(num_instances,length_series,y_pred.shape[-1])
		# average the predictions of instances 
		new_y_pred = np.average(new_y_pred, axis=1)
		# get the label with maximum prediction over the last label axis 
		new_y_pred = np.argmax(new_y_pred,axis=1)
		return new_y_pred

if __name__ == "__main__":
	networkTests(TWIESN())