from sklearn.ensemble._forest import ForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble._forest import _generate_sample_indices, compute_sample_weight
import numpy as np
from numpy import random
from sklearn.decomposition import PCA
from scipy.stats import mode
from copy import deepcopy

#they extract random subsections of the data, and perform PCA with variancecoverage of 1.0
#the number of subsections of the data is based on the method,
#generateGroupsFromNumbers or generateGroupsFromSizes.

class RotationForest(ForestClassifier):

    def __init__(self,
                 n_estimators=10,
                 minGroup=3,
                 maxGroup=3,
                 remove_percentage=50,
                 random_state=None,
                 verbose=0,):
        super(RotationForest, self).__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators)

        self.verbose = verbose
        self.random_state = random_state
        self.minGroup = minGroup
        self.maxGroup = maxGroup
        self.remove_percentage = remove_percentage
        self.base_projectionFilter = PCA()
        self.base_estimator = DecisionTreeClassifier()
        self._std = []
        self._mean = []
        self._noise = []
        self._classifiers = []
        self._num_classes = 0
        self._num_atts = 0
        self.classes_ = None

        self._groups = [None] * self.n_estimators
        self._pcas = [None] * self.n_estimators

        random.seed(random_state)


    def fit(self, X, y, sample_weight=None):

        n_samps, self._num_atts = X.shape
        self._num_classes = np.unique(y).shape[0]
        self.classes_ = list(set(y))
        print(self._num_classes)
        print(self._num_atts)

        #need to remove 0 variance attributes.


        # Compute mean, std and noise for z-normalisation
        self._std = np.std(X, axis=0)
        self._med = np.mean(X, axis=0)
        self._noise = [random.uniform(-0.000005, 0.000005) for p in range(0, X.shape[1])] #in case we have 0 std.

        # Normalisae the data
        X = (X - self._med) / (self._std + self._noise)

        classGrouping = self.createClassesGrouping(X,y)
        #print(np.array(classGrouping).shape)


        for i in range(0, self.n_estimators):
            #generate groups.
            self._groups[i] = self.generateGroupsFromSizes(X)
            self._pcas[i] = []

            #construct the slices to fit the PCAs too.
            for j, grp in enumerate(self._groups[i]):
                rot_j = []
                #randomly add the classes
                #with the randomly selected attributes.
                for index, selected_class in enumerate(self.selectClasses()):
                    if not selected_class:
                        continue
                    #we have to reshape the array in case it's a size 1, and gets squeezed.
                    for inst in classGrouping[index]:
                        rot_j.append(inst[grp])
                #randomly sample 50% of the indices.
                sample_ind = random.choice(len(rot_j), int((float(len(rot_j))/100.0) * self.remove_percentage), replace=False)
                rot_j = np.array(rot_j).reshape((-1, grp.shape[0]))
                #only sample if we have lots of instances.
                if sample_ind.shape[0] > 2:
                    rot_j = rot_j[sample_ind]

				##try to fit the PCA if it fails, remake it, and add 10 random data instances.
                while True:
				    #ignore err state on PCA because we account if it fails.
                    with np.errstate(divide='ignore',invalid='ignore'):
                        pca_data = deepcopy(self.base_projectionFilter).fit(rot_j)

                    self.addRandomInstance(rot_j, 10)
                    if not np.isnan(pca_data.explained_variance_ratio_).all():
                        break
                    rot_j = self.addRandomInstance(rot_j, 10)

                self._pcas[i].append(pca_data)

            #merge all the pca_transformed data into one instance and build a classifier on it.
            transformed_X = self.convertData(X, i)
            #with this transformed data, we need to train a tree.
            tree = deepcopy(self.base_estimator)
            tree.fit(transformed_X, y)
            self._classifiers.append(tree)

    def addRandomInstance(self, data, num_instances):
        random_instances = np.random.random_sample((num_instances,data.shape[1]))
        output = np.concatenate([data,random_instances])
        return output

    def convertData(self, X, i):
       return np.array([self.createRow(row,i) for row in X])

    #could project along the axis, but then i couldn't use it for new instances.
    def createRow(self, row, i):
        return np.concatenate([self._pcas[i][j].transform([row[grp]])[0] for j, grp in enumerate(self._groups[i])], axis=0)

    def createClassesGrouping(self, X, Y):
        return np.array([np.squeeze(X[np.argwhere(Y == i)]).reshape((-1,self._num_atts))  for i in self.classes_])

    def selectClasses(self):
        numSelected = 0
        selected = np.zeros(self._num_classes, dtype=np.bool)

        for i in range(0,self._num_classes):
            if random.choice([True,False]):
                selected[i] = True
                numSelected +=1

        if numSelected == 0:
            selected[random.choice(self._num_classes)] = True
        return selected

    def attributes_permuation(self, numAtts, classAtt):
        #arange is no inclusive
        vals = np.concatenate(( np.arange(0, classAtt, 1), np.arange(classAtt, numAtts)))
        return np.random.permutation(vals)

    def generateGroupsFromSizes(self, data):
        # generate one permuation, then do it for all classifiers.

        numGroupsOfSize = np.zeros(self.maxGroup - self.minGroup + 1)
        permutation = self.attributes_permuation(self._num_atts, self._num_classes)

        perm_length = permutation.shape[0]

        #select the size of each group
        numAttributes = 0
        numGroups = 0
        while numAttributes < perm_length:
            n = random.randint(numGroupsOfSize.shape[0])
            numGroupsOfSize[n] += 1
            numAttributes += self.minGroup + n
            numGroups += 1

        currentAttribute =0
        currentSize = 0

        Groups = []
        for j in range(0, numGroups):
            while numGroupsOfSize[currentSize] == 0:
                currentSize +=1
            numGroupsOfSize[currentSize] -=1

            n = self.minGroup + currentSize
            Groups.append(np.zeros(n))
            for k in range(0,n):
                if currentAttribute < perm_length:
                    Groups[j][k] = permutation[currentAttribute]
                else:
                    Groups[j][k] = permutation[random.randint(perm_length)]
                currentAttribute+=1

        return np.array(Groups, dtype=np.int)

    def predict(self, X):
        return [self.classes_[np.argmax(prob)] for prob in self.predict_proba(X)]

    def predict_proba(self, X):
        #need to normalise and remove atts
        self._std = np.std(X, axis=0)
        self._med = np.mean(X, axis=0)
        self._noise = [random.uniform(-0.000005, 0.000005) for p in range(0, X.shape[1])] #in case we have 0 std.

        # Normalise the data
        X = (X - self._med) / (self._std + self._noise)

        sums = np.zeros((X.shape[0],self._num_classes))
        for i, clf in enumerate(self._classifiers):
            transformX = self.convertData(X,i)
            sums += clf.predict_proba(transformX)

        output = sums / (np.ones(self._num_classes) * self.n_estimators)
        return output
