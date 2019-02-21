import numpy as np
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sklearn.neighbors.classification import KNeighborsClassifier as KNN
from sklearn.model_selection import GridSearchCV
from sktime.classifiers.example_classifiers import dtw_distance, derivative_dtw_distance, weighted_dtw_distance, weighted_derivative_dtw_distance, lcss_distance, erp_distance, msm_distance


class ElasticEnsemble:

    def __init__(self):
        self.classifiers = [
            KNN(metric=dtw_distance, n_neighbors=1, algorithm="brute"),                         # ED
            KNN(metric=dtw_distance, n_neighbors=1, algorithm="brute"),                         # DTW Full
            KNN(metric=derivative_dtw_distance, n_neighbors=1, algorithm="brute"),              # DDTW Full
            KNN(metric=dtw_distance, n_neighbors=1, algorithm="brute"),                         # DTW CV
            KNN(metric=derivative_dtw_distance, n_neighbors=1, algorithm="brute"),              # DDTW CV
            KNN(metric=weighted_dtw_distance, n_neighbors=1, algorithm="brute"),                # WDTW
            KNN(metric=weighted_derivative_dtw_distance, n_neighbors=1, algorithm="brute"),     # WDDTW
            KNN(metric=lcss_distance, n_neighbors=1, algorithm="brute"),                        # LCSS
            KNN(metric=erp_distance, n_neighbors=1, algorithm="brute"),                         # ERP
            KNN(metric=msm_distance, n_neighbors=1, algorithm="brute"),                         # MSM
        ]

        # note: params not representative of published algorithm - for development only
        self.params_to_search = [
            {'metric_params': [{'window': 0.0}]},                                                               # Euclidean distance
            {'metric_params': [{'window': 1.0}]},                                                               # DTW Full
            {'metric_params': [{'window': 1.0}]},                                                               # DDTW Full
            {'metric_params': [{'window': x / 10} for x in range(0, 10)]},                                      # DTW CV
            {'metric_params': [{'window': x / 10} for x in range(0, 10)]} ,                                     # DDTW CV
            {'metric_params': [{'g': x / 10} for x in range(0, 10)]},                                           # WDTW
            {'metric_params': [{'g': x / 10} for x in range(0, 10)]},                                           # WDDTW
            {'metric_params': [{'delta': d, 'epsilon': e/10} for d in range(3, 6) for e in range(1, 3)]},       # LCSS
            {'metric_params': [{'bandsize': b, 'g': g} for b in range(3, 20, 3) for g in range(0, 2)]},         # ERP
            {'metric_params': [{'c': c} for c in [0.1,0.001,0.0001]]},  # MSM
        ]

        self.cv_accs = np.zeros(len(self.classifiers))
        self.cv_sum = 0

    def fit(self, X, y):
        # get each classifier
        for c in range(0,len(self.classifiers)):
            grid = GridSearchCV(
                estimator=self.classifiers[c],
                param_grid=self.params_to_search[c],
                cv=3,
                scoring='accuracy'
            )
            grid.fit(X,y)
            self.classifiers[c] = grid
            self.cv_accs[c] = grid.best_score_
            print(str(self.classifiers[c].estimator.metric)+": "+str(grid.best_score_))
        self.cv_sum = np.sum(self.cv_accs)

    def predict_proba(self, X):

        output_probs = None
        for c in range(len(self.classifiers)):
            this_probs = self.classifiers[c].predict_proba(X)
            this_probs = this_probs*self.cv_accs[c]

            if output_probs is None:
                output_probs = this_probs
            else:
                output_probs = [[output_probs[x][y] + this_probs[x][y] for y in range(0,len(output_probs[x]))] for x in range(0,len(output_probs))]

        output_probs /= self.cv_sum
        return output_probs

    def predict(self, X):
        probs = self.predict_proba(X)
        labels = self.classifiers[0].classes_
        preds = [labels[np.argmax(probs[x])] for x in range(0,len(probs))]
        return preds


# Example usage with a limited amount of the GunPoint problem
if __name__ == "__main__":
    dataset = "GunPoint"
    train_x, train_y = load_from_tsfile_to_dataframe(file_path="C:/temp/sktime_temp_data/" + dataset + "/", file_name=dataset + "_TRAIN.ts")
    ee = ElasticEnsemble()
    ee.fit(train_x.iloc[0:10], train_y[0:10])
    preds = ee.predict(train_x.iloc[10:15])
    print(preds)
