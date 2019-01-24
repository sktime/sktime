'''
classes and functions for model validation
'''
import sklearn

class GridSearchCV(sklearn.model_selection.GridSearchCV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.scoring is None:
            raise AttributeError('supply an external scorer for GridSearchCV')
