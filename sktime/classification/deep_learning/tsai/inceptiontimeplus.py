"""InceptionTime+ classifier via tsai."""

__author__ = ["oguiza", "obaidsafi51"]
__all__ = ["InceptionTimeplusClassifier"]


from sktime.classification.deep_learning.tsai._base import BaseTsaiClassifier

class InceptionTimePlusClassifier(BaseTsaiClassifier):
    """InceptionTime+ for time series classification, via tsai.

    Wrap tsai's InceptionTimePlus architecture [1] _ behind the sktime classifier interface.

    Parameters 
    ----------
    n_epochs : int, default = 16
    batch_size : int , default = 16
    lr : float, default = 0.001
    nf : int , default = 32
        Number of filters per convolutional layer.
    depth : int , default =6
        Number of Inception modules.
    valid_size : float , default = 0.2
    random_state : int or None , default = None 
    verbose :bool, default = False

    References 
    ----------

    ..[1] Ismail Fawas et al. , InceptionTime: Finding AlextNet for Time Series Classification,
    Data Mining and knowledge Discovery, 2020.

        Examples
    --------
    >>> from sktime.classification.deep_learning.tsai.inceptiontimeplus import (
    ...     InceptionTimePlusClassifier
    ... )
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = InceptionTimePlusClassifier(n_epochs=1, batch_size=4)  # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    InceptionTimePlusClassifier(...)
    """
    

    _tags = {
        "authors" : ["oguiza", "agolinski"],
        "maintainers" : ["obaidsafi51"]
    }

    def __init__(
        self,
        n_epochs = 16,
        batch_size = 16,
        lr = 0.001,
        nf = 32,
        depth = 6,
        valid_size = 0.2,
        random_state = None,
        verbose = False,
        
    ):
        self.nf = nf
        self.depth = depth
        super().__init__(
            n_epochs= n_epochs,
            batch_size= batch_size,
            lr=lr,
            valid_size=valid_size,
            random_state=random_state,
            verbose=verbose,
        )

    def _build_model(self, n_vars, n_classes):
        from tsai.models.InceptionTimePlus import InceptionTimePlus
        return InceptionTimePlus(n_vars, n_classes, nf = self.nf, depth= self.depth)
    
    def _get_arch_config(self): 
        return {"nf" : self.nf, "depth" : self.depth}
    

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        
        param1 = {
            "n_epochs": 1,
            "batch_size": 4,
            "nf": 8,
            "depth": 2,
        }
        param2 = {
            "n_epochs": 1,
            "batch_size": 4,
            "nf": 16,
            "depth": 3,
            "lr": 1e-4,
        }
        return [param1, param2]