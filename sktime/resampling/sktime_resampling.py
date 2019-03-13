from abc import ABC, abstractmethod

class SKtime_resampling:
    """
    Abstact class that all MLaut resampling strategies should inherint from
    """
    @abstractmethod
    def resample(self):
        """
        
        """