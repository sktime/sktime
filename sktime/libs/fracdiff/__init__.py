from .fdiff import fdiff, fdiff_coef
from .sklearn.fracdiff import Fracdiff
#from .sklearn.fracdiff import Fracdiff as _FracdiffSklearn
# from .sklearn.fracdiffstat import FracdiffStat as _FracdiffStatSklearn


#class Fracdiff(_FracdiffSklearn):
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#        raise DeprecationWarning(
#            "Fracdiff has been moved to `fracdiff.sklearn`. "
#            "Please import it as:\n"
#            "from fracdiff.sklearn import Fracdiff"
#        )


#class FracdiffStat(_FracdiffStatSklearn):
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#        raise DeprecationWarning(
#            "FracdiffStat has been moved to `fracdiff.sklearn`. "
#            "Please import it as:\n"
#            "from fracdiff.sklearn import FracdiffStat"
#        )
