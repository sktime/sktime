from sktime.datatypes._series import SeriesMtype
from sktime.datatypes._panel import PanelMtype


class Mtype:
    """Mtype class containing valid types.

    Attributes
    -----------
    Series: Enum
        Attribute containing the valid Series mtypes.
    Panel: Enum
        Attribute containing the valid Panel mtypes.
    """
    Series = SeriesMtype
    Panel = PanelMtype

    @staticmethod
    def get_scitypes():
        """Method to get all valid scitypes.

        Returns
        -------
        list
            List containing the valid scitypes
        """
        temp = []
        for attr in dir(Mtype()):
            if not attr.startswith("__") and attr[0].isupper():
                temp.append(attr)
        return temp

    @staticmethod
    def get_mtypes():
        """Get list of valid mtypes.

        Returns
        -------
        dict
            Dict where the scitype is the key and the value is a list of
            valid mtypes
        """
        scitypes = Mtype.get_scitypes()
        temp = {}

        for type in scitypes:
            curr = [str(val) for val in getattr(Mtype(), type)]
            temp[type] = curr
        return temp
