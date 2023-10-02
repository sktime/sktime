"""Registry of mtypes for Table scitype.

See datatypes._registry for API.
"""

__all__ = [
    "MTYPE_REGISTER_TABLE",
    "MTYPE_LIST_TABLE",
]


MTYPE_REGISTER_TABLE = [
    ("pd_DataFrame_Table", "Table", "pd.DataFrame representation of a data table"),
    ("numpy1D", "Table", "1D np.narray representation of a univariate table"),
    ("numpy2D", "Table", "2D np.narray representation of a univariate table"),
    ("pd_Series_Table", "Table", "pd.Series representation of a data table"),
    ("list_of_dict", "Table", "list of dictionaries with primitive entries"),
]

MTYPE_LIST_TABLE = [x[0] for x in MTYPE_REGISTER_TABLE]
