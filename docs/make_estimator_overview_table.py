# -*- coding: utf-8 -*-
__author__ = ["Toby Hodges", "Afzal Ansari", "Abdul"]

import pandas as pd
from sktime.utils import all_estimators
from importlib import import_module

# creates dataframe as df
COLNAMES = ["Class_Name", "Estimator_Types", "Authors"]
df = pd.DataFrame([], columns=COLNAMES)


def _process_author_info(author_info):
    """
    Process author information from source code files.

    Parameters
    ----------
    author_info : str
        Author information string from source code files.

    Returns
    -------
    author_info : str
        Preprocessed author information.

    Notes
    -----
    A list of author names is turned into a string.
    Multiple author names will be separated by a comma,
    with the final name always preceded by "&".
    """
    if isinstance(author_info, list):
        if len(author_info) > 1:
            return ", ".join(author_info[:-1]) + " & " + author_info[-1]
        else:
            return author_info[0]
    else:
        return author_info


def _does_not_start_with_underscore(input_string):
    return not input_string.startswith("_")


for modname, modclass in all_estimators():
    algorithm_type = "::".join(str(modclass).split(".")[1:-2])
    try:
        author_info = _process_author_info(modclass.__author__)
    except AttributeError:
        try:
            author_info = _process_author_info(
                import_module(modclass.__module__).__author__
            )
        except AttributeError:
            author_info = "no author info"

    # includes part of class string
    modpath = str(modclass)[8:-2]
    path_parts = modpath.split(".")
    # joins strings excluding starting with '_'
    clean_path = ".".join(list(filter(_does_not_start_with_underscore, path_parts)))
    # adds html link reference
    modname = str(
        '<a href="https://www.sktime.org/en/latest/api_reference/modules'
        + "/auto_generated/"
        + clean_path
        + '.html">'
        + modname
        + "</a>"
    )

    df = df.append(
        pd.Series([modname, algorithm_type, author_info], index=COLNAMES),
        ignore_index=True,
    )
# creates a table in html format
df.to_html(
    "./source/estimator_overview/estimator_overview_table.html",
    index=False,
    escape=False,
)
