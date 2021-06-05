# -*- coding: utf-8 -*-
__author__ = ["Toby Hodges", "Afzal Ansari", "Abdul"]

import pandas as pd
from sktime.utils import all_estimators
from importlib import import_module

# creates dataframe as df
colnames = ["Class_Name", "Estimator_Types", "Authors"]
df = pd.DataFrame([], columns=colnames)


def process_author_info(author_info):
    """
    turn a list of author names into a string.
    Multiple author names will be separated by a comma,
    with the final name always preceded by "&"

    """
    if isinstance(author_info, list):
        if len(author_info) > 1:
            return ", ".join(author_info[:-1]) + " & " + author_info[-1]
        else:
            return author_info[0]
    else:
        return author_info


def does_not_start_with_underscore(input_string):
    return not input_string.startswith("_")


for modname, modclass in all_estimators():
    algorithm_type = "::".join(str(modclass).split(".")[1:-2])
    try:
        author_info = process_author_info(modclass.__author__)
    except AttributeError:
        try:
            author_info = process_author_info(
                import_module(modclass.__module__).__author__
            )
        except AttributeError:
            author_info = "no author info"

    # includes part of class string
    modpath = str(modclass)[8:-2]
    path_parts = modpath.split(".")
    # joins strings excluding starting with '_'
    clean_path = ".".join(list(filter(does_not_start_with_underscore, path_parts)))
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
        pd.Series([modname, algorithm_type, author_info], index=colnames),
        ignore_index=True,
    )
    # creates a table in html format
    df.to_html("algorithm_overview.html", index=False, escape=False)
