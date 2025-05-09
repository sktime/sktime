#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Test script to check for invalid use of single-backticks."""

__author__ = ["geetu040"]

import argparse
import ast
import glob
import re


def find_py_files(folder_path):
    """Find all Python files in a given folder path."""
    return glob.glob(f"{folder_path}/**/*.py", recursive=True)


def extract_docstrings(filename):
    """Extract docstrings from a Python file."""
    # create abstract syntax tree from the file
    with open(filename) as f:
        tree = ast.parse(f.read())

    # walk through all nodes in the tree
    docstrings = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            # if the node is an expression and
            # its value is a constant and
            # constant's value is a string
            # the node represents a docstring
            # See https://docs.python.org/3/library/ast.html#abstract-grammar
            docstring = node.value.value
            lineno = node.value.lineno
            docstrings[lineno] = docstring

    return docstrings


def find_invalid_backtick_text(docstring):
    """Find invalid backtick text in a docstring."""
    # remove all multiple backticks to avoid interference
    # we are looking only for invalid single-backtick
    docstring = re.sub(r"`{2,}.*?`{2,}", "", docstring, flags=re.DOTALL)

    all_backtick_text = re.findall(r"`.*?`", docstring, flags=re.DOTALL)
    # expressions like :math:`d(x, y):= (x-y)^2` are valid cases
    valid_backtick_text = re.findall(r":.*?:(`.*?`)", docstring, flags=re.DOTALL)

    # find all the invalid backtick code snippets
    invalid_backtick_text = set()
    for text in all_backtick_text:
        if text in valid_backtick_text:
            continue
        # rst hyperlinks are valid cases
        if re.match(r"`.*?<http.*?>`", text, flags=re.DOTALL):
            continue
        invalid_backtick_text.add(text)

    return invalid_backtick_text


def main():
    """Execute the main function of the script."""
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test script to check for invalid use of single-backticks."
    )
    parser.add_argument(
        "folder_path",
        nargs="?",
        default="./sktime",
        help="Folder path to search for Python files",
    )
    args = parser.parse_args()

    folder_path = args.folder_path
    results = {}

    # list all the python files in the project
    py_files = find_py_files(folder_path)

    for file in py_files:
        docstrings = extract_docstrings(file)
        results_on_file = {}

        for lineno, docstring in docstrings.items():
            invalid_backtick_text = find_invalid_backtick_text(docstring)

            if len(invalid_backtick_text) > 0:
                results_on_file[lineno] = invalid_backtick_text

        if len(results_on_file) > 0:
            results[file] = results_on_file

    # print the lines along with the invalid backticks text
    print(f'Results in "{folder_path}"')  # noqa: T201
    if len(results) > 0:
        print(f"Total Files with invalid backticks: {len(results)}")  # noqa: T201
        for filename, result in results.items():
            for lineno, errors in result.items():
                print(f"{filename}:{lineno} {' '.join(errors)}")  # noqa: T201
    else:
        print("No invalid backticks found")  # noqa: T201


if __name__ == "__main__":
    """
    Usage: defaults to "./sktime"
    python build_tools/check_backticks.py

    Usage: folder path as argument
    python build_tools/check_backticks.py sktime/classification/distance_based
    """
    main()
