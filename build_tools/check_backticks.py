#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Test script to check for invalid use of single-backticks."""

__author__ = ["geetu040"]

import ast
import os
import re


def find_py_files(folder_path):
    """Find all Python files in a given folder path."""
    py_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files


def extract_docstrings(filename):
    """Extract docstrings from a Python file."""
    with open(filename) as file:
        tree = ast.parse(file.read(), filename=filename)

    docstrings = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Str)
            ):
                lineno = 0 if isinstance(node, ast.Module) else node.lineno
                docstring = node.body[0].value.s
                docstrings[lineno] = docstring
    return docstrings


def find_invalid_backtick_text(docstring):
    """Find invalid backtick text in a docstring."""
    # remove all the double-backticks
    doc = docstring.replace("``", "")

    all_backtick_text = re.findall(r"`.*?`", doc)
    valid_backtick_text = re.findall(r":.*?:(`.*?`)", doc)

    # find all the invalid backtick code snippets
    invalid_backtick_text = []
    for text in all_backtick_text:
        if text in valid_backtick_text:
            continue
        if text in invalid_backtick_text:
            continue
        invalid_backtick_text.append(text)

    return invalid_backtick_text


def main():
    """Execute the main function of the script."""
    import sktime

    results = {}

    # list all the python files in the project
    py_files = find_py_files(sktime.__path__[0])

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
    if len(results) > 0:
        print(f"Total Files with invalid backticks: {len(results)}")  # noqa
        for filename, result in results.items():
            for lineno, errors in result.items():
                print(f"{filename}:{lineno} {' '.join(errors)}")  # noqa
    else:
        print("No invalid backticks found")  # noqa


if __name__ == "__main__":
    main()
