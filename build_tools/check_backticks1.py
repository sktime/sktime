#!/usr/bin/env python3
"""
Utility script to detect invalid use of single backticks in Python docstrings.

Valid uses include:
    - RST roles such as :math:`x^2`
    - Hyperlinks like `text <http://example.com>`
    - Double-backtick inline code blocks: ``code``

Anything else wrapped in single backticks will be reported.
"""

import argparse
import ast
import re
from pathlib import Path


def find_py_files(folder_path):
    """
    Recursively yield all Python files inside the given folder.
    Using pathlib keeps the code clean and OS-independent.
    """
    return Path(folder_path).rglob("*.py")


def extract_docstrings(filename):
    """
    Extract *actual* docstrings from a file.

    We only consider:
        - module docstrings
        - class docstrings
        - function / async function docstrings

    This avoids picking up random string literals that are not docstrings.
    """
    docstrings = {}

    try:
        # Read file safely with UTF-8 encoding
        source = Path(filename).read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        # Skip files that cannot be parsed
        return docstrings

    for node in ast.walk(tree):
        # Only these node types can legally contain docstrings
        if isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            docstring = ast.get_docstring(node)
            if docstring:
                # lineno gives the line number where the definition starts
                docstrings[node.lineno] = docstring

    return docstrings


def find_invalid_backticks(docstring):
    """
    Identify invalid single-backtick usage inside a docstring.

    Strategy:
        1. Remove known valid patterns.
        2. Search for remaining single-backtick segments.
    """

    cleaned = docstring

    # Remove RST roles like :math:`...` or :class:`...`
    cleaned = re.sub(r":[a-zA-Z_]+:`.*?`", "", cleaned, flags=re.DOTALL)

    # Remove inline hyperlinks: `text <http://example.com>`
    cleaned = re.sub(r"`[^`]*?<http[^`]*?>`", "", cleaned)

    # Remove double-backtick inline code blocks: ``code``
    cleaned = re.sub(r"``.*?``", "", cleaned, flags=re.DOTALL)

    # Now capture remaining single-backtick patterns
    invalid = re.findall(r"`[^`]+`", cleaned)

    return set(invalid)


def main():
    """
    Entry point of the script.
    Parses arguments, scans files, and prints results.
    """
    parser = argparse.ArgumentParser(
        description="Check Python docstrings for invalid single-backtick usage."
    )

    parser.add_argument(
        "folder_path",
        nargs="?",
        default="./sktime",
        help="Folder to scan (defaults to ./sktime)",
    )

    args = parser.parse_args()
    folder_path = args.folder_path

    results = {}

    for file_path in find_py_files(folder_path):
        docstrings = extract_docstrings(file_path)
        file_issues = {}

        for lineno, docstring in docstrings.items():
            invalid_segments = find_invalid_backticks(docstring)

            if invalid_segments:
                file_issues[lineno] = invalid_segments

        if file_issues:
            results[str(file_path)] = file_issues

    # Output section
    print(f'Results in "{folder_path}"')

    if not results:
        print("No invalid backticks found.")
        return

    print(f"Total files with invalid backticks: {len(results)}\n")

    for filename, issues in results.items():
        for lineno, segments in issues.items():
            joined = " ".join(segments)
            print(f"{filename}:{lineno} {joined}")


if __name__ == "__main__":
    """
    Example usage:

        python check_backticks.py
        python check_backticks.py path/to/project
    """
    main()