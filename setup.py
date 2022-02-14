#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Install script for sktime."""

__author__ = ["Markus Löning", "lmmentel"]

import codecs

import toml
from setuptools import find_packages, setup

pyproject = toml.load("pyproject.toml")


def long_description():
    """Read and return README as long description."""
    with codecs.open("README.md", encoding="utf-8-sig") as f:
        return f.read()


# ground truth package metadata is loaded from pyproject.toml
# for context see:
#   - [PEP 621 -- Storing project metadata in pyproject.toml]
#     (https://www.python.org/dev/peps/pep-0621)
pyproject = toml.load("pyproject.toml")


def setup_package():
    """Set up package."""
    setup(
        author_email=pyproject["project"]["authors"][0]["email"],
        author=pyproject["project"]["authors"][0]["name"],
        classifiers=pyproject["project"]["classifiers"],
        description=pyproject["project"]["description"],
        download_url=pyproject["project"]["urls"]["download"],
        extras_require=pyproject["project"]["optional-dependencies"],
        include_package_data=True,
        install_requires=pyproject["project"]["dependencies"],
        keywords=pyproject["project"]["keywords"],
        license=pyproject["project"]["license"],
        long_description=long_description(),
        maintainer_email=pyproject["project"]["maintainers"][0]["email"],
        maintainer=pyproject["project"]["maintainers"][0]["name"],
        name=pyproject["project"]["name"],
        package_data={
            "sktime": [
                "*.csv",
                "*.csv.gz",
                "*.arff",
                "*.arff.gz",
                "*.txt",
                "*.ts",
                "*.tsv",
            ]
        },
        packages=find_packages(
            where=".",
            exclude=["tests", "tests.*"],
        ),
        project_urls=pyproject["project"]["urls"],
        python_requires=pyproject["project"]["requires-python"],
        setup_requires=pyproject["build-system"]["requires"],
        url=pyproject["project"]["urls"]["repository"],
        version=pyproject["project"]["version"],
        zip_safe=False,
    )


if __name__ == "__main__":
    setup_package()
