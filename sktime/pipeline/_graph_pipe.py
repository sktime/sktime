#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements forecaster for selecting among different model classes."""

import pandas as pd
from sklearn.base import clone

from sktime.base import BaseEstimator
from sktime.base._simple_base import SimpleBaseEstimator, SimpleBaseObject
from sktime.base._tests_and_checks import LengthChecker
from sktime.forecasting.base import BaseForecaster
from sktime.transformations.base import BaseTransformer

__author__ = ["miraep8"]
__all__ = ["GraphOutput", "GraphNode", "GraphPipeline"]


class GraphOutput:
    """Manage the output of GraphPipeline as it is generated."""

    def __init__(self):
        """Makes empty dict for GraphOutput."""
        self.output = {}

    def add_output(self, name: str, output: pd.DataFrame):
        """Add a new named entry to output.

        NOTE: May add some checking to makes sure we aren't overwriting.

        Parameters
        ----------
            - name (str) the name of the new output being added.
            - output (pd.DataFrame)
        """
        self.output[name] = output

    def _index_output(self, name, index):
        """Lookup the output associated with name, and index it.

        NOTE - once again - you see the assumption of a pd.DataFrame
        """
        return self.output[name].loc[:, index]

    def _join_outputs(self, outputs):
        """Join together all the outputs in outputs into 1 to return.

        For now assumes outputs are all pd.DataFrame objects, but can
        be extended to be more general if needed.
        """
        return pd.concat(outputs)

    def get_input(self, names, indices):
        """Generate new input from outputs given names and indices.

        Parameters
        ----------
            - names (list of str) the
            - indices (list of tuples or ints), default = None
        """
        if not names:  # if no parents are specified assumed to be an 'input node':
            names = ["raw_input"]

        outputs = [self._index_output(n, index) for n, index in zip(names, indices)]
        return self._join_outputs(outputs)


class GraphNode(SimpleBaseObject):
    """GraphStep represents the logic and identity of a single step in GraphPipeline."""

    parameters = {
        "estimator": BaseEstimator,
        "name": str,
        "tags": list,
        "input_edges": list,
        "input_indices": list,
    }

    input_checks = [
        LengthChecker(
            to_check=["input_edges", "input_indices"],
            ignore_none=True,
        )
    ]

    def __init__(
        self,
        estimator: BaseEstimator,
        name: str,
        tags: list = [],
        input_edges: list = [],
        input_indices: list = [],
    ):
        params = {
            "estimator": estimator,
            "name": name,
            "tags": tags,
            "input_edges": input_edges,
            "input_indices": input_indices,
        }
        self.run_input_checks(**params)
        self.set_param(**params)

    def get_estimator(self):
        """Return the estimator of this GraphNode"""
        return self.estimator


class GraphPipeline(SimpleBaseEstimator):
    """GraphPipeline facilitates the construction of estimator DAGs.

    This work builds upon work from Benedikt Heidrich (@benHeid)
    ** Happy to change and add attribution, just putting in the above as
    a place holder.

    Parameters
    ----------
    - steps : list of GraphNode objects, (optional), default = None,
        contains the GraphNode objects in the order they should be fit.

    Attributes
    ----------

    - steps : list,
    - assembled_steps : list,
    - outputs : GraphOutput,

    Examples
    --------
    """

    parameters = {
        "steps": list,
        "assembled_steps": list,
        "estimators": list,
        "outputs": GraphOutput,
    }

    @classmethod
    def assemble(cls, steps: list, outputs=None):
        estimators = [node.get_estimator() for node in steps]
        if not outputs:
            outputs = GraphOutput()
        est_dict = {}
        for est in estimators:
            est_dict[id(est)] = clone(est)
        assembled_steps = []
        for node in steps:
            new_node = GraphNode(
                est_dict[id(node.estimator)],
                node.name,
                node.tags,
                node.input_edges,
                node.input_indices,
            )
            assembled_steps.append(new_node)
        params = {
            "steps": steps,
            "assembled_steps": assembled_steps,
            "estimators": estimators,
            "outputs": outputs,
        }

    def __init__(
        self,
        steps: list = [],
    ):
        params = GraphPipeline.assemble(steps)
        self.set_params(**params)

    def add(
        self, name: str, estimator: BaseEstimator, input_edges, input_indices, tags
    ):
        """add a new node to the end of the GraphPipeline"""
        new_node = GraphNode(
            estimator,
            name,
            tags,
            input_edges,
            input_indices,
        )
        self.steps.append(new_node)
        params = GraphPipeline.assemble(self.steps)
        self.set_params(**params)

    def fit(self, y, X=None):
        """Actually fit the GraphPipeline:"""

        self.outputs["raw_input"] = (y, X)
        for node in self.assembled_steps:
            input_y, input_X = self.outputs.get_input(
                node.input_edges, node.input_indices
            )
            if isinstance(node.get_estimator, BaseTransformer):
                transform_args = {"X": input_X, "y": input_y}
                if "y" in node.tags:  # if y in in transform, switch order of inputs
                    transform_args = {"X": input_y, "y": input_X}
                if "inverse" in node.tags:
                    output = node.estimator.inverse_transform(**transform_args)
                else:
                    output = node.estimator.fit_transform(**transform_args)
                if "y" in node.tags:
                    input_y = output
                else:
                    input_X = output
            if isinstance(node.get_estimator, BaseForecaster):
                node.estimator.fit(y=input_y, X=input_X)
                input_y, input_X = node.estimator.y, node.estimator.X

            self.outputs[node.name] = (input_y, input_X)
