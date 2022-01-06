# -*- coding: utf-8 -*-
"""Testing utility for easy generation of conditional fixtures in pytest_generate_tests.

Exports conditional_fixtures_and_names utility
"""

__author__ = ["fkiraly"]

__all__ = ["conditional_fixtures_and_names"]

import numpy as np


def conditional_fixtures_and_names(
    test_name, fixture_vars, generator_dict, fixture_sequence=None
):
    """Create conditional fixtures for pytest_generate_tests.

    Creates arguments for pytest.fixture.parameterize,
        using conditional fixture generation functions in generator_dict.

    Example: we want to loop over two fixture variables, "number" and "multiples"
        "number" are integers from 1 to 10,
        "multiples" are multiples of "number" up to "number"-squared
        we then write a generator_dict with two entries
        generator_dict["number"] is a function (test_name, **kwargs) -> list
            that returns [1, 2, ..., 10]
        generator_dict["multiples"] is a function (test_name, number, **kwargs) -> list
            that returns [number, 2* number, ..., number*number]

    This function automatically creates the inputs for pytest.mark.parameterize
        fixture_param_str = "number,multiples"
        fixture_prod = [(1, 1), (2, 2), (2, 4), (3, 3), (3, 6), ...]
        fixture_names = ["1-1", "2-2", "2-4", "3-3", "3-6", ...]

    Parameters
    ----------
    test_name : str, name of the test, from pytest_generate_tests
    fixture_vars : list of str
        fixture variables to loop over
    generator_dict : dict of generator functions
        keys are possible str in fixture_vars, expected signature is
            (test_name: str, **kwargs) -> fixtures: Listof[object], or
                (returning only fixtures)
            (test_name: str, **kwargs) -> fixtures, fixture_names: Listof[object]
                (returning fixture names as well as fixtures)
        generator_dict[my_variable] can take arguments with names
            in fixture_sequence to the left of my_variable
            it should return a list of fixtures for my_variable
            under the assumption that arguments have given values
    fixture_sequence : list of str, optional, default = None
        used in prioritizing conditional generators, sequentially (see above)

    Returns
    -------
    fixture_param_str : str, string to use in pytest.fixture.parameterize
        this is strings in "fixture_vars" concatenated, separated by ","
    fixture_prod : list of tuples, fixtures to use in pytest.fixture.parameterize
        fixture tuples, generated according to the following conditional rule:
            let fixture_vars = [fixture_var1, fixture_var2, ..., fixture_varN]
            all fixtures are obtained as following:
                for i in 1 to N
                    pick fixture[i] any element of generator_dict(
                        test_name,
                        fixture_var1 = fixture[1], ...,
                        fixture_var(i-1) = fixture[i-1],
                    )
    fixture_names : list of str, fixture ids to use in pytest.fixture.parameterize
        fixture names, generated according to the following conditional rule:
            let fixture_vars = [fixture_var1, fixture_var2, ..., fixture_varN]
            all fixtures names are obtained as following:
                for i in 1 to N
                    pick fixture_str_part[i] any element of generator_dict(
                        test_name,
                        fixture_var1 = fixture[1], ...,
                        fixture_var(i-1) = fixture[i-1],
                    ), second return is exists; otherwise str(first return)
            return "fixture_str_part[1]-fixture_str_part[2]-...-fixture_str_part[N]"
        fixture names correspond to fixtures with the same indices at picks (from lists)
    """
    fixture_vars = _check_list_of_str(fixture_vars, name="fixture_vars")

    # order fixture_vars according to fixture_sequence if provided
    if fixture_sequence is not None:
        fixture_sequence = _check_list_of_str(fixture_sequence, name="fixture_sequence")
        ordered_fixture_vars = []
        for fixture_var_name in fixture_sequence:
            if fixture_var_name in fixture_vars:
                ordered_fixture_vars += [fixture_var_name]
        fixture_vars = ordered_fixture_vars

    def get_fixtures(fixture_var, **kwargs):
        res = generator_dict[fixture_var](test_name, **kwargs)
        if len(res) == 2:
            fixture_prod = res[0]
            fixture_names = res[1]
        else:
            fixture_prod = res
            fixture_names = [str(x) for x in res]
        return fixture_prod, fixture_names

    for i, fixture_var in enumerate(fixture_vars):
        old_fixture_vars = fixture_vars[0:i]

        fixture_prod = [()]
        fixture_names = [""]

        # then take successive left products
        new_fixture_prod = []
        new_fixture_names = []

        for i, fixture in enumerate(fixture_prod):
            fixture_name = fixture_names[i]
            if i == 0:
                kwargs = dict()
            else:
                kwargs = zip(old_fixture_vars, fixture)
            new_fixtures, new_fixture_names_r = get_fixtures(fixture_var, **kwargs)
            new_fixture_prod += [
                fixture + (new_fixture,) for new_fixture in new_fixtures
            ]
            new_fixture_names += [
                f"{fixture_name}-{x}" for x in new_fixture_names_r
            ]

        fixture_prod = new_fixture_prod
        fixture_names = new_fixture_names
        fixture_names = [x[1:] for x in fixture_names]

    fixture_param_str = ",".join(fixture_vars)

    return fixture_param_str, fixture_prod, fixture_names


def _check_list_of_str(obj, name="obj"):
    """Check whether obj is a list of str.

    Parameters
    ----------
    obj : any object, check whether is list of str
    name : str, default="obj", name of obj to display in error message

    Returns
    -------
    obj, unaltered

    Raises
    ------
    TypeError if obj is not list of str
    """
    if not isinstance(obj, list) or not np.all(isinstance(x, str) for x in obj):
        raise TypeError(f"{obj} must be a list of str")
    return obj
