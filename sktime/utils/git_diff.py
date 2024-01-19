"""Git related utilities to identify changed modules."""

__author__ = ["fkiraly"]
__all__ = []

import importlib.util
import inspect
import subprocess


def get_module_from_class(cls):
    """Get full parent module string from class.

    Parameters
    ----------
    cls : class
        class to get module string from, e.g., NaiveForecaster

    Returns
    -------
    str : module string, e.g., sktime.forecasting.naive
    """
    module = inspect.getmodule(cls)
    return module.__name__ if module else None


def get_path_from_module(module_str):
    r"""Get local path string from module string.

    Parameters
    ----------
    module_str : str
        module string, e.g., sktime.forecasting.naive

    Returns
    -------
    str : local path string, e.g., sktime\forecasting\naive.py
    """
    try:
        module_spec = importlib.util.find_spec(module_str)
        if module_spec is None:
            raise ImportError(
                f"Error in get_path_from_module, module '{module_str}' not found."
            )
        return module_spec.origin
    except Exception as e:
        raise ImportError(f"Error finding module '{module_str}'") from e


def is_module_changed(module_str):
    """Check if a module has changed compared to the main branch.

    Parameters
    ----------
    module_str : str
        module string, e.g., sktime.forecasting.naive
    """
    module_file_path = get_path_from_module(module_str)
    cmd = f"git diff remotes/origin/main -- {module_file_path}"
    try:
        output = subprocess.check_output(cmd, shell=True, text=True)
        return bool(output)
    except subprocess.CalledProcessError:
        return True


def is_class_changed(cls):
    """Check if a class' parent module has changed compared to the main branch.

    Parameters
    ----------
    cls : class
        class to get module string from, e.g., NaiveForecaster

    Returns
    -------
    bool : True if changed, False otherwise
    """
    module_str = get_module_from_class(cls)
    return is_module_changed(module_str)


def get_changed_lines(file_path):
    """Get changed or added lines from a file.

    Parameters
    ----------
    file_path : str
        path to file to get changed lines from

    Returns
    -------
    list of str : changed lines
    """
    try:
        # Run 'git diff' command to get the changes in the specified file
        result = subprocess.run(
            ["git", "diff", file_path], capture_output=True, text=True, check=True
        )

        # Extract the changed or new lines and return as a list of strings
        diff_output = result.stdout
        changed_lines = [
            line.strip() for line in diff_output.split("\n") if line.startswith("+ ")
        ]
        # remove first character ('+') from each line
        changed_lines = [line[1:] for line in changed_lines]

        return changed_lines

    except subprocess.CalledProcessError as e:
        # Handle errors, if any
        raise e


def get_packages_with_changed_specs():
    """Get packages with changed or added specs.

    Returns
    -------
    list of str : packages with changed or added specs
    """
    from packaging.requirements import Requirement

    changed_lines = get_changed_lines("pyproject.toml")

    packages = []
    for line in changed_lines:
        if line.find("'") > line.find('"') and line.find('"') != -1:
            sep = '"'
        elif line.find("'") == -1:
            sep = '"'
        else:
            sep = "'"

        req = line.split(sep)[1]

        # deal with ; python_version >= "3.7" in requirements
        if ";" in req:
            req = req.split(";")[0]

        pkg = Requirement(req).name
        packages.append(pkg)

    return packages
