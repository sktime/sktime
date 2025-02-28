"""Interface for running a registry of models on a registry of validations."""

import functools
import logging
import os
import time
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


def run(
    model_registry,
    validation_registry,
    results_path: str = "./validation_results.csv",
    force_rerun: Optional[list[str]] = None,
    artefacts_store_dir: Optional[str] = None,
    run_params: Optional[dict] = None,
) -> pd.DataFrame:
    """Run a registry of models through a registry of validations.

    Parameters
    ----------
    model_registry: ModelRegistry
        contains the registry of models to be run through validations.
    validation_registry: ValidationRegistry
        contains the registry of validations to runeach model through.
    results_path: string, default = "./validation_results.csv"
        The file path to which the results will be written to, and results from prior
        runs will be read from.
    force_rerun: str or list of str, default="all"
        Argument to force models to rerun on validations. Model-validation
        combinations without results will always be ran, as well as models that
        are forced via
        this argument, which will overwrite previous results.
        - if `force_rerun` = None, don't force rerun any models (default)
        - if `force_rerun` = "all", force rerun all models
        - if `force_rerun` = list of string ids, force rerun only these specified models
    artefacts_store_dir: str, default=None
        A directory path or URI location to store extra output artefacts
        of the validations and models.
        If not None, then validations will be passed two kwargs;
        `validation_artefacts_dir` and `model_artefacts_dir`.
    run_params: dict, default=None
        A dictionary of optional run parameters.

    Returns
    -------
    pd.DataFrame: dataframe of validation results.
    """
    if run_params is None:
        run_params = {}

    try:
        results_df = pd.read_csv(results_path)
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=["validation_id", "model_id", "runtime_secs"])
        results_df["runtime_secs"] = results_df["runtime_secs"].astype(int)

    results_df = results_df.set_index(["validation_id", "model_id"], drop=False)
    results_list = []

    for validation_spec in validation_registry.all():
        if validation_spec.deprecated:
            logger.info(
                f"Skipping validation: {validation_spec.id} - as is deprecated."
            )
            continue
        for model_spec in model_registry.all():
            if model_spec.deprecated:
                logger.info(f"Skipping model: {model_spec.id} - as is deprecated.")
                continue

            if (
                not force_rerun == "all"
                and not (isinstance(force_rerun, list) and model_spec.id in force_rerun)
                and (validation_spec.id, model_spec.id) in results_df.index
            ):
                logger.info(
                    f"Skipping validation - model: "
                    f"{validation_spec.id} - {model_spec.id}"
                    ", as found prior result in results."
                )
                continue

            logger.info(
                f"Running validation - model: {validation_spec.id} - {model_spec.id}"
            )

            validation = validation_spec.make()
            validation = _form_validation_partial_with_store_dirs(
                validation,
                artefacts_store_dir,
                validation_spec,
                model_spec,
            )

            model = model_spec.make()
            results, elapsed_secs = _run_validation_model(validation, model, run_params)
            results = _add_meta_data_to_results(
                results, elapsed_secs, validation_spec, model_spec
            )
            results_list.append(results)

    additional_results_df = pd.DataFrame.from_records(results_list)
    results_df = pd.concat([results_df, additional_results_df], ignore_index=True)
    results_df = results_df.drop_duplicates(
        subset=["validation_id", "model_id"], keep="last"
    )
    results_df = results_df.sort_values(by=["validation_id", "model_id"])
    results_df = results_df.reset_index(drop=True)
    _write(
        results_df,
        results_path,
        to_front_cols=["validation_id", "model_id", "runtime_secs"],
    )
    return results_df


def _form_validation_partial_with_store_dirs(
    validation,
    artefacts_store_dir: Union[str, None],
    validation_spec,
    model_spec,
):
    """Form partial of validation with formed validation and model artefacts dirs.

    Also makes any needed dirs for the artefacts dirs.

    Parameters
    ----------
    validation: Validation
        The validation to form a partial of.
    artefacts_store_dir: str or None
        The directory to store artefacts in.
    validation_spec: ValidationSpec
        The validation spec to form the partial validation for.
    model_spec: ModelSpec
        The model spec to form the partial validation for.
    """
    if artefacts_store_dir is None:
        return validation
    validation_artef_dir = os.path.join(artefacts_store_dir, f"{validation_spec.id}/")
    model_artefacts_dir = os.path.join(validation_artef_dir, f"{model_spec.id}/")
    os.makedirs(model_artefacts_dir, exist_ok=True)
    validation = functools.partial(
        validation,
        validation_artefacts_dir=validation_artef_dir,
        model_artefacts_dir=model_artefacts_dir,
    )
    return validation


def _add_meta_data_to_results(
    results,
    elapsed_secs: float,
    validation_spec,
    model_spec,
):
    """Add meta data to results, raising if keys clash.

    Parameters
    ----------
    results: dict
        The results to add the meta data to.
    elapsed_secs: float
        The elapsed time in seconds.
    validation_spec: ValidationSpec
        The validation spec, id is added to the results.
    model_spec: ModelSpec
        The model spec, id is addes to the results.
    """
    results_meta_data = {
        "validation_id": validation_spec.id,
        "model_id": model_spec.id,
        "runtime_secs": elapsed_secs,
    }
    if bool(set(results) & set(results_meta_data)):
        raise ValueError(
            f"Validation:{validation_spec.id} on model:{model_spec.id} "
            f"returned results:{results} which contains a privileged key name."
        )
    return {**results, **results_meta_data}


def _run_validation_model(validation, model, run_params: dict):
    """Run given validation on given model, and store the results.

    Returns
    -------
    A tuple of (dict of results: Results type, elapsed time in seconds)
    """
    start_time = time.time()
    results = validation(model, **run_params)
    elapsed_secs = time.time() - start_time
    return results, elapsed_secs


def _write(df: pd.DataFrame, results_path: str, to_front_cols: list[str]):
    """Write the results to the results path."""
    df = df[to_front_cols + [col for col in df.columns if col not in to_front_cols]]
    df.to_csv(results_path, index=False)
