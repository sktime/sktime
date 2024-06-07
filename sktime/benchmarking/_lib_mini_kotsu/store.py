"""Functionality for storing validation results."""
from typing import List

import pandas as pd


def write(df: pd.DataFrame, results_path: str, to_front_cols: List[str]):
    """Write the results to the results path."""
    df = df[to_front_cols + [col for col in df.columns if col not in to_front_cols]]
    df.to_csv(results_path, index=False)
