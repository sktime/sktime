"""Time series forecasting datasets."""
from pathlib import Path
from typing import Optional, Union

from sktime.datasets._data_io import load_tsf_to_dataframe
from sktime.datasets.base._base import BaseDataset
from sktime.datasets.base._metadata import ExternalDatasetMetadata
from sktime.datasets.tsf_dataset_names import tsf_all

DEFAULT_PATH = Path.cwd().parent / "data"
CITATION = ""


class TSFDatasetLoader(BaseDataset):
    """Forecasting datasets from Monash Time Series Forecasting Archive."""

    def __init__(
        self,
        name: str,
        save_dir: Optional[str] = None,
        return_data_type: str = "default_tsf",
        replace_missing_vals: Union[str, float] = "NaN",
    ):
        metadata = ExternalDatasetMetadata(
            name=name,
            task_type="forecasting",
            url=[f"https://zenodo.org/record/{tsf_all[name]}/files"],
            download_file_format="zip",
            citation=CITATION,
        )
        if save_dir is None:
            save_dir = Path(DEFAULT_PATH, name)
        else:
            save_dir = Path(save_dir, name)
        super().__init__(metadata, save_dir, return_data_type)
        self._missing_val = replace_missing_vals

    def _load(self):
        """Load the dataset into memory."""
        file_path = Path(self.save_dir, f"{self._metadata.name}.tsf")
        y, self._info = load_tsf_to_dataframe(
            full_file_path_and_name=file_path,
            replace_missing_vals_with=self._missing_val,
            return_type=self._return_data_type,
        )
        return y

    @property
    def metadata(self):
        """Return the dataset metadata."""
        # TODO: comine with self._metadata class
        return self._info
