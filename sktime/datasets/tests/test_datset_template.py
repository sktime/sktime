import pytest
from extension_templates._dataset_template import MyDataset

def test_my_dataset_template():
    """
    Minimal test for dataset template extension.
    Checks that the class can be instantiated and is of correct type.
    """
    ds = MyDataset()
    assert isinstance(ds, MyDataset)
