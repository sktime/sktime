import pandas as pd

from sktime.transformations.compose import ColumnEnsembleTransformer
from sktime.transformations.series.detrend import Detrender
from sktime.transformations.series.difference import Differencer


def test_column_ensemble_transformer_html_representation():
    """Test HTML representation of ColumnEnsembleTransformer includes column names."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    transformer = ColumnEnsembleTransformer(
        [("foo", Differencer(), "a"), ("bar", Detrender(), "b")]
    )
    transformer.fit(df)
    html_repr = transformer._repr_html_()

    assert "foo" in html_repr
    assert "bar" in html_repr
    assert "a" in html_repr
    assert "b" in html_repr
    assert "Differencer" in html_repr
    assert "Detrender" in html_repr
    assert "ColumnEnsembleTransformer" in html_repr
