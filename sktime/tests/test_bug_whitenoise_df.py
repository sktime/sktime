import pandas as pd
from sktime.transformations.series.augmenter import WhiteNoiseAugmenter

def test_whitenoiseaugmenter_dataframe_support():
    df = pd.DataFrame({"a": range(10), "b": range(10, 20)})
    wa = WhiteNoiseAugmenter()
    result = wa.fit_transform(df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape
