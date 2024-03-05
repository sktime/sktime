import numpy as np
import pandas as pd
from summarize import WindowSummarizer

df = pd.DataFrame(np.arange(10))
# print(df)
transformer = WindowSummarizer(lag_feature={"mean": [[-1, 1], [-1, 2]]}, n_jobs=1)
df_transformed = transformer.fit_transform(df)

print(df_transformed)