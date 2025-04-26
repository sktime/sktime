#%%
from sktime.utils._testing.series import _make_series
from sktime.forecasting.dynamic_factor import DynamicFactor
import pandas as pd
import numpy as np
import statsmodels.tsa.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sktime.forecasting.base import ForecastingHorizon
df = pd.read_csv('/home/haixi/Documents/projects/data/LCDMA_March_2025/balanced_can_md.csv')
df = (df.set_index(pd.DatetimeIndex(df['Date']))
       .loc[:,['GDP_new','BSI_new','GPI_new','SPI_new','IP_new','NDM_new','DM_new','OILP_new','CON_new','RT_new']]
     )
df.index = df.index.to_period('M')
y = df.loc[:,['GDP_new','BSI_new','GPI_new','SPI_new']]
x = df.loc[:,['IP_new','NDM_new','DM_new','OILP_new','CON_new','RT_new']]
forecaster = DynamicFactor()  
forecaster.fit(df)  
fh = ForecastingHorizon([1,2,3],is_relative=True)
y_pred = forecaster.predict(fh=fh)  

