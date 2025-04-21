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
y = df.loc[:,['GDP_new','BSI_new','GPI_new','SPI_new']]
x = df.loc[:,['IP_new','NDM_new','DM_new','OILP_new','CON_new','RT_new']]
 
y_train = y.iloc[0:430,:]
x_train = x.iloc[0:430,:]
model = sm.DynamicFactorMQ(endog=y_train, factors=4, factor_orders=1, idiosyncratic_ar1=True)
results = model.fit(maxiter=1000, disp=True)

# in-sample predictions
dfm_pred = results.predict(start=231,end=530)

# %%
