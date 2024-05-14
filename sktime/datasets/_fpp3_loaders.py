"""Utilities for loading datasets."""

__author__ = [
    "ericjb",
]

__all__ = [
    "load_fpp3",
]

import os
# import zipfile
# from urllib.error import HTTPError, URLError
# from warnings import warn

# import numpy as np
import pandas as pd

# from sktime.datasets._data_io import (
#     _download_and_extract,
#     _list_available_datasets,
#     _load_dataset,
#     _load_provided_dataset,
# )
# from sktime.datasets._readers_writers.tsf import load_tsf_to_dataframe
# from sktime.datasets.tsf_dataset_names import tsf_all, tsf_all_datasets
# from sktime.utils.validation._dependencies import _check_soft_dependencies

MODULE = os.path.dirname(__file__)

def load_fpp3(dataset):
    """Load a dataset from the fpp3 package.

    Parameters
    ----------
    dataset : str - The name of the dataset to load - also the key to a dictionary with the dataset parameters.
    
    Returns
    -------
    y : pd.DataFrame
    """
    # Define a dictionary with dataset parameters
    datasets = {
        ##############################################################################################################################
        ####  fpp3/fma datasets 
        ##############################################################################################################################
        "auto": { 'fn': 'auto.csv', 'a': 'fpp3', 'b': 'fma', 'freq': 'NA', 'name': 'Automobiles', 
               'dtype': {0: str, 1: str, 2: float, 3: float}, 'no_index': True
        },
        "bank": { 'fn': 'bank.csv', 'a': 'fpp3', 'b': 'fma', 'freq': 'NA', 'name': 'Bank', 
               'dtype': {0: float, 1: float, 2: float}, 'no_index': True
        },
        "cement": { 'fn': 'cement.csv', 'a': 'fpp3', 'b': 'fma', 'freq': 'NA', 'name': 'Cement', 
               'dtype': {0: float, 1: float, 2: float, 3: float}, 'no_index': True
        },
        "dexter": { 'fn': 'dexter.csv', 'a': 'fpp3', 'b': 'fma', 'freq': 'NA', 'name': 'Dexter', 
               'dtype': {0: float, 1: float}, 'no_index': True
        },
        "econsumption": { 'fn': 'econsumption.csv', 'a': 'fpp3', 'b': 'fma', 'freq': 'NA', 'name': 'econsumption', 
               'dtype': {0: float, 1: float}, 'no_index': True
        },
        "kkong": { 'fn': 'kkong.csv', 'a': 'fpp3', 'b': 'fma', 'freq': 'NA', 'name': 'kkong', 
               'dtype': {0: float, 1: float}, 'no_index': True
        },
        "mortal": { 'fn': 'mortal.csv', 'a': 'fpp3', 'b': 'fma', 'freq': 'NA', 'name': 'mortal', 
               'dtype': {0: float, 1: float}, 'no_index': True
        },
        "olympic": { 'fn': 'olympic.csv', 'ic': 0, 'a': 'fpp3', 'b': 'fma', 'freq': '4Y', 'name': 'Olympic', 
               'dtype': {0: int, 1: float}
        },
        "ozone": { 'fn': 'ozone.csv', 'a': 'fpp3', 'b': 'fma', 'freq': 'NA', 'name': 'ozone', 
               'dtype': {0: float, 1: float}, 'no_index': True
        },
        "pcv": { 'fn': 'pcv.csv', 'a': 'fpp3', 'b': 'fma', 'freq': 'NA', 'name': 'pcv', 
               'dtype': {0: float, 1: float}, 'no_index': True
        },
        "pulpprice": { 'fn': 'pulpprice.csv', 'a': 'fpp3', 'b': 'fma', 'freq': 'NA', 'name': 'Pulp Price', 
               'dtype': {0: float, 1: float}, 'no_index': True
        },
        "running": { 'fn': 'running.csv', 'a': 'fpp3', 'b': 'fma', 'freq': 'NA', 'name': 'running', 
               'dtype': {0: float, 1: float}, 'no_index': True
        },
        "texasgas": { 'fn': 'texasgas.csv', 'a': 'fpp3', 'b': 'fma', 'freq': 'NA', 'name': 'Texas Gas', 
               'dtype': {0: float, 1: float}, 'no_index': True
        },
        ##############################################################################################################################
        ####  fpp3/fpp3 datasets 
        ##############################################################################################################################
        "aus_accommodation": { 'fn': 'aus_accommodation.csv', 'ic': 0, 'a': 'fpp3', 'b': 'fpp3', 'freq': 'Q', 'name': 'Australian Accommodation', 
               'dtype': {0: str, 1: str, 2: float, 3: float, 4: float}, 'fmtfunc': lambda x: x.replace(' Q', '-Q') 
        },
        "aus_airpassengers": { 'fn': 'aus_airpassengers.csv', 'ic': 0, 'a': 'fpp3', 'b': 'fpp3', 'freq': 'Y', 'name': 'Australian Air Passengers', 
               'dtype': {0: int, 1: float}
        },
        "aus_arrivals": { 'fn': 'aus_arrivals.csv', 'ic': 0, 'a': 'fpp3', 'b': 'fpp3', 'freq': 'Q', 'name': 'Australian Arrivals', 
               'dtype': {0: str, 1: str, 2: float}, 'fmtfunc': lambda x: x.replace(' Q', '-Q') 
        },
        "bank_calls": { 'fn': 'bank_calls.csv', 'ic': 0, 'a': 'fpp3', 'b': 'fpp3', 'freq': 'NA', 'name': 'Bank Calls', 
               'dtype': {0: str, 1: int}, 'fmt': '%Y-%m-%dT%H:%M:%SZ'
        },
        "boston_marathon": { 'fn': 'boston_marathon.csv', 'ic': 1, 'a': 'fpp3', 'b': 'fpp3', 'freq': 'Y', 'name': 'Boston Marathon', 
               'dtype': {0: str, 1: int, 2: str, 3: str, 4: str}, 'tdcol': 'Time'
        },
        "canadian_gas": { 'fn': 'canadian_gas.csv', 'ic': 0, 'a': 'fpp3', 'b': 'fpp3', 'freq': 'M', 'name': 'Canadian Gas', 
               'dtype': {0: str, 1: float}, 'fmt': '%Y %b'
        },
        "guinea_rice": { 'fn': 'guinea_rice.csv', 'ic': 0, 'a': 'fpp3', 'b': 'fpp3', 'freq': 'Y', 'name': 'Guinea Rice', 
               'dtype': {0: int, 1: float}
        },
        "insurance": { 'fn': 'insurance.csv', 'ic': 0, 'a': 'fpp3', 'b': 'fpp3', 'freq': 'M', 'name': 'Insurance', 
                   'dtype': {0: str, 1: float, 2: float}, 'fmt': '%Y %b'
        },
        "prices": { 'fn': 'prices.csv', 'ic': 0, 'a': 'fpp3', 'b': 'fpp3', 'freq': 'Y', 'name': 'Prices', 
               'dtype': {0: int, **{i: float for i in range(1, 7)}}
        },
        "souvenirs": { 'fn': 'souvenirs.csv', 'ic': 0, 'a': 'fpp3', 'b': 'fpp3', 'freq': 'M', 'name': 'Souvenirs', 
                   'dtype': {0: str, 1: float}, 'fmt': '%Y %b'
        },
        "us_change": { 'fn': 'us_change.csv', 'ic': 0, 'a': 'fpp3', 'b': 'fpp3', 'freq': 'Q', 'name': 'U.S. Change', 
               'dtype': {0: str, **{i: float for i in range(1, 6)}}, 'fmtfunc': lambda x: x.replace(' Q', '-Q')
        },
        "us_employment": { 'fn': 'us_employment.csv', 'ic': 0, 'a': 'fpp3', 'b': 'fpp3', 'freq': 'M', 'name': 'U.S. Employment', 
                   'dtype': {0: str, 1: str, 2: str, 3: float}, 'fmt': '%Y %b'
        },
        "us_gasoline": { 'fn': 'us_gasoline.csv', 'ic': 0, 'a': 'fpp3', 'b': 'fpp3', 'freq': 'W', 'name': 'U.S. Gasoline', 
                   'dtype': {0: str, 1: float}, 'fmt': '%G-W%V-%u', 'fmtfunc': lambda x: x.replace(' ', '-') + '-7'
        },
        ##############################################################################################################################
        ####  fpp3/tsibble datasets 
        ##############################################################################################################################
        "pedestrian": { 'fn': 'pedestrian.csv', 'ic': 1, 'a': 'fpp3', 'b': 'tsibble', 'freq': 'NA', 'name': 'Pedestrian Traffic', 
               'dtype': {0: str, 1: str, 2: str, 3: int, 4: int}, 'fmt': '%Y-%m-%dT%H:%M:%SZ'
        },
        "tourism": { 'fn': 'tourism.csv', 'ic': 0, 'a': 'fpp3', 'b': 'tsibble', 'freq': 'Q', 'name': 'Australian Tourism', 
               'dtype': {0: str, 1: str, 2: str, 3: str, 4: float}, 'fmtfunc': lambda x: x.replace(' Q', '-Q') 
        },
        ##############################################################################################################################
        ####  fpp3/tsibbledata datasets 
        ##############################################################################################################################
        "ansett": { 'fn': 'ansett.csv', 'ic': 0, 'a': 'fpp3', 'b': 'tsibbledata', 'freq': 'W', 'name': 'Passengers', 
                   'dtype': {0: str, 1: str, 2: str, 3: int}, 'fmt': '%G-W%V-%u', 'fmtfunc': lambda x: x.replace(' ', '-') + '-7'
        },
        "aus_livestock": { 'fn': 'aus_livestock.csv', 'ic': 0, 'a': 'fpp3', 'b': 'tsibbledata', 'freq': 'M', 'name': 'Count', 
                   'dtype': {0: str, 1: str, 2: str, 3: float}, 'fmt': '%Y %b'
        },
        "aus_production": { 'fn': 'aus_production.csv', 'ic': 0, 'a': 'fpp3', 'b': 'tsibbledata', 'freq': 'Q', 'name': 'Australian Production', 
               'dtype': {0: str, 1: 'Int64', 2: 'Int64', 3: 'Int64', 4: 'Int64', 5: 'Int64'}, 'fmtfunc': lambda x: x.replace(' Q', '-Q') 
        },
        "aus_retail": { 'fn': 'aus_retail.csv', 'ic': 3, 'a': 'fpp3', 'b': 'tsibbledata', 'freq': 'M', 'name': 'Australian Retail', 
               'dtype': {0: str, 1: str, 2: str, 3: str, 4: float}, 'fmt': '%Y %b'
        },
        "gafa_stock": { 'fn': 'gafa_stock.csv', 'ic': 1, 'a': 'fpp3', 'b': 'tsibbledata', 'freq': 'D', 'name': 'Stock Prices', 
               'dtype': {0: str, 1: str, **{i: float for i in range(2, 8)}}, 'fmt': '%Y-%m-%d'
        },
        "global_economy": { 'fn': 'global_economy.csv', 'ic': 2, 'a': 'fpp3', 'b': 'tsibbledata', 'freq': 'Y', 'name': 'Global Economy', 
               'dtype': {0: str, 1: str, 2: int, **{i: float for i in range(3, 9)}}
        },
        "hh_budget": { 'fn': 'hh_budget.csv', 'ic': 1, 'a': 'fpp3', 'b': 'tsibbledata', 'freq': 'Y', 'name': 'HH Budget', 
               'dtype': {0: str, 1: int, **{i: float for i in range(2, 8)}}
        },
        "nyc_bikes": { 'fn': 'nyc_bikes.csv', 'ic': 1, 'a': 'fpp3', 'b': 'tsibbledata', 'freq': 'NA', 'name': 'NYC Bikes', 
               'dtype': {0: int, 1: str, 2: str, **{i: float for i in range(3, 9)}, 9: str, 10: int, 11: str}, 'fmt': '%Y-%m-%dT%H:%M:%SZ'
        },
        "olympic_running": { 'fn': 'olympic_running.csv', 'ic': 0, 'a': 'fpp3', 'b': 'tsibbledata', 'freq': '4Y', 'name': 'Olympic Running Race Times',
                            'dtype': {0: int, 1: int, 2: str, 3: float}, 
        },
        "PBS": { 'fn': 'PBS.csv', 'ic': 0, 'a': 'fpp3', 'b': 'tsibbledata', 'freq': 'M', 'name': 'PBS', 
               'dtype': {**{i: str for i in range(0, 7)}, 7: float, 8: float}, 'fmt': '%Y %b'
        },
        "pelt": { 'fn': 'pelt.csv', 'ic': 0, 'a': 'fpp3', 'b': 'tsibbledata', 'freq': 'Y', 'name': 'Pelt', 
               'dtype': {0: int, 1: float, 2: float}
        },
        "vic_elec": { 'fn': 'vic_elec.csv', 'ic': 0, 'a': 'fpp3', 'b': 'tsibbledata', 'freq': 'NA', 'name': 'Victoria Electric Demand', 
               'dtype': {0: str, 1: float, 2: float, 3: str, 4: bool}, 'fmt': '%Y-%m-%dT%H:%M:%SZ'
        },
    }
    
    p = datasets[dataset]

    path = os.path.join(MODULE, "data", p["a"], p["b"], p["fn"])
    if 'no_index' in p:
        y = pd.read_csv(path, dtype=p["dtype"], na_values='NA')
    else:
        y = pd.read_csv(path, index_col=p["ic"], dtype=p["dtype"], na_values='NA')
        if 'tdcol' in p:
            y.loc[:, p["tdcol"]] = pd.to_timedelta(y.loc[:, p["tdcol"]])
        if 'fmtfunc' in p:
            y.index = y.index.map(p['fmtfunc'])
        if 'fmt' in p:
            y.index = pd.to_datetime(y.index, format=p['fmt'])
            if (not dataset in {"aus_livestock", "us_employment", "aus_retail", "PBS"}):
                if p["freq"] == 'M':
                    y = y.asfreq('MS')
        else:
            y.index = pd.PeriodIndex(y.index, freq=p["freq"], name="Period")
    y.name = p["name"]
    
    if ( dataset in {"aus_airpassengers", "bank_calls", "canadian_gas", "guinea_rice", \
                     "souvenirs", "us_gasoline"} ):
        y = y.squeeze()
        
    if ( dataset in {"canadian_gas", "souvenirs", "insurance"} ):
       y.index = y.index.to_period('M')

    if ( dataset in {"us_gasoline"} ):
       y.index = y.index.to_period('W')
       
    if ( dataset in {"vic_elec"} ):
        y['Date'] = pd.to_datetime(y['Date'])

    if ( dataset in {"aus_arrivals"} ):
       y.reset_index(inplace=True)
       y.set_index(['Origin', 'Period'], inplace=True)
    
    if ( dataset in {"aus_livestock"} ):
       y.reset_index(inplace=True)
       y['Month'] = y['Month'].dt.to_period('M')
       y.set_index(['Animal', 'State', 'Month'], inplace=True)

    if ( dataset in {"olympic_running"} ):
       y.reset_index(inplace=True)
       y.columns = ['Year', 'Length', 'Sex', 'Time']
       y.set_index(['Length', 'Sex', 'Year'], inplace=True)

    if ( dataset in {"tourism"} ):
       y.reset_index(inplace=True)
       y.columns = ['Quarter', 'Region', 'State', 'Purpose', 'Trips']
       y.set_index(['Region', 'State', 'Purpose', 'Quarter'], inplace=True)

    if ( dataset in {"aus_accommodation"} ):
       y.reset_index(inplace=True)
       y.columns = ['Date', 'State', 'Takings', 'Occupancy', 'CPI']
       y.set_index(['State', 'Date'], inplace=True)

    if ( dataset in {"boston_marathon"} ):
       y.reset_index(inplace=True)
       y.columns = ['Year', 'Event', 'Champion', 'Country', 'Time']
       y['Time'] = y['Time'].apply(lambda x: x.total_seconds() / 60)
       y.set_index(['Event', 'Year'], inplace=True)
       
    if ( dataset in {"gafa_stock"} ):
       y.reset_index(inplace=True)
       y.set_index(['Symbol', 'Date'], inplace=True)

    if ( dataset in {"global_economy"} ):
       y.reset_index(inplace=True)
       y.columns = ['Year', 'Country', 'Code', 'GDP', 'Growth', 'CPI', 'Imports', 'Exports', 'Population']
       y.set_index(['Country', 'Year'], inplace=True)

    if ( dataset in {"hh_budget"} ):
       y.reset_index(inplace=True)
       y.columns = ['Year', 'Country', 'Debt', 'DI', 'Expenditure', 'Savings', 'Wealth', 'Unemployment']
       y.set_index(['Country', 'Year'], inplace=True)

    if ( dataset in {"nyc_bikes"} ):
       y.reset_index(inplace=True)
       y.set_index(['bike_id', 'start_time'], inplace=True)

    if ( dataset in {"pedestrian"} ):
       y.reset_index(inplace=True)
       y['Date'] = pd.to_datetime(y['Date'])
       y.set_index(['Sensor', 'Date_Time'], inplace=True)
       
    if ( dataset in {"us_employment"} ):
       y.reset_index(inplace=True)
       y['Month'] = y['Month'].dt.to_period('M')
       y.set_index(['Series_ID', 'Month'], inplace=True)

    if ( dataset in {"ansett"} ):
       y.reset_index(inplace=True)
       #print(type(y['Week'][0]))
       #y['Week'] = y['Week'].to_period('W')
       y.set_index(['Airports', 'Class', 'Week'], inplace=True)
       
    if ( dataset in {"aus_retail"} ):
       y.reset_index(inplace=True)
       y['Month'] = y['Month'].dt.to_period('M')
       y.set_index(['State', 'Industry', 'Month'], inplace=True)

    if ( dataset in {"PBS"} ):
       y.reset_index(inplace=True)
       y['Month'] = y['Month'].dt.to_period('M')
       y.set_index(['Concession', 'Type', 'ATC1', 'ATC2', 'Month'], inplace=True)

    return y
