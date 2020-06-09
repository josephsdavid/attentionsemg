#%%
import sys,os

# General Stuff
import numpy as np
import pandas as pd

default_format = lambda v : f'{v[0]} [{" - ".join(map(str,v[1]))}]'

def table_data(metrics, format_fn=default_format):
    return [[format_fn(c) for c in line] for line in metrics]

def table_df(metrics, headers):
    return pd.DataFrame(table_data(metrics[0]), index=metrics[1], columns= headers)

# %%
