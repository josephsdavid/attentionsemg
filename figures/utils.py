#%%
import sys,os

# General Stuff
import numpy as np
import pandas as pd
from scipy import stats as st



def mask_exercise(y, mask_id='all'):
    masks = {
        'all':  np.where(y==y),
        'abc':  np.where(y!=0),
        'a':    np.where((y > 0) & (y < 13)),
        'b':    np.where((y >= 13) & (y < 30)),
        'c':    np.where(y >= 30)
    }

    return masks[mask_id]