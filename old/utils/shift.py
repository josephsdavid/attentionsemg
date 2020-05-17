import numpy as np
import random

def shift_right(arr: np.ndarray):
    # 2d arr
    out = np.zeros_like(arr)
    for i in range(arr.shape[-1]):
        if i == arr.shape[-1]-1:
            out[:,i] = arr[:,[0,i]].mean(1)
        else:
            out[:,i] = arr[:,[i,i+1]].mean(1)
    return out


def shift_left(arr: np.ndarray):
    # 2d arr
    out = np.zeros_like(arr)
    for i in range(arr.shape[-1]):
        if i == arr.shape[-1]-1:
            out[:,i] = arr[:,[0,i]].mean(1)
        else:
            out[:,i] = arr[:,[i,i-1]].mean(1)
    return out

def shift(arr: np.ndarray):
    out = shift_left(arr) if random.random() > 0.5 else shift_right(arr)
    if random.random() <0.5:
        return out
    else:
        return arr

