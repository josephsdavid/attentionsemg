from scipy import signal
from sklearn.preprocessing import StandardScaler
import numpy as np
#import pywt
#from PyEMD import EMD

def _butter_highpass(cutoff, fs, order=3):
    # nyquist frequency!!
    nyq = .5*fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff=2, fs=200, order=3):
    b, a = _butter_highpass(cutoff=cutoff, fs=fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

## TODO: either allow for true scaling transformer to be trained
## and resused on validation/test data. Or make new function for this
## Fix addresses issue of array containing +/- values.
def scale(arr3d):
    for i in range(arr3d.shape[0]):
        arr3d[i,:,:] /= np.abs(arr3d[i,:,:]).max(axis=0)
    return arr3d
