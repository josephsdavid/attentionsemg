import multiprocessing
import numpy as np
import scipy.io

def load_mat(matPath, full_emg=False):

    n_channels = 16 if full_emg else 8
    res = scipy.io.loadmat(matPath)
    maxlen = res['emg'].shape[0]
    rep = res['rerepetition'][:maxlen].copy()
    emg = res['emg'][:maxlen,:n_channels].copy()
    lab = res['restimulus'][:maxlen].copy()
    subject = np.repeat(res['subject'], lab.shape[0])
    subject = subject.reshape(subject.shape[0],1)
    del res

    return emg, lab, rep, subject

def convert_mat_to_np(matPath, numPath, **kwargs):
    data = load_mat(matPath, kwargs)
    data = np.concatenate(data, axis=1)
    return np.save(numPath , data)
    
def _build_paths_by_sub(path, sub, prefix):
    subPaths = []
    for i in range(1,4):
        mPath = f"{path}/s{str(sub)}/S{str(sub)}_E{str(i)}_A1.mat"
        nPath = f"{path}/np/e{str(i)}_s{str(sub)}.npy" if prefix is None else f"{path}/np/{prefix}_e{str(i)}_s{str(sub)}.npy"
        subPaths.append((mPath,nPath))
    return subPaths


def build_nps(path, prefix=None, full_emg=False, **kwargs):
    res = [[convert_mat_to_np(matP, numP, full_emg=full_emg) for (matP,numP) in _build_paths_by_sub(path, i, prefix)] for i in range(1,11)]
    
    return res