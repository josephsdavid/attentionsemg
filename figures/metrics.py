#%%
import sys,os

# General Stuff
import numpy as np
import pandas as pd
from scipy import stats as st

## sk-learn
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, 
average_precision_score, accuracy_score,
roc_auc_score, classification_report, matthews_corrcoef, 
precision_recall_fscore_support, precision_score, 
recall_score, f1_score)

#%%
def _class_line(yt,yp):
    line_rep = pd.DataFrame(classification_report(t, p, output_dict=True, digits=2)).T
    line_rep[['recall','f1-score','precision']]= np.round(line_rep[['recall','f1-score','precision']],3)
    line_rep[['support']] = line_rep[['support']].astype(np.int)
    return line_rep
def _class_rep(yys,**kwargs):
    _rep = [_class_line(t,p) for t,p in yys]
    # _rep[:][['recall','f1-score','precision']]= np.round(_rep[:][['recall','f1-score','precision']],3)
    # _rep[:][['support']] = _rep[:][['support']].astype(np.int)
    return _rep
#%%
def underperformers(truth, pred, n_std = 0):
    cm = confusion_matrix(truth, pred, normalize='true')
    vals = cm.diagonal()
    vals = vals[~np.isnan(vals)]
    uniq = np.unique(truth)
    c_dict = dict(zip(vals, uniq))
    return {v: k for k, v in c_dict.items() if k <= vals.mean() - n_std* vals.std()}
def outperformers(truth, pred, n_std = 0):
    cm = confusion_matrix(truth, pred, normalize='true')
    vals = cm.diagonal()
    vals = vals[~np.isnan(vals)]
    uniq = np.unique(truth)
    c_dict = dict(zip(vals, uniq))
    return {v: k for k, v in c_dict.items() if k >= vals.mean() + n_std* vals.std()}


def gen_class_dist(y):
    classDist = freqs = np.array(np.unique(y, return_counts=True)).T
    options, freqs = classDist[:,0],classDist[:,1].astype('float')/y.shape[0]
    return options, freqs

def gen_baselines(options, y, freqs):
    y_randomW = np.random.choice(options,y.shape, replace=False ,p=freqs)
    y_random = np.random.choice(options,y.shape)
    y_zeros = np.zeros_like(y)
    y_ones = np.ones_like(y)
    baselines = {
        'Weighted Random': y_randomW,
        'Unweighted Random': y_random,
        'All Zeros': y_zeros,
        'All Ones': y_ones
    }
    return baselines

def _metric_col(yys, m_fn):
    # print('STARTING COL')
    # yp = np.reshape(yp, (1,np.shape(yp)[0])) if np.ndim(yp)<2 else yp
    m_fn, m_par = m_fn
    results = [m_fn(t,p,**m_par) for t,p in yys]
    [_mean, _ci, _std, _min_max] = [np.mean(results),
                                    st.t.interval(0.95,
                                                  len(results)-1,
                                                  loc=np.mean(results),
                                                  scale=st.sem(results)
                                                  ),
                                    np.std(results),
                                    (np.min(results), np.max(results))]
    return [_mean, _ci, _std, _min_max]
    
def _metric_line(yys, m_fns, m_pars = None):
    print('STARTING LINE')
    return [[v for v in _metric_col(yys, m_fn)] for m_fn in m_fns]

def build_metrics(y_sets={}, metrics =[
        ('Acc', (accuracy_score, {})),
        ('Balanced Acc', (balanced_accuracy_score, {})),
        ('MCC', (matthews_corrcoef, {})),
        ('Precision', (precision_score, {'average':'weighted'})),
        ('Recall', (recall_score,  {'average':'weighted'})),
        ('f1-Score', (f1_score,  {'average':'weighted'}))
    ], 
    return_df=False):
    print('STARTING Build')
    col, m_fns = zip(*metrics)
    lines = [[v for v in _metric_line(y,m_fns)] for y in list(y_sets.values())]
    metrics= [[[np.round(v, decimals=4) for v in c] for c in l] for l in lines]
    errors = [[[np.round(cc[0]-cc[1][0], decimals=4),np.round(cc[1][1]-cc[0], decimals=4)] for ii,cc in enumerate(l)] for l in metrics]
    print('ENDING Build')
    if return_df:
        return col, np.array(metrics), np.array(errors), pd.DataFrame(metrics,index=y_sets.keys() ,columns=col)
    else:
        return col, np.array(metrics), np.array(errors)

def 

#%%
default_metrics = [
        ('Acc', (accuracy_score, {})),
        ('Balanced Acc', (balanced_accuracy_score, {})),
        ('MCC', (matthews_corrcoef, {})),
        ('Precision', (precision_score, {'average':'weighted'})),
        ('Recall', (recall_score,  {'average':'weighted'})),
        ('f1-Score', (f1_score,  {'average':'weighted'}))
    ]

# %%
# 
path_pairs = [
    ('test_yy.npy','predictions_t.npy'),
    ('test_yy.npy','predictions_t_imu.npy'),
    ('val_yy.npy','predictions_v.npy'),
    ('val_yy.npy','predictions_v_imu.npy'),
    ]

data_titles = [
    'sEMG\n(test)', 
    'sEMG+IMU\n(test)', 
    'sEMG\n(validation)', 
    'sEMG+IMU\n(validation)'
    ]


#%%
# np.save('data/columns.npy',cols)
# np.save('data/keys.npy',np.array(list(ysets.keys())))
# np.save('data/metrics.npy',lines)
# np.save('data/errors.npy',errors)
# %%



#%%
def main():
    data_sets = map(lambda t: (np.load(f'figures/data/{t[0]}'),np.load(f'figures/data/{t[1]}')),path_pairs)
    ysets = dict(zip(data_titles,data_sets))
    return ysets
#%%

if __name__ == "__main__":
    ysets = main()

# %%
