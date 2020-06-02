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

# Plot Stuff
import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
import seaborn as sns

#%%

def metric_col(yt, yp, m_fn):
    # print('STARTING COL')
    yp = np.reshape(yp, (1,np.shape(yp)[0])) if np.ndim(yp)<2 else yp
    m_fn, m_par = m_fn
    results = [m_fn(yt,p,**m_par) for p in yp]
    _mean, _ci, _std, _min_max = np.mean(results), 
                        st.t.interval(0.95, 
                            len(results)-1, 
                            loc=np.mean(results), 
                            scale=st.sem(results)
                            ),
                        np.std(results),
                        (np.min(results), np.max(results))
    return [_mean, _ci, _std, _min_maxS]
    
def metric_line(y_true, y_pred, m_fns, m_pars = None):
    print('STARTING LINE')
    y_true, y_pred = y_true, y_pred
    return [[v for v in metric_col(y_true, y_pred, m_fn)] for m_fn in m_fns]

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
    lines = [[v for v in metric_line(_yt,_yp,m_fns)] for (_yt,_yp) in list(y_sets.values())]
    metrics= [[[np.round(v, decimals=4) for v in c] for c in l] for l in lines]
    errors = [[[cc[0]-cc[1][0],cc[1][1]-cc[0]] for ii,cc in enumerate(l)] for l in metrics]
    print('ENDING Build')
    if return_df:
        return col, np.array(metrics), errors, pd.DataFrame(metrics,index=y_sets.keys() ,columns=col)
    else:
        return col, np.array(metrics), errors

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
    'sEMG (test)', 
    'sEMG+IMU (test)', 
    'sEMG (validation)', 
    'sEMG+IMU (validation)'
    ]

data_sets = map(lambda t: (np.load(f'data/{t[0]}'),np.load(f'data/{t[1]}')),path_pairs)

ysets = dict(zip(data_titles,data_sets))

#%%
cols, lines, errors = build_metrics(ysets, default_metrics, return_df=False)

#%%
# np.save('data/columns.npy',cols)
# np.save('data/keys.npy',np.array(list(ysets.keys())))
# np.save('data/metrics.npy',lines)
# np.save('data/errors.npy',errors)
# %%
plot_conf = {
    'title':'Gesture Classification',
    'xlabel':[list(ysets.keys())],
    'ylabel': 'Balanced Accuracy'
}

def build_bar_plot(lines,errors,filePath=None,**kwargs):
    x_pos = np.arange(lines.shape[0])

    fig, ax = plt.subplots()
    ax.bar(x_pos, lines, yerr=errors, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel(kwargs['ylabel'])
    ax.set_ylim([0,1])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(kwargs['xlabel']))
    ax.set_title(kwargs['title'])
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    if filePath:
        plt.savefig(filePath)
    
    plt.show()

#%%
build_bar_plot(lines[:,0,0],lines[:,0,2], filePath='balanced_bplot_w_err_v_and_t.png', **plot_conf)

# %%
