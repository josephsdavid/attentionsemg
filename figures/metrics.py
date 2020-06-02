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



def metric_col(yys, m_fn):
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
    
def metric_line(yys, m_fns, m_pars = None):
    print('STARTING LINE')
    return [[v for v in metric_col(yys, m_fn)] for m_fn in m_fns]

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
    lines = [[v for v in metric_line(y,m_fns)] for y in list(y_sets.values())]
    metrics= [[[np.round(v, decimals=4) for v in c] for c in l] for l in lines]
    errors = [[[np.round(cc[0]-cc[1][0], decimals=4),np.round(cc[1][1]-cc[0], decimals=4)] for ii,cc in enumerate(l)] for l in metrics]
    print('ENDING Build')
    if return_df:
        return col, np.array(metrics), np.array(errors), pd.DataFrame(metrics,index=y_sets.keys() ,columns=col)
    else:
        return col, np.array(metrics), np.array(errors)

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


def build_bar_plot(bars,filePath=None,**kwargs):
    # x_pos = np.arange(lines.shape[0])
    _w = 0.3
    fig, ax = plt.subplots(figsize=(12,5.5))
    for bar in bars:
        ax.bar(bar[0], bar[1],width=_w, 
            yerr=bar[2],
            capsize = 3,
            align='center', alpha=0.5, label=bar[3])
    
    # ax.bar(x_pos+_w/2, width= _w ,lines, yerr=errors, align='center', alpha=0.5)
    ax.set_ylabel(kwargs['ylabel'])
    ax.set_ylim([0,1])
    ax.set_xticks(bars[0][0]+0.3/2)
    ax.set_xticklabels(kwargs['xlabel'],rotation=45, ha='center')
    ax.set_title(kwargs['title'])
    ax.yaxis.grid(True)
    ax.legend(loc="upper right")
    # Save the figure and show
    plt.tight_layout()
    if filePath:
        plt.savefig(filePath,dpi=500, size=(12, 5))
    
    plt.show()

#%%
def main():
    data_sets = map(lambda t: (np.load(f'figures/data/{t[0]}'),np.load(f'figures/data/{t[1]}')),path_pairs)
    ysets = dict(zip(data_titles,data_sets))
    return ysets
#%%

if __name__ == "__main__":
    

    #%%
    cols, lines, errors = build_metrics(ysets, default_metrics, return_df=False)
    plot_conf = {
        'title':'Model Accuracy\n(Simple vs Balanced)',
        'xlabel':data_titles,
        'ylabel': 'Accuracy'
    }

    bars = [
        [np.arange(lines[:,0,0].shape[0])-0.3/2,lines[:,0,0], errors[:,0].T,'Acc'],
        [np.arange(lines[:,1,0].shape[0])+0.3/2,lines[:,1,0], errors[:,1].T,'Bal. Acc']
        ]

    build_bar_plot(bars, filePath='plots/acc_vt.png', **plot_conf)

# %%
