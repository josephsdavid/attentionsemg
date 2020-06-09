
#%%
import sys,os

# General Stuff
import numpy as np
import pandas as pd


# Plot Stuff
import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
import seaborn as sns

#%%
def plot_cm(cm,mask=None, filePath=None, title=None):
    # Generate a custom diverging colormap  
    
    
    cmap = sns.diverging_palette(10, 220, as_cmap=True)
    f, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(cm, 
        mask=mask,
        ax=ax,
        vmax=1, 
        vmin=0, 
        cmap="vlag",
        center=.2,
        square=True, 
        linewidths=.15, 
        cbar_kws={"shrink": .65})
    ax.set(title=title)
    if filePath:
        plt.savefig(filePath,dpi=500, size=(12, 12))
    return 

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
if __name__ == "__main__":
    
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

    plot_conf = {
        'title' : 'Model Accuracy\n(Simple vs Balanced)',
        'xlabel':  data_titles,
        'ylabel': 'Accuracy'
    }

    data_sets = map(lambda t: (np.load(f'data/{t[0]}'),np.load(f'data/{t[1]}')),path_pairs)

    ysets = dict(zip(data_titles,data_sets))

    bars = [
        [np.arange(lines[:,0,0].shape[0])-0.3/2,lines[:,0,0], errors[:,0].T,'Acc'],
        [np.arange(lines[:,1,0].shape[0])+0.3/2,lines[:,1,0], errors[:,1].T,'Bal. Acc']
        ]

    build_bar_plot(bars, filePath='plots/acc_vt.png', **plot_conf)

# %%
