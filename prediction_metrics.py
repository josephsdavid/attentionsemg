#%%
import sys,os

# General Stuff
import numpy as np
import pandas as pd
from scipy import stats

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Conv1D,
    Flatten,
    Lambda,
    Permute,
    Multiply,
)
import tensorflow.keras.backend as K
import tensorflow as tf


from activations import Mish
from optimizers import Ranger
import losses as l
import callbacks as cb
from layers import Attention, LayerNormalization
from data import dataset
from generator import generator

# Plot Stuff
import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
import seaborn as sns

## sk-learn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, average_precision_score, accuracy_score
from sklearn.metrics import roc_auc_score, classification_report, matthews_corrcoef, precision_recall_fscore_support

#%%
# strategy = tf.distribute.MirroredStrategy()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#%%
## BUILD non IMU DATA AND MODEL PARAMS

data = dataset("data/ninaPro")

reps = np.unique(data.repetition)
val_reps = reps[3::2]
train_reps = reps[np.where(np.isin(reps, val_reps, invert=True))]
test_reps = val_reps[-1].copy()
val_reps = val_reps[:-1]

train = generator(data, list(train_reps))
validation = generator(data, list(val_reps), augment=False)
test = generator(data, [test_reps][0], augment=False)

# n_time = train[0][0].shape[1]
# n_class = 53
# n_features = train[0][0].shape[-1]
# model_pars = {
#     "n_time": n_time,
#     "n_class": n_class,
#     "n_features": n_features,
#     "dense": [500, 500, 2000],
#     "drop": [0.36, 0.36, 0.36],
# }

loss = l.focal_loss(gamma=3., alpha=6.)

def build_model_pars(n_time, n_class, n_features):
    return {
        "n_time": n_time,
        "n_class": n_class,
        "n_features": n_features,
        "dense": [500, 500, 2000],
        "drop": [0.36, 0.36, 0.36],
    }


def build(model_fn, params):
    cosine = cb.CosineAnnealingScheduler(
        T_max=50, eta_max=1e-3, eta_min=1e-5, verbose=1, epoch_start=5
    )
    
    # with strategy.scope():
    model = model_fn(**params)
    
    print(model.summary())
    return model, cosine


def attention_simple(inputs, n_time):
    input_dim = int(inputs.shape[-1])
    a = Permute((2, 1), name='temporalize')(inputs)
    a = Dense(n_time, activation='softmax',  name='attention_probs')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply(name='focused_attention')([inputs, a_probs])
    output_flat = Lambda(lambda x: K.sum(x, axis=1), name='temporal_average')(output_attention_mul)
    return output_flat, a_probs


def base_model(n_time, n_class, n_features, dense=[50, 50, 50], drop=[0.2, 0.2, 0.2]):
    inputs = Input((n_time, n_features))
    x = inputs
    x = Conv1D(filters=128, kernel_size=3, padding="same", activation=Mish())(x)
    x = LayerNormalization()(x)
    x, a = attention_simple(x, n_time)
    for d, dr in zip(dense, drop):
        x = Dropout(dr)(x)
        x = Dense(d, activation=Mish())(x)
        x = LayerNormalization()(x)
    outputs = Dense(n_class, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model

def set_weights(model, h5_path):
    # model, cosine = build(base_model)
    model.load_weights(h5_path)
    model.compile(Ranger(learning_rate=1e-3), loss=loss, metrics=["accuracy"])
    return model

#%%

i = sys.argv[-1]
model, cosine = build(base_model)
