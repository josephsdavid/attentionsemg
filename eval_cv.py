from typing import Iterable
from functools import partial
import os
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, balanced_accuracy_score, log_loss
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Conv1D,
    Flatten,
    Lambda,
    Permute,
    Multiply,
    LSTM
)
import tensorflow.keras.backend as K
import tensorflow as tf


from functools import partial
from activations import Mish
from optimizers import Ranger
import losses as l
import callbacks as cb
from layers import Attention, LayerNormalization
from data import dataset, ma_batch
from generator import generator

print("data time")
data = dataset("data/ninaPro")




def get_arrays(g: generator) -> Iterable[np.ndarray]:
	return np.moveaxis(ma_batch(g.X, g.ma_len), -1, 0), g.y


no_imu_pars = {
    "n_time": 38,
    "n_class": 53,
    "n_features": 16,
    "dense": [500, 500, 2000],
    "drop": [0.36, 0.36, 0.36],
}

imu_pars = {
    "n_time": 38,
    "n_class": 53,
    "n_features": 19,
    "dense": [500, 500, 2000],
    "drop": [0.36, 0.36, 0.36],
}


def build(model_fn, h5_file, pars):
    loss = l.focal_loss(gamma=3., alpha=6.)
    model = model_fn(**pars)
    model.compile(Ranger(learning_rate=1e-3), loss=loss, metrics=["accuracy"])
    model.load_weights(h5_file)
    return model


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

path_dict = {
        'sEMG':'h5/cv_error_bar/',
        'sEMG+IMU':'h5/cv_imu_error_bar/'
        }

imu_dict = {
        'sEMG':False,
        'sEMG+IMU':True
        }


preds = {
    'sEMG':[],
   # 'sEMG+IMU':[]
# uncomment this when that has run
    }
labels = {'sEMG':[],'sEMG+IMU':[]}

print("prinintg")
for p in preds.keys():
    path = path_dict[p]
    for f in os.listdir(path):
        rep = [int(f[0])]
        x, y = get_arrays(generator(
            data,
            rep,
            augment = False,
            imu=imu_dict[p],
            shuffle=False))
        model_pars =  no_imu_pars if not imu_dict[p] else imu_pars
        model = build(base_model, f"{path}{f}", model_pars)
        print('predicting!')
        preds[p].append(model.predict(x))
        labels[p].append(y)

 # dont forget the argmax!