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



## sk-learn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, average_precision_score, accuracy_score
from sklearn.metrics import roc_auc_score, classification_report, matthews_corrcoef, precision_recall_fscore_support

#%%
# strategy = tf.distribute.MirroredStrategy()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#%%
def gen_to_nmpy(gen, file_name=None, y_raw=False):
    xx_imu = np.vstack([x for x,y in gen])
    xx = xx_imu[:,:,:-3]
    yy = np.vstack([y for x,y in gen])
    if file_name:
        np.save(f'figures/data/{file_name}_xx_imu',xx_imu)
        np.save(f'figures/data/{file_name}_xx',xx)
        np.save(f'figures/data/{file_name}_yy_raw',yy)
        np.save(f'figures/data/{file_name}_yy',np.argmax(yy, axis=1))
    if not y_raw:
        yy = np.argmax(yy, axis=1)
    return xx_imu, xx, yy

def build_model_pars(n_time, n_class, n_features):
    return {
        "n_time": n_time,
        "n_class": n_class,
        "n_features": n_features,
        "dense": [500, 500, 2000],
        "drop": [0.36, 0.36, 0.36],
    }

#%%
## BUILD IMU DATA AND MODEL PARAMS
data = dataset("data/ninaPro")

reps = np.unique(data.repetition)
val_reps = reps[3::2]
train_reps = reps[np.where(np.isin(reps, val_reps, invert=True))]
test_reps = val_reps[-1].copy()
val_reps = val_reps[:-1]

train = generator(data, list(train_reps), imu=True)
validation = generator(data, list(val_reps), augment=False, imu=True)
test = generator(data, [test_reps][0], augment=False, imu=True)

#%%
# UNCOMMENT FOR NPY
# val_x_imu, val_x, val_y = gen_to_nmpy(validation, 'val')
# test_x_imu, test_x, test_y = gen_to_nmpy(test, 'test')
# del train
val_x_imu, val_x, val_y = np.load('figures/data/val_xx_imu.npy'), np.load('figures/data/val_xx.npy'), np.load('figures/data/val_yy.npy')
test_x_imu, test_x, test_y = np.load('figures/data/test_xx_imu.npy'), np.load('figures/data/test_xx.npy'), np.load('figures/data/test_yy.npy')
#%%
loss = l.focal_loss(gamma=3., alpha=6.)
model_pars = build_model_pars(val_x.shape[1], 53, val_x.shape[-1])
model_pars_imu = build_model_pars(val_x_imu.shape[1], 53, val_x_imu.shape[-1])
#%%
## Model Builders

def build(model_fn, pars):
    cosine = cb.CosineAnnealingScheduler(
        T_max=50, eta_max=1e-3, eta_min=1e-5, verbose=1, epoch_start=5
    )
    model = model_fn(**pars)
    
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

# i = sys.argv[-1]
model, cosine = build(base_model, model_pars)
model_imu, cosine_imu = build(base_model, model_pars_imu)

#%%
#######
# RUN THIS CHUNK TO CREATE PREDICTIONS PER sEMG ONLY MODEL
predictions_v_raw, predictions_t_raw = [],[]

for i in range(1,31):
    print(50*'#'+f'\nPrediction set {i} in progress...')
    _model = set_weights(model, f'h5/error_bar/{i}.h5')
    _pv, _pt = _model.predict(val_x), _model.predict(test_x)
    predictions_v_raw.append(_pv)
    predictions_t_raw.append(_pt)
    print(f'Prediction set {i} complete!')
predictions_v = [np.argmax(p, axis=1) for p in predictions_v_raw]
predictions_t = [np.argmax(p, axis=1) for p in predictions_t_raw]
#%%
# ** UNCOMMENT HERE TO BUILD npy FILES **
# Need to make this whole generation process into fuctions...maybe classes
# np.save('figures/data/predictions_v_raw',predictions_v_raw)
# predictions_v_raw = None
# np.save('figures/data/predictions_v',predictions_v)
# predictions_v = None
# np.save('figures/data/predictions_t_raw',predictions_t_raw)
# predictions_t_raw = None
# np.save('figures/data/predictions_t',predictions_t)
# predictions_t = None
model =None
#%%
#######
# RUN THIS CHUNK TO CREATE PREDICTIONS PER IMU MODEL
predictions_v_imu_raw, predictions_t_imu_raw = [],[]

for i in range(1,31):
    print(50*'#'+f'\nPrediction set {i} in progress...')
    _model = set_weights(model_imu, f'h5/imu_error_bar/{i}.h5')
    _pv, _pt = _model.predict(val_x_imu), _model.predict(test_x_imu)
    predictions_v_imu_raw.append(_pv)
    predictions_t_imu_raw.append(_pt)
    print(f'Prediction set {i} complete!')
predictions_v_imu = [np.argmax(p, axis=1) for p in predictions_v_imu_raw]
predictions_t_imu = [np.argmax(p, axis=1) for p in predictions_t_imu_raw]

#%%
# ** UNCOMMENT HERE TO BUILD npy FILES **
# np.save('figures/data/predictions_v_imu_raw',predictions_v_imu_raw)
# predictions_v_imu_raw = None
# np.save('figures/data/predictions_v_imu',predictions_v_imu)
# predictions_v_imu = None
# np.save('figures/data/predictions_t_imu_raw',predictions_t_imu_raw)
# predictions_t_imu_raw = None
# np.save('figures/data/predictions_t_imu',predictions_t_imu)
# predictions_t_imu = None
model_imu = None

# %%
