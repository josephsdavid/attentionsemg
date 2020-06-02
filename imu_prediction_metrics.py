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
        np.save(f'{file_name}_xx_imu',xx_imu)
        np.save(f'{file_name}_xx',xx)
        np.save(f'{file_name}_yy_raw',yy)
        np.save(f'{file_name}_yy',np.argmax(yy, axis=1))
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
# val_x_imu, val_x, val_y = gen_to_nmpy(validation, 'val')
# test_x_imu, test_x, test_y = gen_to_nmpy(test, 'test')
# del train
val_x_imu, val_x, val_y = np.load('val_xx_imu.npy'), np.load('val_xx.npy'), np.load('val_yy.npy')
test_x_imu, test_x, test_y = np.load('test_xx_imu.npy'), np.load('test_xx.npy'), np.load('test_yy.npy')
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
## Prediction Functions
# def build_pred(model, weight_fn, weightPath='h5', data, process_fn=None):
#     pred = []
def baselineMetrics(t,b):
    _acc = accuracy_score(t,b)
    _accBal = balanced_accuracy_score(t,b)
    _matt = matthews_corrcoef(t,b)
    _prfs = precision_recall_fscore_support(t,b,average='weighted')
    return {
        'Acc': np.round(_acc,4),
        'Balanced Acc': np.round(_accBal,4),
        'MCC': np.round(_matt,4),
        'Precision': np.round(_prfs[0],4),
        'Recall': np.round(_prfs[1],4),
        'f1-Score': np.round(_prfs[2],4)
    }

def get_pred_data(path_pairs=[]):
    pairs = []
    for y,p in path_pairs:
        t = np.load(y)
        m = [[v for v in baselineMetrics(t,_p).values()] for _p in np.load(p)]
        pairs.append(m)
    return np.array(pairs)


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
# Need to make this whole generation process into fuctions...maybe classes
# np.save('predictions_v_raw',predictions_v_raw)
# predictions_v_raw = None
# np.save('predictions_v',predictions_v)
# predictions_v = None
# np.save('predictions_t_raw',predictions_t_raw)
# predictions_t_raw = None
# np.save('predictions_t',predictions_t)
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
# np.save('predictions_v_imu_raw',predictions_v_imu_raw)
# predictions_v_imu_raw = None
# np.save('predictions_v_imu',predictions_v_imu)
# predictions_v_imu = None
# np.save('predictions_t_imu_raw',predictions_t_imu_raw)
# predictions_t_imu_raw = None
# np.save('predictions_t_imu',predictions_t_imu)
# predictions_t_imu = None
model_imu = None

# %%
