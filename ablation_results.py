from typing import Iterable
import os
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, balanced_accuracy_score, log_loss, jaccard_score, confusion_matrix
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


from activations import Mish
from optimizers import Ranger
import losses as l
import callbacks as cb
from layers import Attention, LayerNormalization
from data import dataset, ma_batch
from generator import generator

data = dataset("data/ninaPro")

reps = np.unique(data.repetition)
val_reps = reps[3::2]
train_reps = reps[np.where(np.isin(reps, val_reps, invert=True))]
test_reps = val_reps[-1].copy()
val_reps = val_reps[:-1]

train = generator(data, list(train_reps), augment = False, shuffle = False)
validation = generator(data, list(val_reps), augment=False, shuffle = False)
test = generator(data, [test_reps][0], augment=False, shuffle = False)


def get_arrays(g: generator) -> Iterable[np.ndarray]:
	return np.moveaxis(ma_batch(g.X, g.ma_len), -1, 0), g.y


# we will look at non imu data here, this takes a long time (ma_batch is not
# fast)
(train_x, train_y), (val_x, val_y), (test_x, test_y) = (get_arrays(g) for g in [train, validation, test])


h5_dir = "h5/"

# making the models!

model_pars = {
    "n_time": 38,
    "n_class": 53,
    "n_features": 16,
    "dense": [500, 500, 2000],
    "drop": [0.36, 0.36, 0.36],
}


def build(model_fn, h5_file):
    loss = l.focal_loss(gamma=3., alpha=6.)
    model = model_fn(**model_pars)
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


def dense_model(n_time, n_class, n_features, dense=[50, 50, 50], drop=[0.2, 0.2, 0.2]):
    inputs = Input((n_time, n_features))
    x = Dense(128, activation=Mish())(inputs)
    x = LayerNormalization()(x)
    x, a = attention_simple(x, n_time)
    for d, dr in zip(dense, drop):
        x = Dropout(dr)(x)
        x = Dense(d, activation=Mish())(x)
        x = LayerNormalization()(x)
    outputs = Dense(n_class, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model


def raw_model(n_time, n_class, n_features, dense=[50, 50, 50], drop=[0.2, 0.2, 0.2]):
    inputs = Input((n_time, n_features))
    x = inputs
    x, a = attention_simple(x, n_time)
    for d, dr in zip(dense, drop):
        x = Dropout(dr)(x)
        x = Dense(d, activation=Mish())(x)
        x = LayerNormalization()(x)
    outputs = Dense(n_class, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model

def raffel_model(n_time, n_class, n_features, dense=[50, 50, 50], drop=[0.2, 0.2, 0.2]):
    inputs = Input((n_time, n_features))
    x = inputs
    x = Conv1D(filters=128, kernel_size=3, padding="same", activation=Mish())(x)
    x = LayerNormalization()(x)
    x = Attention()(x)
    for d, dr in zip(dense, drop):
        x = Dropout(dr)(x)
        x = Dense(d, activation=Mish())(x)
        x = LayerNormalization()(x)
    outputs = Dense(n_class, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model


def sum_model(n_time, n_class, n_features, dense=[50, 50, 50], drop=[0.2, 0.2, 0.2]):
    inputs = Input((n_time, n_features))
    x = inputs
    x = Conv1D(filters=128, kernel_size=3, padding="same", activation=Mish())(x)
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.sum(x, axis=1), name='sum')(x)
    for d, dr in zip(dense, drop):
        x = Dropout(dr)(x)
        x = Dense(d, activation=Mish())(x)
        x = LayerNormalization()(x)
    outputs = Dense(n_class, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model


def no_class_model(n_time, n_class, n_features, dense=None, drop=None):
    inputs = Input((n_time, n_features))
    x = inputs
    x = Conv1D(filters=128, kernel_size=3, padding="same", activation=Mish())(x)
    x = LayerNormalization()(x)
    x, a = attention_simple(x, n_time)
    x = Dropout(0.36)(x)
    outputs = Dense(n_class, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model

def small_class_model(n_time, n_class, n_features, dense=None, drop=None):
    inputs = Input((n_time, n_features))
    x = inputs
    x = Conv1D(filters=128, kernel_size=3, padding="same", activation=Mish())(x)
    x = LayerNormalization()(x)
    x, a = attention_simple(x, n_time)
    x = Dropout(0.36)(x)
    x = Dense(500)(x)
    x = LayerNormalization()(x)
    outputs = Dense(n_class, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model

def no_layer_norm_model(n_time, n_class, n_features, dense=None, drop=None):
    inputs = Input((n_time, n_features))
    x = inputs
    x = Conv1D(filters=128, kernel_size=3, padding="same", activation=Mish())(x)
    x, a = attention_simple(x, n_time)
    for d, dr in zip(dense, drop):
        x = Dropout(dr)(x)
        x = Dense(d, activation=Mish())(x)
    outputs = Dense(n_class, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model


def relu_model(n_time, n_class, n_features, dense=None, drop=None):
    inputs = Input((n_time, n_features))
    x = inputs
    x = Conv1D(filters=128, kernel_size=3, padding="same", activation="relu")(x)
    x, a = attention_simple(x, n_time)
    for d, dr in zip(dense, drop):
        x = Dropout(dr)(x)
        x = Dense(d, activation="relu")(x)
        x = LayerNormalization()(x)
    outputs = Dense(n_class, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model


def rnn_model(n_time, n_class, n_features, dense=[50, 50, 50], drop=[0.2, 0.2, 0.2]):
    inputs = Input((n_time, n_features))
    x = inputs
    x = Conv1D(filters=128, kernel_size=3, padding="same", activation=Mish())(x)
    x = LayerNormalization()(x)
    x = LSTM(128, activation=Mish())(x)
    for d, dr in zip(dense, drop):
        x = Dropout(dr)(x)
        x = Dense(d, activation=Mish())(x)
        x = LayerNormalization()(x)
    outputs = Dense(n_class, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model

def rnn_no_ln(n_time, n_class, n_features, dense=[50, 50, 50], drop=[0.2, 0.2, 0.2]):
    inputs = Input((n_time, n_features))
    x = inputs
    x = Conv1D(filters=128, kernel_size=3, padding="same", activation=Mish())(x)
    x = LSTM(128, activation=Mish())(x)
    for d, dr in zip(dense, drop):
        x = Dropout(dr)(x)
        x = Dense(d, activation=Mish())(x)
    outputs = Dense(n_class, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model



model_file_list = [
    (base_model, 'h5/baseline.h5'),
    (dense_model, 'h5/dense.h5'),
    (raw_model, 'h5/none.h5'),
    (raffel_model, 'h5/raffel.h5'),
    (sum_model, 'h5/sum.h5'),
    (no_class_model, 'h5/no_class.h5'),
    (small_class_model, 'h5/small_class.h5'),
    (no_layer_norm_model, 'h5/no_norm.h5'),
    (relu_model, 'h5/relu.h5'),
    (rnn_model, 'h5/rnn.h5'),
    (rnn_no_ln, 'h5/rnn_no_ln.h5')
]


# utility for defining our dict
def get_name(obj):
    name =[x for x in globals() if globals()[x] is obj][0]
    return(name)

results = {}
for model_tuple in model_file_list:
    name = get_name(model_tuple[0])
    results[name] = {}
    model = build(*model_tuple)
    print(name)
    print(model.count_params())
    train_loss = model.evaluate(train_x, train_y)[0]
    test_loss = model.evaluate(test_x, test_y)[0]
    results[name]['test/train'] = train_loss/test_loss
    test_preds = model.predict(test_x)
    results[name]['accuarcy'] = accuracy_score(test_y.argmax(-1),test_preds.argmax(-1))
    results[name]['matthews_corrcoef'] = matthews_corrcoef(test_y.argmax(-1),test_preds.argmax(-1))
    results[name]['roc_auc_score'] = roc_auc_score(test_y,test_preds)
    results[name]['balacc'] = balanced_accuracy_score(test_y.argmax(-1),test_preds.argmax(-1))
    results[name]['logloss'] = log_loss(test_y, test_preds)
    results[name]['jaccard_weighted'] = jaccard_score(test_y.argmax(-1), test_preds.argmax(-1), average='weighted')
    results[name]['jaccard_multiclass'] = jaccard_score(test_y.argmax(-1), test_preds.argmax(-1), average=None)
    results[name]['conmat'] = confusion_matrix(test_y.argmax(-1), test_preds.argmax(-1), normalize='all')
    print(results[name])
import joblib; joblib.dump(results, 'ablation_results.dmp')
