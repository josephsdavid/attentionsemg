import numpy as np
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

import joblib

results = joblib.load("stage_2.dmp")
print(results)

strategy = tf.distribute.MirroredStrategy()

data = dataset("data/ninaPro")

reps = np.unique(data.repetition)
val_reps = reps[3::2]
train_reps = reps[np.where(np.isin(reps, val_reps, invert=True))]
test_reps = val_reps[-1].copy()
val_reps = val_reps[:-1]

train = generator(data, list(train_reps))
validation = generator(data, list(val_reps), augment=False)
test = generator(data, [test_reps][0], augment=False)

n_time = train[0][0].shape[1]
n_class = 53
n_features = train[0][0].shape[-1]
model_pars = {
    "n_time": n_time,
    "n_class": n_class,
    "n_features": n_features,
    "dense": [500,500,2000],
    "drop": [0.36 for _ in range(3)],
}

def build(model_fn):
    cosine = cb.CosineAnnealingScheduler(
        T_max=50, eta_max=1e-3, eta_min=1e-5, verbose=1, epoch_start=5
    )
    loss = l.focal_loss(gamma=3., alpha=6.)
    with strategy.scope():
        model = model_fn(**model_pars)
        model.compile(Ranger(learning_rate=1e-3), loss=loss, metrics=["accuracy"])
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



'''
stage 2: interest in feed forward area:
    directly to output layer
    small classifier
    no layer norm
    relu

'''

stage_3 = dict(zip(["no_class","small_class","no_norm","relu"], [no_class_model, small_class_model, no_layer_norm_model, relu_model]))

for k in stage_3.keys():
    model, cosine = build(stage_3[k])
    model.fit(
        train,
        epochs=55,
        validation_data=validation,
        callbacks=[
            ModelCheckpoint(
                f"h5/{k}.h5",
                monitor="val_loss",
                keep_best_only=True,
                save_weights_only=True,
            ),
            cosine,
        ],
        use_multiprocessing=True,
        workers=8,
        shuffle = False,
    )
    results[k] = {}
    results[k]["validation"] = model.evaluate(validation)
    results[k]["test"] = model.evaluate(test)

print("results")
print()
print(results)


joblib.dump(results, "stage_3.dmp")