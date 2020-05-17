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
)
import tensorflow.keras.backend as K

from activations import Mish
from optimizers import Ranger
import losses as l
import callbacks as cb
from layers import Attention, LayerNormalization
from data import dataset
from generator import generator

data = dataset("data/ninaPro")

reps = np.unique(data.repetition)
val_reps = reps[3::2]
train_reps = reps[np.where(np.isin(reps, val_reps, invert=True))]
test_reps = val_reps[-1].copy()
val_reps = val_reps[:-1]

train = generator(data, list(train_reps))
validation = generator(data, list(val_reps), augment=False)
test = generator(data, [test_reps][0], augment=False, batch_size=1)

n_time = train[0][0].shape[1]
n_class = 53
n_features = train[0][0].shape[-1]
model_pars = {
    "n_time": n_time,
    "n_class": n_class,
    "n_features": n_features,
    "dense": [500, 500, 2000],
    "drop": [0.2, 0.2, 0.2],
}


def build(model_fn):
    cosine = cb.CosineAnnealingScheduler(
        T_max=50, eta_max=1e-3, eta_min=1e-5, verbose=1, epoch_start=5
    )
    loss = l.focal_loss(gamma=3.0, alpha=6.0)
    model = model_fn(**model_pars)
    model.compile(Ranger(learning_rate=1e-3), loss=loss, metrics=["accuracy"])
    print(model.summary())
    return model, cosine


def attention_simple(inputs, n_time):
    a = Permute((2, 1), name="temporalize")(inputs)
    a = Dense(n_time, activation="softmax", name="attention_probs")(a)
    a_probs = Permute((2, 1), name="attention_vec")(a)
    output_attention_mul = Multiply(name="focused_attention")([inputs, a_probs])
    output_flat = Lambda(lambda x: K.sum(x, axis=1), name="temporal_average")(
        output_attention_mul
    )
    return output_flat


def base_model(n_time, n_class, n_features, dense=[50, 50, 50], drop=[0.2, 0.2, 0.2]):
    inputs = Input((n_time, n_features))
    x = inputs
    x = Conv1D(filters=128, kernel_size=3, padding="same", activation=Mish())(x)
    x = LayerNormalization()(x)
    x = attention_simple(x, n_time)
    for d, dr in zip(dense, drop):
        x = Dropout(dr)(x, training=True)
        x = Dense(d, activation=Mish())(x)
        x = LayerNormalization()(x)
    outputs = Dense(n_class, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model


def dense_model(n_time, n_class, n_features, dense=[50, 50, 50], drop=[0.2, 0.2, 0.2]):
    inputs = Input((n_time, n_features))
    x = inputs
    x = Dense(128, activation=Mish())(x)
    x = LayerNormalization()(x)
    x = attention_simple(x, n_time)
    for d, dr in zip(dense, drop):
        x = Dropout(dr)(x, training=True)
        x = Dense(d, activation=Mish())(x)
        x = LayerNormalization()(x)
    outputs = Dense(n_class, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model


def raw_model(n_time, n_class, n_features, dense=[50, 50, 50], drop=[0.2, 0.2, 0.2]):
    inputs = Input((n_time, n_features))
    x = inputs
    x = attention_simple(x, n_time)
    for d, dr in zip(dense, drop):
        x = Dropout(dr)(x, training=True)
        x = Dense(d, activation=Mish())(x)
        x = LayerNormalization()(x)
    outputs = Dense(n_class, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model


"""
stage 1:
    baseline model
    replace conv with dense
    replace conv with nothing
"""

stage_1 = dict(zip(["baseline", "dense", "none"], [base_model, dense_model, raw_model]))
results = {}

for k in stage_1.keys():
    model, cosine = build(stage_1[k])
    model.fit(
        train,
        epochs=55,
        validation_data=validation,
        callbacks=[
            ModelCheckpoint(
                f"h5/{k}".h5,
                monitor="val_loss",
                keep_best_only=True,
                save_weights_only=True,
            ),
            cosine,
        ],
        use_multiprocessing=True,
        workers=8,
    )
    results[k] ={}
    results[k]['validation'] = model.evaluate(validation)
    results[k]['test'] = model.evaluate(test)

import joblib
joblib.dump(results, 'stage_1.dmp')
