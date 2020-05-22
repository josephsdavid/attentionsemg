import joblib
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
    LSTM,
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
    "dense": [500, 500, 2000],
    "drop": [0.36, 0.36, 0.36],
}

result = joblib.load("results.dmp")
print(result)




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


def rnn_model(n_time, n_class, n_features, dense=[50, 50, 50], drop=[0.2, 0.2, 0.2]):
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


model, cosine = build(rnn_model)
model.fit(
        train,
        epochs=55,
        validation_data=validation,
        callbacks=[
            ModelCheckpoint(
                f"h5/rnn_no_ln.h5",
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
result["rnn_no_ln"] = {}
result["rnn_no_ln"]["validation"] = model.evaluate(validation)
result["rnn_no_ln"]["test"] = model.evaluate(test)

print(result)
joblib.dump(result, "results.dmp")

