from csv import DictReader
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow.keras as keras
import utils as u
import multiprocessing
import numpy as np
import callbacks as cb
import losses as l
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
    TensorBoard,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.layers import (
    Add,
    Input,
    Dense,
    GRU,
    PReLU,
    Dropout,
    TimeDistributed,
    Conv1D,
    Flatten,
    MaxPooling1D,
    LSTM,
    Lambda,
    Permute,
    Reshape,
    Multiply,
    RepeatVector,
)
import builders.recurrent as br
import builders.attentional as ba
import builders.conv as bc
from activations import Mish
from optimizers import Ranger
from layers import Attention, LayerNormalization
import tensorflow.keras.backend as K
import joblib


reader = DictReader(open("harold/harold.csv"))
data = {}
for row in reader:
    for k, v in row.items():
        data.setdefault(k, []).append(v)
data = {k: np.array(v).astype(np.float16) for k, v in data.items()}
data["exercise_id"] -= 1
data.pop("TimeStamp_s")
data.pop("exercise_amt")
# we dont really care about the session
data.pop("session_id")
y = data.pop("exercise_id")
subjects = data.pop("subject_id")
data = {
    k: v
    for k, v in data.items()
    if k
    in [
        "sID1_AccX_g",
        "sID1_AccY_g",
        "sID1_AccZ_g",
        "sID1_GyroX_deg/s",
        "sID1_GyroY_deg/s",
        "sID1_GyroZ_deg/s",
    ]
}
X = np.column_stack(list(data.values()))


class generator(keras.utils.Sequence):
    def __init__(self, x, y, test=False, batch_size=128):
        self.x = x
        self.y = y
        self.test = test
        self.shuffle = not self.test
        self.batch_size = batch_size
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(self.x.shape[0])
        if not self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(self.x.shape[0] / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        out = self.x[indexes, :, :].copy()
        if not self.test:
            for i in range(out.shape[0]):
                out[i, :, :] = out[i, :, :] + np.random.normal(
                    loc=0, scale=0.5, size=out[i, :, :].shape
                )
                scalingFactor = np.random.normal(
                    loc=1.0, scale=0.5, size=1
                )  # shape=(1,3)
                out[i, :, :] *= scalingFactor
        return out, self.y[indexes, :]


def attention_dumb(inputs, n_time):
    # inputs.shape = (batch_size, time_steps, features)
    # assume away batch size --> inputs.shape = (time_steps, features)
    # this is a transpose
    a = Permute((2, 1), name="temporalize")(inputs)
    # a.shape = (features, time_steps)
    # gets fed in feature by feature now
    # goes into softmax layer to calculate alpha_t (see attention equation), as
    # columns represent timesteps we have αt=exp(et)/∑Tk=1exp(ek)
    # walking through this equation, (from bahdanau et al attention paper),
    # we have (top half) = e^(result of network at timestep t)
    # (bottom half) = sum across time of the results of a feature
    a = Dense(n_time, activation="softmax", name="attention_probs")(a)
    a_probs = Permute((2, 1), name="attention_vec")(a)
    output_attention_mul = Multiply(name="focused_attention")([inputs, a_probs])
    output_flat = Lambda(lambda x: K.sum(x, axis=1), name="temporal_average")(
        output_attention_mul
    )
    return output_flat, a_probs


def build_conv_attention(
    n_time, n_class, dense=[50, 50, 50], drop=[0.1, 0.1, 0.1], model_id=None
):
    inputs = Input((n_time, 6))
    x = inputs
    # x=Dense(250, activation=Mish())(x)
    x = Conv1D(filters=128, kernel_size=3, padding="same", activation=Mish())(x)
    x = LayerNormalization()(x)
    x, a = attention_dumb(x, n_time)
    for d, dr in zip(dense, drop):
        x = Dropout(dr)(x, training=True)
        x = Dense(d, activation=Mish())(x)
        x = LayerNormalization()(x)
    outputs = Dense(n_class, activation="softmax")(x)
    model = Model(inputs, outputs)
    if model_id is not None:
        model.load_weights(f"{model_id}.h5")
    return model, Model(inputs, Dense(16)(a))


def split(X, y, subjects, id):
    train_ids = np.where(subjects != id)
    test_ids = np.where(subjects == id)
    return (X[train_ids], y[train_ids]), (X[test_ids], y[test_ids])


out = {}
for i in np.unique(subjects):
    train, test = split(X, y, subjects, i)

    x_train = np.moveaxis(u.window_roll(train[0], 50, 200), -1, 1)
    y_train = to_categorical(u.window_roll(train[-1], 50, 200)[0, :, 0])
    x_test = np.moveaxis(u.window_roll(test[0], 50, 200), -1, 1)
    y_test = to_categorical(u.window_roll(test[-1], 50, 200)[0, :, 0])

    n_time = x_train.shape[1]
    n_class = y_train.shape[-1]


    model, attn = build_conv_attention(
        n_time, n_class, [500, 100, 2000], drop=[0.1 for _ in range(3)]
    )
    cosine = cb.CosineAnnealingScheduler(
        T_max=50, eta_max=1e-3, eta_min=1e-5, verbose=1, epoch_start=5
    )
    loss = l.focal_loss(gamma=3.0, alpha=6.0)
    model.compile(
        Ranger(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, callbacks=[cosine], epochs=55, validation_data=(x_test, y_test), batch_size=1)
    out[i] = model.evaluate(x_test, y_test)
joblib.dump(out, "out.dmp")
