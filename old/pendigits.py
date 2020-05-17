import utils as u
import multiprocessing
import numpy as np
import callbacks as cb
import losses as l
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.layers import Add, Input, Dense, GRU, PReLU, Dropout, TimeDistributed, Conv1D, Flatten, MaxPooling1D, LSTM, Lambda, Permute, Reshape, Multiply, RepeatVector
import builders.recurrent as br
import builders.attentional as ba
import builders.conv as bc
from activations import Mish
from optimizers import Ranger
from layers import Attention, LayerNormalization
import tensorflow.keras.backend as K
batch=128


def load_pendigits(data_path="toydata/pendigits"):
    if not os.path.exists(data_path + "/pendigits.tra"):
        os.makedirs(data_path, exist_ok=True)

        os.system(
            "wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra -P %s"
            % data_path
        )
        os.system(
            "wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes -P %s"
            % data_path
        )
        os.system(
            "wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.names -P %s"
            % data_path
        )

    # load training data
    with open(data_path + "/pendigits.tra") as file:
        data = file.readlines()
        data = [list(map(float, line.split(","))) for line in data]
        data = np.array(data).astype(np.float32)
        data_train, labels_train = data[:, :-1], data[:, -1]
        data_train /=100
        labels_train = labels_train.astype("int")

    # load testing data
    with open(data_path + "/pendigits.tes") as file:
        data = file.readlines()
        data = [list(map(float, line.split(","))) for line in data]
        data = np.array(data).astype(np.float32)
        data_test, labels_test = data[:, :-1], data[:, -1]
        data_test/=100
        labels_test = labels_test.astype("int")
    return (data_train, labels_train), (data_test, labels_test)

train, test = load_pendigits()

def roll(xy, wid, step):
    x, y = xy


x_train = np.moveaxis(u.window_roll(train[0], 1, 50), -1, 1)
y_train = to_categorical(u.window_roll(train[-1], 1, 50)[0,:,0])
x_test = np.moveaxis(u.window_roll(test[0], 1, 50), -1, 1)
y_test = to_categorical(u.window_roll(test[-1], 1, 50)[0,:,0])

print(train[-1].shape)





def attention_dumb(inputs, n_time):
    # inputs.shape = (batch_size, time_steps, features)
    # assume away batch size --> inputs.shape = (time_steps, features)
    # this is a transpose
    a = Permute((2, 1), name='temporalize')(inputs)
    # a.shape = (features, time_steps)
    # gets fed in feature by feature now
    # goes into softmax layer to calculate alpha_t (see attention equation), as
    # columns represent timesteps we have αt=exp(et)/∑Tk=1exp(ek)
    # walking through this equation, (from bahdanau et al attention paper),
    # we have (top half) = e^(result of network at timestep t)
    # (bottom half) = sum across time of the results of a feature
    a = Dense(n_time, activation='softmax',  name='attention_probs')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply(name='focused_attention')([inputs, a_probs])
    output_flat = Lambda(lambda x: K.sum(x, axis=1), name='temporal_average')(output_attention_mul)
    return output_flat, a_probs

def build_conv_attention(n_time, n_class, dense = [50,50,50], drop=[0.1, 0.1, 0.1], model_id=None):
    inputs = Input((n_time, 16))
    x = inputs
    #x=Dense(250, activation=Mish())(x)
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation=Mish())(x)
    x = LayerNormalization()(x)
    x, a = attention_dumb(x, n_time)
    for d, dr in zip(dense, drop):
        x = Dropout(dr)(x, training=True)
        x = Dense(d, activation=Mish())(x)
        x = LayerNormalization()(x)
    outputs = Dense(n_class, activation='softmax')(x)
    model = Model(inputs, outputs)
    if model_id is not None:
        model.load_weights(f"{model_id}.h5")
    return model, Model(inputs, Dense(16)(a))


n_time = x_train.shape[1]
n_class = y_train.shape[-1]
model, attn = build_conv_attention(n_time, n_class, [500,500,2000], drop = [0.5 for _ in range(3)])

cosine = cb.CosineAnnealingScheduler(T_max=50, eta_max=1e-3, eta_min=1e-5, verbose=1, epoch_start=5)
loss = l.focal_loss( gamma=3., alpha=6.)
model.compile(Ranger(learning_rate=1e-3), loss="categorical_crossentropy", metrics=['accuracy'])
import pdb; pdb.set_trace()  # XXX BREAKPOINT

model.fit(x_train, y_train, callbacks = [cosine], epochs = 55, batch_size = 32, validation_data = (x_test, y_test))


from sklearn.metrics import accuracy_score


ss = accuracy_score(y_test.argmax(-1), model.predict(x_test).argmax(-1))
print(ss)

import pdb; pdb.set_trace()  # XXX BREAKPOINT


