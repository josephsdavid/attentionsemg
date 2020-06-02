import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Conv1D,
    Lambda,
    Permute,
    Multiply,
    Flatten,
)
import tensorflow.keras.backend as K
import tensorflow as tf


from activations import Mish
from optimizers import Ranger
import losses as l
import callbacks as cb
from layers import Attention, LayerNormalization
from data import nina4_dataset
from generator import generator

# I will leave all the flags to default, so you can fiddle
data = nina4_dataset(
    "nina4", # path to dataset
    butter = True, # butter high pass filter (performed after rectification bc that changes the power spectrum)
    rectify = True, # rectifies the data, proven to expose more information on the firing rate, see https://pubmed.ncbi.nlm.nih.gov/12706845/
    ma = 150, # moving average with a window size of 15, has not been tuned, performed lazily
    step = 50, # 5 observation step between windows
    window = 520, # 52 time steps per window at 200hx = 260 ms, chosen off precedent set in other papers, but not tuned. shorter is better
    exercises = ["a","b","c"], # which exercise set to use. Dont change for now
    features = None # if we want to engineer some classical features we can, never used
)

# splitting by repetition
reps = np.unique(data.repetition)
val_reps = reps[3::2]
train_reps = reps[np.where(np.isin(reps, val_reps, invert=True))]
test_reps = val_reps[-1].copy()
val_reps = val_reps[:-1]

# generator class indexes the dataset and lazily applies augmentation and moving
# average. Indexes return tuples, len returns number of batches

train = generator(
    data, # the dataset to use
    list(train_reps), # the repetitions to use
    shuffle = True,
    batch_size = 128, # chosen arbitrarily
    imu = False, # if you set this to true, set it to true on other generators as well
    augment = True, # applies noise at a spectrum of SNR ratios to data. Does not augment if class is zero to combat imbalance
    ma = True # bool for moving average or not (applied in __getitem__)
)

print("bam!")
validation = generator(data, list(val_reps), augment=False, ma=True)
test = generator(data, [test_reps][0], augment=False, ma=True)

n_time = train[0][0].shape[1]
n_class = train[0][-1].shape[-1]
n_features = train[0][0].shape[-1]

model_pars = {
    "n_time": n_time,
    "n_class": n_class,
    "n_features": n_features,
    "dense": [500, 500, 2000], # arbitrarily chosen classifier network
    "drop": [0.36, 0.36, 0.36], # "tuned" dropout rate, not sure how good it is
}

# cosine annealing rate scheduler (pairs well with our optimization strategy)
# for five epochs, stick at a high learning rate, then cosine anneal. Has not
# really been tuned
cosine = cb.CosineAnnealingScheduler(
    T_max=50, eta_max=1e-3, eta_min=1e-5, verbose=1, epoch_start=5
)

# focal loss stolen from computer vision. Softmax but it weights easy examples
# (aka when the class is 0) less than difficult examples, using the class
# alignment of the logits (if grounud truth is [0, 1, 0] and we predict [0, 0.9, 0.1])
# that contributes to the loss less than [0, 0.6, 0.4]. Gamma has an effect on
# this weighting, alpha does something that affects steepness of loss fn i
# think, higher alpha
loss = l.focal_loss(gamma=3., alpha=6.)


def attention_simple(inputs, n_time):
    input_dim = int(inputs.shape[-1])
    a = Permute((2, 1), name='temporalize')(inputs)
    a = Dense(n_time, activation='softmax',  name='attention_probs')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply(name='focused_attention')([inputs, a_probs])
    output_flat = Lambda(lambda x: K.sum(x, axis=1), name='temporal_average')(output_attention_mul)
    return output_flat

def make_model(n_time, n_class, n_features, dense, drop):
    inputs = Input((n_time, n_features))
    x = inputs
    x = Conv1D(filters=128, kernel_size=3, padding="same", activation=Mish())(x)
    x = LayerNormalization()(x)
    x = attention_simple(x, n_time)
    for d, dr in zip(dense, drop):
        x = Dropout(dr)(x)
        x = Dense(d, activation=Mish())(x)
        x = LayerNormalization()(x)
    outputs = Dense(n_class, activation="softmax")(x)
    model = Model(inputs, outputs)
    print(model.summary())
    return model

model = make_model(**model_pars)
# fancy optimizer, combination of rectified adam and lookahead
model.compile(Ranger(learning_rate=1e-3), loss=loss, metrics=["accuracy"])

model.fit(
    train,
    epochs=55,
    validation_data=validation,
    callbacks=[
        cosine,
    ],
    shuffle = False, # shuffling is done by the generator, if you shuffle here it will be infinitely slower
)


x = model.evaluate(test)

print(f"test_acc: {x[-1]}")
