from tensorflow.keras.layers import Dense, Input, GRU, Add, Dropout
from tensorflow.keras.models import Model
from activations import Mish
from optimizers import Ranger
from layers import Attention, LayerNormalization

'''
recurrent stuff lives here
'''

def gru(inn, nodes=40, **kwargs):
    return GRU(nodes, activation=Mish(),  return_state=True, return_sequences=True)(inn, **kwargs)


def block(inn, nodes=40,**kwargs):
    val, state = gru(inn, nodes,**kwargs)
    val2 = Attention()(val)
    return val, state, val2


def build_att_gru(n_time, n_classes, nodes=40, blocks=3,
                  loss='categorical_crossentropy', optimizer=Ranger,model_id=None, **optim_args):
    '''
    build_att_gru
    --------------
    args:
        n_time
        n_out
        nodes=40
        blocks=3, represents number of att_gru blocks in the model
        loss='categorical_crossentropy'
        optimizer=Ranger (no parentheses)
        model_id = model id
        args for optimizer
    Notes:
        returns compiled model.
        requires by default one hot encoded Y data
    '''
    inputs = Input((n_time, 16))
    x = Dense(128)(inputs)
    x, h, a = block(x, nodes)
    attention=[a]
    for _ in range(blocks-1):
        x, h, a = block(x, nodes, initial_state=h)
        attention.append(a)
    out = Add()(attention)
    outputs = Dense(n_classes, activation="softmax")(out)
    model = Model(inputs, outputs)
    model.compile(optimizer(**optim_args), loss=loss,  metrics=['accuracy'])
    if model_id is not None:
        model.load_weights(f"{model_id}.h5")
    return model



def _dblock(inn, nodes=40,**kwargs):
    '''block with dropout'''
    val, state = gru(inn, nodes,**kwargs)
    val2 = Dropout(0.5)(val)
    val2 = Attention()(val2)
    return val, state, val2
def build_att_gru_dropout(n_time, n_classes, nodes=40, blocks=3,
                  loss='categorical_crossentropy', optimizer=Ranger, model_id=None, **optim_args):
    '''
    build_att_gru
    --------------
    args:
        n_time
        n_out
        nodes=40
        blocks=3, represents number of att_gru blocks in the model,
        dropout=0.5
        loss='categorical_crossentropy'
        optimizer=Ranger (no parentheses)
        args for optimizer
        model_id: id of model for reload of weights
    Notes:
        returns compiled model.
        requires by default one hot encoded Y data
    '''
    inputs = Input((n_time, 16))
    x = Dense(128)(inputs)
    x, h, a = block(x, nodes)
    attention=[a]
    for _ in range(blocks-1):
        x, h, a = block(x, nodes, initial_state=h)
        attention.append(a)
    out = Add()(attention)
    out = Dropout(0.5)(out)
    outputs = Dense(n_classes, activation="softmax")(out)
    model = Model(inputs, outputs)
    model.compile(optimizer(**optim_args), loss=loss,  metrics=['accuracy'])
    if model_id is not None:
        model.load_weights(f"{model_id}.h5")
    return model



def normblock(inn, nodes=40,**kwargs):
    val, state = gru(inn, nodes,**kwargs)
    val = LayerNormalization()(val)
    val2 = Attention()(val)
    return val, state, val2



def build_att_gru_norm(n_time, n_classes, nodes=40, blocks=3,
                  loss='categorical_crossentropy', optimizer=Ranger, model_id=None, **optim_args):
    '''
    build_att_gru_norm
    --------------
    att_gru with layer norm on gru
    args:
        n_time
        n_out
        nodes=40
        blocks=3, represents number of att_gru blocks in the model,
        loss='categorical_crossentropy'
        optimizer=Ranger (no parentheses)
        args for optimizer
        model_id: id of model for reload of weights
    Notes:
        returns compiled model.
        requires by default one hot encoded Y data
    '''
    inputs = Input((n_time, 16))
    x = Dense(128)(inputs)
    x, h, a = normblock(x, nodes)
    attention=[a]
    for _ in range(blocks-1):
        x, h, a = normblock(x, nodes, initial_state=h)
        attention.append(a)
    out = Add()(attention)
    outputs = Dense(n_classes, activation="softmax")(out)
    model = Model(inputs, outputs)
    model.compile(optimizer(**optim_args), loss=loss,  metrics=['accuracy'])
    if model_id is not None:
        model.load_weights(f"{model_id}.h5")
    return model

