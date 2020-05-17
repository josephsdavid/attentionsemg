from tensorflow.keras.layers import Dense, Input,  Add, Dropout
from tensorflow.keras.models import Model
from activations import Mish
from optimizers import Ranger
from layers import Attention, LayerNormalization

'''
two stage feed forward attentional model
'''

class TwoStageAtt:
    '''
    TwoStageAtt:
    ----------------
    build a two stage simple feed forward attention model!
    methods:
        __init__ args:
            n_time: int, number of timesteps
            n_class: int, output size
            dense: list of ints, same length as dropout, number of nodes
            drop: list of ints, same length as dense, amount of dropout per node
            model_id: str, if not None where the weights are located, minus file
            extension

        compile args:
            *args: args to compiler
            **kargs: kwargs to compiler

        pretrain args:
            *args: args to fit
            **args: kwargs to fit

        adapt args:
            *args: args to fit
            **args: kwargs to fit

        get_model:
            returns the model, useful for not having to type as much
    '''

    def __init__(self, n_time: int, n_class: int,
                 dense: list= [128, 256, 512],
                 drop: list=[0.1, 0.1, 0.1],
                 model_id: str=None):
        self.model = self._build_model(n_time, n_class, dense, drop)
        self.compiler_args = []
        self.compiler_kwargs = {}
        if model_id is not None:
            self.model.load_weights(f"{model_id}.h5")

    def _build_model(self, n_time, n_class, dense, drop):
        inputs = Input((n_time, 16))
        x = inputs
        domain = Dense(128, activation='linear', name='domain_layer')(x)
        x = Attention()(domain)
        for d, dr in zip(dense, drop):
            x = Dropout(dr)(x)
            x = Dense(d, activation=Mish())(x)
        outputs = Dense(n_class, activation='softmax')(x)
        return Model(inputs, outputs)

    def compile(self, *args, **kwargs):
        self.compiler_args = args
        self.compiler_kwargs = kwargs
        self.model.compile(*self.compiler_args, **self.compiler_kwargs)

    def pretrain(self, *args, **kwargs):
        self.model.get_layer('domain_layer').trainable = False
        self.model.compile(*self.compiler_args, **self.compiler_kwargs)
        history = self.model.fit(*args, **kwargs)
        return history

    def adapt(self, *args, **kwargs):
        self.model.get_layer('domain_layer').trainable = True
        for layer in self.model.layers:
            if not layer.name.startswith("do"):
                layer.trainable = False
        history = self.model.fit(*args, **kwargs)
        return history

    def get_model(self):
        return self.model
