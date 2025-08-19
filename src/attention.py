import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class Attention(Layer):
    """A simple attention mechanism for sequence outputs.
    Computes weights over timesteps and returns context vector.
    Output shape: (batch, features)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, timesteps, features)
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, x):
        # x shape: (batch, timesteps, features)
        e = K.tanh(K.dot(x, self.W) + self.b)  # (batch, timesteps, 1)
        a = K.softmax(e, axis=1)               # (batch, timesteps, 1)
        context = K.sum(a * x, axis=1)         # (batch, features)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
