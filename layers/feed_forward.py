import tensorflow as tf

# Workaround for TF 2.8 concerning IDE imports and lazy loading
from keras.api._v2.keras.layers import (
    Layer,
    TimeDistributed,
    Dense,
)


class FeedForward(Layer):
    def __init__(self, ff_dim=2048, **kwargs):
        # Initialize superclass
        super(FeedForward, self).__init__(**kwargs)

        # Initialize class variables
        self._ff_dim = ff_dim

    def build(self, input_shape):
        self._model_dim = input_shape[-1]

        self._dense_0 = TimeDistributed(Dense(self._ff_dim, activation="relu"))
        self._dense_1 = TimeDistributed(Dense(self._model_dim))

        super(FeedForward, self).build(input_shape)

    def call(self, x, training=False):
        x = self._dense_0(x, training=training)
        x = self._dense_1(x, training=training)
        return x
