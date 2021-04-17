import tensorflow as tf

from tensorflow.keras.layers import (
    Layer,
    TimeDistributed,
    Dense,
)

class FeedForwardNetwork(Layer):
    def __init__(self):
        super(FeedForwardNetwork, self).__init__()
        self._dense_0 = TimeDistributed(Dense(12288, activation="relu"))
        self._dense_1 = TimeDistributed(Dense(3072, activation="relu"))

    def call(self, x, training=False):
        x = self._dense_0.call(x, training=training)
        x = self._dense_1.call(x, training=training)
        return x