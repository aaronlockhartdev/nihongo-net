import tensorflow as tf

from tensorflow.keras.layers import (
    Layer,
    LayerNormalization
)
from mixed_multihead_attention import MixedMultiHeadAttention
from feed_forward_network import FeedForwardNetwork

class Encoder(Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        self._self_attention = MixedMultiHeadAttention( num_heads=48,
                                key_dim=64,
                                local_scope=4,
                                num_timesteps=128,
                                num_features=3072,
                                dropout=0.2)
        self._feed_forward = FeedForwardNetwork()
        self._layer_norm = LayerNormalization()

    def call(self, x, training=False):
        tmp = tf.cast(x, tf.float16)
        x = self._self_attention(x, x, training=training)
        x = tf.add(tmp, x)
        x = self._layer_norm(x)

        tmp = tf.cast(x, tf.float16)
        x = self._feed_forward(x, training=training)
        x = tf.add(tmp, x)
        x = self._layer_norm(x)

        return x


