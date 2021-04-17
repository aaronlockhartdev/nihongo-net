import tensorflow as tf

from tensorflow.keras.layers import (
    Layer,
    LayerNormalization
)
from mixed_multihead_attention import MixedMultiHeadAttention
from feed_forward_network import FeedForwardNetwork

class Encoder(Layer):
    """
    Encoder layer.

    Encoder layer based on model architectures of "Attention is all you Need" and 
    "Mixed Multi-Head Self-Attention for Neural Machine Translation".

    This layer passes its input through two layers: a self-attention layer using
    the mixed, multi-head attention layer and a feed forward network consisting of
    several dense layers. Residual connections and layer normalization are performed
    for both.
    """
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


