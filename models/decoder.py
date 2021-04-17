import tensorflow as tf

from tensorflow.keras.layers import (
    Layer,
    LayerNormalization
)
from mixed_multihead_attention import MixedMultiHeadAttention
from feed_forward_network import FeedForwardNetwork

class Decoder(Layer):
    def __init__(self):
        super(Decoder, self).__init__()
        self._self_attention = MixedMultiHeadAttention( num_heads=48,
                                key_dim=64,
                                local_scope=4,
                                num_timesteps=128,
                                num_features=3072,
                                dropout=0.2)
        self._encoder_decoder_attention = MixedMultiHeadAttention(  num_heads=48,
                                            key_dim=64,
                                            local_scope=4,
                                            num_timesteps=128,
                                            num_features=3072,
                                            dropout=0.2)
        self._feed_forward = FeedForwardNetwork()
        self._layer_norm = LayerNormalization()

    def call(self, x, state, training=False):
        tmp = x
        x = self._self_attention(x, x, training=training)
        x = tf.add(tmp, x)
        x = self._layer_norm(x)

        tmp = x
        x = self._encoder_decoder_attention(x, state, training=training)
        x = tf.add(tmp, x)
        x = self._layer_norm(x)


        tmp = x
        x = self._feed_forward(x, training=training)
        x = tf.add(tmp, x)
        x = self._layer_norm(x)

        return x


