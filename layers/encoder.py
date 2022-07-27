import tensorflow as tf

# Workaround for TF 2.8 concerning IDE imports and lazy loading
from keras.api._v2.keras.layers import Layer, Dropout, LayerNormalization

from mixed_multihead_attention import MixedMultiHeadAttention
from feed_forward import FeedForward


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

    def __init__(
        self,
        num_heads=8,
        key_dim=64,
        value_dim=None,
        ff_dim=2048,
        dropout=0.1,
        **kwargs
    ):
        # Initialize superclass
        super(Encoder, self).__init__(**kwargs)

        # Initialize class variables
        self._num_heads = num_heads
        self._key_dim = key_dim
        self._value_dim = value_dim if value_dim else key_dim
        self._ff_dim = ff_dim
        self._dropout = dropout

        # Initialize mixed multi-head attention sublayer
        self._mma_layer = MixedMultiHeadAttention(
            num_heads=self._num_heads,
            key_dim=self._key_dim,
            value_dim=self._value_dim,
        )
        self._mma_drop_layer = Dropout(self._dropout)
        self._mma_norm_layer = LayerNormalization()

        # Initialize feed forward sublayer
        self._ff_layer = FeedForward(ff_dim=self._ff_dim)
        self._ff_drop_layer = Dropout(self._dropout)
        self._ff_norm_layer = LayerNormalization()

    def call(self, x, training=False):
        # Mixed multi-head attention sublayer
        residual = x

        x = self._mma_layer(x, x, training=training)
        x = self._mma_drop_layer(x, training=training)

        x = tf.add(x, residual)
        x = self._mma_norm_layer(x)

        # Feed forward sublayer
        residual = x

        x = self._ff_layer(x, training=training)
        x = self._ff_drop_layer(x, training=training)

        x = tf.add(x, residual)
        x = self._ff_norm_layer(x)

        return x
