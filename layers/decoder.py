from operator import mod
from statistics import mode
import tensorflow as tf

# Workaround for TF 2.8 concerning IDE imports and lazy loading
from keras.api._v2.keras.layers import Layer, Dropout, LayerNormalization

from .mixed_multihead_attention import MixedMultiHeadAttention
from .feed_forward import FeedForward


class Decoder(Layer):
    """
    Decoder layer.

    Decoder layer based on model architectures of "Attention is all you Need" and
    "Mixed Multi-Head Self-Attention for Neural Machine Translation".

    This layer passes its input through three layers: a self-attention layer and an
    encoder-decoder attention layer both using the mixed, multi-head attention layer
    and a feed forward network consisting of several dense layers. Residual connections
    and layer normalization are performed for all three.
    """

    def __init__(
        self,
        num_heads=8,
        model_dim=512,
        key_dim=64,
        value_dim=None,
        ff_dim=2048,
        dropout=0.1,
        seq_len=512,
        **kwargs
    ):
        # Initialize superclass
        super(Decoder, self).__init__(**kwargs)

        # Initialize class variables
        self._num_heads = num_heads
        self._model_dim = model_dim
        self._key_dim = key_dim
        self._value_dim = value_dim if value_dim else key_dim
        self._ff_dim = ff_dim
        self._dropout = dropout
        self._seq_len = seq_len

        # Initialize masked mixed multi-head attention sublayer
        def _init_mask():
            import numpy as np

            mask = np.zeros((self._seq_len, self._seq_len))
            for i in range(self._seq_len):
                for j in range(self._seq_len):
                    if i > j:
                        mask[i, j] = 1

            return tf.constant(mask, dtype=tf.float32)

        self._mmma_layer = MixedMultiHeadAttention(
            num_heads=self._num_heads,
            att_mask=_init_mask(),
            model_dim=self._model_dim,
            key_dim=self._key_dim,
            value_dim=self._value_dim,
        )
        self._mmma_drop_layer = Dropout(self._dropout)
        self._mmma_norm_layer = LayerNormalization()

        # Initialize mixed multi-head attention sublayer
        self._mma_layer = MixedMultiHeadAttention(
            num_heads=self._num_heads,
            model_dim=self._model_dim,
            key_dim=self._key_dim,
            value_dim=self._value_dim,
        )
        self._mma_drop_layer = Dropout(self._dropout)
        self._mma_norm_layer = LayerNormalization()

        # Initialize feed forward sublayer
        self._ff_layer = FeedForward(
            model_dim=self._model_dim,
            ff_dim=self._ff_dim,
        )
        self._ff_drop_layer = Dropout(self._dropout)
        self._ff_norm_layer = LayerNormalization()

    def call(self, x, enc_out, training=False):
        # Masked mixed multi-head attention sublayer
        residual = x
        x = self._mmma_layer(x, x, training=training)
        x = self._mmma_drop_layer(x, training=training)

        x = tf.add(x, residual)
        x = self._mmma_norm_layer(x)

        # Mixed multi-head attention sublayer
        residual = x
        x = self._mma_layer(x, enc_out, training=training)
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
