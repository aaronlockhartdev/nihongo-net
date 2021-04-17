import tensorflow as tf
import numpy as np
import math

from tensorflow.keras.layers import (
    Layer,
    Dropout,
    Softmax,
)
from tensorflow.python.keras.layers.einsum_dense import EinsumDense

from tensorflow.python.keras.utils import tf_utils

class MixedMultiHeadAttention(Layer):
    def __init__(   self,
                    num_heads,
                    key_dim,
                    local_scope,
                    num_timesteps,
                    num_features,
                    value_dim=None,
                    dropout=0.0):
        super(MixedMultiHeadAttention, self).__init__()

        assert num_heads % 4 == 0

        self._num_mixed_heads = int(num_heads / 4)
        self._key_dim = key_dim
        self._local_scope = local_scope
        self._num_timesteps = num_timesteps
        self._num_features = num_features
        self._value_dim = value_dim if value_dim else key_dim
        self._dropout = dropout

        self._query_dense = EinsumDense(    equation="abe,cdef->acdbf",
                                            output_shape=(  4,
                                                            self._num_mixed_heads,
                                                            self._num_timesteps,
                                                            self._key_dim),
                                            bias_axes="f")

        self._key_dense = EinsumDense(      equation="abe,cdef->acdfb",
                                            output_shape=(  4,
                                                            self._num_mixed_heads,
                                                            self._key_dim,
                                                            self._num_timesteps),
                                            bias_axes="f")

        self._value_dense = EinsumDense(    equation="abe,cdef->acdbf",
                                            output_shape=(  4,
                                                            self._num_mixed_heads,
                                                            self._num_timesteps,
                                                            self._value_dim),
                                            bias_axes="f")

        self._softmax = Softmax()
        self._dropout_layer = Dropout(self._dropout)

        self._output_dense = EinsumDense(   equation="abc,cd->abd",
                                            output_shape=(  self._num_timesteps,
                                                            self._num_features),
                                            bias_axes="d")

        with tf_utils.maybe_init_scope(self):
            g = np.zeros((self._num_timesteps, self._num_timesteps))
            l = np.zeros((self._num_timesteps, self._num_timesteps))
            f = np.zeros((self._num_timesteps, self._num_timesteps))
            b = np.zeros((self._num_timesteps, self._num_timesteps))

            for i in range(self._num_timesteps):
                for j in range(self._num_timesteps):
                    if i - self._local_scope > j or j > i + self._local_scope:
                        l[i,j] = np.NINF
                    if i > j:
                        f[i,j] = np.NINF
                    if i < j:
                        b[i,j] = np.NINF

            m = np.stack([g, l, f, b])
            m = tf.convert_to_tensor(m, tf.float16)
            m = tf.expand_dims(m, 1)
            m = tf.expand_dims(m, 0)

            self._masks = m

    def call(   self,
                query,
                value,
                key=None,
                training=None):
        if key is None:
            key = value

        query = self._query_dense(query)

        key = self._key_dense(key)

        value = self._value_dense(value)


        query = tf.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))

        attention_scores = tf.einsum("...ab,...bc->...ac", query, key)

        attention_scores = tf.add(attention_scores, self._masks)

        attention_scores = self._softmax.call(attention_scores)

        attention_scores_dropout = self._dropout_layer.call(    attention_scores,
                                                                training=training)

        attention_output = tf.einsum("abcde,abcef->adbcf", attention_scores_dropout, value)

        attention_output = tf.reshape(attention_output, [-1, self._num_timesteps, self._num_mixed_heads * 4 * self._value_dim])

        attention_output = self._output_dense(attention_output)

        return attention_output



        



        



    