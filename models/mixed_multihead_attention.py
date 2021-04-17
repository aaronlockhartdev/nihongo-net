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
    """
    MixedMultiHeadAttention layer.

    Implementation of mixed, multi-headed attention (MMA) based on "Mixed Multi-Head 
    Self-Attention for Neural Machine Translation". 

    This layer first applies learned projections to `query`, `key`, and `value`. These
    are a 2D matrix of tensors of the shape (4, num_heads / 4).

    The resulting query and key tensors are dot-product and scaled by 1 / sqrt(key_dim),
    then a different mask is added to each of the 4 projection groups corresponding to
    g (global), l (local), f (forward), and b (backward) and consisting of values 0 or
    -inf. The output is then softmaxed to obtain attention probabilities (all -inf values
    become 0 after softmax). In training, dropout is applied, and the dot-product of the new, 
    sparse probablities and the value tensor is taken.

    The result is flattened along the axes corresponding to the 4 projection groups and the
    `num_heads` projections, effectively concatenating the attention outputs of each set of
    learned qkv projections along the last axis. Finally, another projection is applied to each
    timestep, bringing the shape back to the original.

    Arguments:
        num_heads: Number of attention heads.
        key_dim: Size of each attention head for query and key.
        local_scope: Size of range considered local for local mask.
        num_timesteps: Number of timesteps.
        num_features: Number of features.
        value_dim:  Size of each attention head for value.
        dropout: Dropout probability.
    Call arguments:
        query: Query `Tensor` of shape `[B, T, dim]`.
        value: Value `Tensor` of shape `[B, S, dim]`.
        key: Optional key `Tensor` of shape `[B, S, dim]`. If not given, will use
            `value` for both `key` and `value`, which is the most common case.
        training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (no dropout).
            Defaults to either using the training mode of the parent layer/model,
            or False (inference) if there is no parent layer.
    
    Returns:
        attention_output: Result of computation.
    """
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

        attention_scores = self._softmax(attention_scores)

        attention_scores_dropout = self._dropout_layer(    attention_scores,
                                                                training=training)

        attention_output = tf.einsum("...bcde,...bcef->...dbcf", attention_scores_dropout, value)

        attention_output = tf.reshape(attention_output, [-1, self._num_timesteps, self._num_mixed_heads * 4 * self._value_dim])

        attention_output = self._output_dense(attention_output)

        return attention_output



        



        



    