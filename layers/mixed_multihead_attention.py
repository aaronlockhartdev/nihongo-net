import math
import tensorflow as tf

# Workaround for TF 2.8 concerning IDE imports and lazy loading
from keras.api._v2.keras.layers import Layer, Softmax
from keras.api._v2.keras.layers.experimental import EinsumDense


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
    become 0 after softmax).

    The result is flattened along the axes corresponding to the 4 projection groups and the
    `num_heads` projections, effectively concatenating the attention outputs of each set of
    learned qkv projections along the last axis. Finally, another projection is applied to each
    timestep, bringing the shape back to the original.

    Arguments:
        num_heads: Number of attention heads. Must be divisible evenly by 4.
        key_dim: Size of each attention head for query and key.
        value_dim:  Size of each attention head for value.
    Build arguments:
        input_shape: Shape of input tensor. Expected to be [batch_size, timesteps, model_dim].
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

    def __init__(
        self,
        num_heads=8,
        local_scope=4,
        att_mask=None,
        seq_len=512,
        model_dim=512,
        key_dim=64,
        value_dim=None,
        **kwargs,
    ):
        # Initialize superclass
        super(MixedMultiHeadAttention, self).__init__(**kwargs)

        assert num_heads % 4 == 0

        # Initialize class variables
        self._num_heads = num_heads
        self._local_scope = local_scope
        self._att_mask = att_mask
        self._seq_len = seq_len
        self._model_dim = model_dim
        self._key_dim = key_dim
        self._value_dim = value_dim if value_dim else key_dim

        # Initialize learned linear projections for query, key, and value
        self._query_projection = EinsumDense(
            "abx,cxd->acbd",
            output_shape=(self._num_heads, self._seq_len, self._key_dim),
        )

        self._key_projection = EinsumDense(
            "abx,cxd->acbd",
            output_shape=(self._num_heads, self._seq_len, self._key_dim),
        )

        self._value_projection = EinsumDense(
            "abx,cxd->acbd",
            output_shape=(self._num_heads, self._seq_len, self._key_dim),
        )

        # Initialize softmax layer
        self._softmax = Softmax(axis=[-1, -2])

        # Initialize learned projections of attention heads
        self._output_projection = EinsumDense(
            equation="axby,xyc->abc",
            output_shape=(self._seq_len, self._model_dim),
        )

        # Expand dims for att_mask
        if self._att_mask is not None:
            while len(self._att_mask.shape) < 4:
                self._att_mask = tf.expand_dims(self._att_mask, axis=0)

        # Initialize mixed masks
        def _init_masks():
            import numpy as np

            g = np.zeros((self._seq_len, self._seq_len))
            l = np.zeros((self._seq_len, self._seq_len))
            f = np.zeros((self._seq_len, self._seq_len))
            b = np.zeros((self._seq_len, self._seq_len))

            for i in range(self._seq_len):
                for j in range(self._seq_len):
                    if not (i - self._local_scope <= j <= i + self._local_scope):
                        l[i, j] = np.NINF
                    if i > j:
                        f[i, j] = np.NINF
                    if i < j:
                        b[i, j] = np.NINF

            masks = tf.repeat(
                tf.constant([g, l, f, b], dtype=tf.float32),
                repeats=self._num_heads // 4,
                axis=0,
            )
            return masks

        self._mixed_masks = _init_masks()

    def call(self, query, value, key=None, training=False):
        if not key:
            key = value

        # Apply learned projections to query, key, and value
        query = self._query_projection(query)
        key = self._key_projection(key)
        value = self._value_projection(value)

        # Compute scaled attention probabilities
        att_probs = tf.einsum("...ax,...bx->...ab", query, key)
        att_probs = tf.divide(att_probs, math.sqrt(self._key_dim))

        # Add mixed masks to attention probabilities
        att_probs = tf.add(att_probs, self._mixed_masks)

        # Apply softmax to num_heads * 2D attention arrays
        if self._att_mask is not None:
            att_probs = self._softmax(att_probs, self._att_mask)
        else:
            att_probs = self._softmax(att_probs)

        # Compute attention values
        att_vals = tf.einsum("...ax,...xb->...ab", att_probs, value)

        att_vals = self._output_projection(att_vals)

        return att_vals
