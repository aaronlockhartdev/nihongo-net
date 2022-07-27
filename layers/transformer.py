import tensorflow as tf

from encoder import Encoder
from decoder import Decoder

# Workaround for TF 2.8 concerning IDE imports and lazy loading
from keras.api._v2.keras import Model
from keras.api._v2.keras.layers import Dense, TimeDistributed, Embedding

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)


class Transformer(Model):
    def __init__(
        self,
        num_layers=6,
        num_heads=8,
        key_dim=64,
        value_dim=None,
        model_dim=512,
        ff_dim=2048,
        dropout=0.1,
        timesteps=512,
        **kwargs
    ):
        super(Transformer, self).__init__(**kwargs)

        self._num_layers = num_layers
        self._num_heads = num_heads
        self._key_dim = key_dim
        self._value_dim = value_dim if value_dim else key_dim
        self._model_dim = model_dim
        self._ff_dim = ff_dim
        self._dropout = dropout
        self._timesteps = timesteps

        self._embedding = Embedding(37000, 512)

        self._encoders = [
            Encoder(
                num_heads=self._num_heads,
                key_dim=self._key_dim,
                value_dim=self._value_dim,
                ff_dim=self._ff_dim,
                dropout=self._dropout,
            )
            for _ in range(self._num_layers)
        ]

        self._decoders = [
            Decoder(
                num_heads=self._num_heads,
                key_dim=self._key_dim,
                value_dim=self._value_dim,
                ff_dim=self._ff_dim,
                dropout=self._dropout,
            )
            for _ in range(self._num_layers)
        ]

        self._dense = TimeDistributed(Dense(self._model_dim, activation="softmax"))

    def call(self, x, training=False):
        x = self._embedding(x)

        for encoder in self._encoders:
            x = encoder(x, training=training)

        enc_out = x

        for decoder in self._decoders:
            x = decoder(x, enc_out, training=training)

        x = self._dense(x)

        return x


if __name__ == "__main__":
    nihongo_net = Transformer()
    nihongo_net.compile(
        optimizer="Adam", loss="cross_entropy", metrics=["cross_entropy", "acc"]
    )
    nihongo_net.build((None, 128))
    nihongo_net.summary()
