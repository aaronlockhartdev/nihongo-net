import tensorflow as tf

from encoder import Encoder
from decoder import Decoder
from tensorflow.keras import mixed_precision
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    TimeDistributed,
    Embedding
)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

class NihongoNet(Model):
    def __init__(self):
        super(NihongoNet, self).__init__()
        self._embedding = Embedding(30000, 3072, input_length=128)

        self._encoder_0 = Encoder()
        self._encoder_1 = Encoder()

        self._decoder_0 = Decoder()
        self._decoder_1 = Decoder()

        self._dense = TimeDistributed(Dense(3072, activation="softmax"))

    def call(self, x, training=False):
        x = self._embedding(x)
        x = self._encoder_0(x, training=training)
        x = self._encoder_1(x, training=training)

        enc_out = x
        x = self._decoder_0(x, enc_out, training=training)
        x = self._decoder_1(x, enc_out, training=training)

        x = self._dense(x)

        return x

    def prep_build(self):
        self._encoder_0.build((32, 128, 3072))
        self._encoder_1.build((32, 128, 3072))

        self._decoder_0.build((32, 128, 3072))
        self._decoder_1.build((32, 128, 3072))

if __name__ == "__main__":
    nihongo_net = NihongoNet()
    nihongo_net.compile(optimizer="Adam", loss="cross_entropy", metrics=["cross_entropy", "acc"])
    nihongo_net.prep_build()
    nihongo_net.build((32, 128))
    nihongo_net.summary()
