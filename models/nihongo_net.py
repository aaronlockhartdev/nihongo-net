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
        # self.emb = Embedding(None, 3072)

        self.enc1 = Encoder()
        self.enc2 = Encoder()

        self.dec1 = Decoder()
        self.dec2 = Decoder()

        self.lin = TimeDistributed(Dense(3072, activation="softmax"))

    def call(self, x, training=False):
        x = self.enc1.call(x, training=training)
        x = self.enc2.call(x, training=training)

        enc_out = x
        x = self.dec1.call(x, enc_out, training=training)
        x = self.dec2.call(x, enc_out, training=training)

        x = self.lin(x)

        return x

    def prep_build(self):
        self.enc1.build((32, 128, 3072))
        self.enc2.build((32, 128, 3072))

        self.dec1.build((32, 128, 3072))
        self.dec2.build((32, 128, 3072))

if __name__ == "__main__":
    nihongo_net = NihongoNet()
    nihongo_net.compile(optimizer="Adam", loss="cross_entropy", metrics=["cross_entropy", "acc"])
    nihongo_net.prep_build()
    nihongo_net.build((32, 128, 3072))
    nihongo_net.summary()
