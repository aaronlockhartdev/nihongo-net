import tensorflow as tf


def build_dataset():
    train = tf.data.TextLineDataset("./data/split/train")
    train = train.map(lambda x: tf.strings.split(x, "\t"))

    test = tf.data.TextLineDataset("./data/split/test")
    test = test.map(lambda x: tf.strings.split(x, "\t"))

    AUTOTUNE = tf.data.AUTOTUNE
    train = train.cache().prefetch(AUTOTUNE)
    test = test.cache().prefetch(AUTOTUNE)

    return train, test
