import tensorflow as tf


class SentimentHead(tf.keras.layers.Layer):
    def __init__(self, args):
        super(SentimentHead, self).__init__()
        self.dense_1 = tf.keras.layers.Dense(args.dim_embedding, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(args.dim_embedding // 2, activation='relu')
        self.classifier = tf.keras.layers.Dense(1, activation='sigmoid')
        self.drop_out = tf.keras.layers.Dropout(args.drop_out)

    def call(self, inputs):
        out = inputs
        out = self.dense_1(out)
        out = self.drop_out(out)
        out = self.dense_2(out)
        out = self.drop_out(out)
        out = self.classifier(out)

        return out
