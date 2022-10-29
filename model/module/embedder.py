import tensorflow as tf


class KerasEmbedder(tf.keras.layers.Layer):
    def __init__(self, args):
        super(KerasEmbedder, self).__init__()
        self.word_embed = tf.keras.layers.Embedding(args.n_words, args.dim_embedding,
                                                    input_length=args.max_len)

    def call(self, inputs):
        out = inputs
        out = self.word_embed(out)
        return out
