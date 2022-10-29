import tensorflow as tf
from model.module import transformer


class GlobalAveragePool(tf.keras.layers.Layer):
    def __init__(self, args):
        super(GlobalAveragePool, self).__init__()
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, inputs):
        out = inputs
        out = self.pooling(out)
        return out


class TransformerBasedDecoder(tf.keras.layers.Layer):
    def __init__(self, args):
        super(TransformerBasedDecoder, self).__init__()
        self.args = args
        self.query = self.add_weight(name='query', shape=(1, args.dim_embedding),
                                     initializer='uniform', trainable=True)
        self.transformer_layers = []

        for i in range(args.decoder_n_layer):
            self.transformer_layers.append(
                transformer.TransformerDecoderLayer(embed_dim=args.dim_embedding,
                                                    num_heads=args.decoder_n_head,
                                                    ff_dim=args.dim_embedding)
            )

    def call(self, inputs):
        x_k, x_v = inputs, inputs
        x_q = self.query
        x_q = tf.expand_dims(tf.repeat(x_q, repeats=[self.args.batch_size, ], axis=0), 1)

        for transformer_layer in self.transformer_layers:
            x_q = transformer_layer(x_q, x_k, x_v)

        x_q = tf.squeeze(x_q, [1, ])
        return x_q
