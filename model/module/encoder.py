import tensorflow as tf
from model.module import transformer


# prac 1
class CNNBasedEncoder(tf.keras.layers.Layer):
    def __init__(self, args):
        super(CNNBasedEncoder, self).__init__()
        self.cnn_layers = []
        for i in range(args.encoder_n_layer):
            self.cnn_layers.append(
                tf.keras.layers.Conv1D(args.dim_embedding, args.cnn_kernel_size, padding='same', activation='relu'
                                       , name='cnn_layer_{}'.format(str(i)))
            )

    def call(self, inputs):
        out = inputs

        for cnn_layer in self.cnn_layers:
            out = cnn_layer(out)

        return out


# prac 1
class LSTMBasedEncoder(tf.keras.layers.Layer):
    def __init__(self, args):
        super(LSTMBasedEncoder, self).__init__()
        self.lstm_layers = []
        for i in range(args.encoder_n_layer):
            self.lstm_layers.append(
                tf.keras.layers.LSTM(args.dim_embedding, return_sequences=True, name='lstm_layer_{}'.format(str(i)))
            )

    def call(self, inputs):
        out = inputs

        for lstm_layer in self.lstm_layers:
            out = lstm_layer(out)

        return out


# prac_1
# Bidirectional : https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional
class BiLSTMBasedEncoder(tf.keras.layers.Layer):
    def __init__(self, args):
        super(BiLSTMBasedEncoder, self).__init__()
        self.lstm_layers = []
        for i in range(args.encoder_n_layer):
            # 빈칸 작성
            print('빈칸 작성')

    def call(self, inputs):
        out = inputs

        for lstm_layer in self.lstm_layers:
            out = lstm_layer(out)

        return out


class TransformerBasedEncoder(tf.keras.layers.Layer):
    def __init__(self, args):
        super(TransformerBasedEncoder, self).__init__()
        self.transformer_layers = []
        for i in range(args.encoder_n_layer):
            self.transformer_layers.append(
                transformer.TransformerEncoderLayer(embed_dim=args.dim_embedding,
                                                    num_heads=args.encoder_n_head,
                                                    ff_dim=args.dim_embedding)
            )

    def call(self, inputs):
        x = inputs
        x_q, x_k, x_v = x, x, x

        for transformer_layer in self.transformer_layers:
            x_q = transformer_layer(x_q, x_k, x_v)
            x_k = x_q
            x_v = x_q

        return x_q
