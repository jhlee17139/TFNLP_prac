import tensorflow as tf
from tensorflow import keras


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert (
            embed_dim % num_heads == 0
        ), "embedding dimension not divisible by num heads"
        self.projection_dim = embed_dim // num_heads
        self.wq = keras.layers.Dense(embed_dim)
        self.wk = keras.layers.Dense(embed_dim)
        self.wv = keras.layers.Dense(embed_dim)
        self.combine_heads = keras.layers.Dense(embed_dim)

    def attention(self, q, k, v):
        score = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dk)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, v)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x_q, x_k, x_v):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(x_q)[0]
        q = self.wq(x_q)  # (batch_size, seq_len, embed_dim)
        k = self.wk(x_k)  # (batch_size, seq_len, embed_dim)
        v = self.wv(x_v)  # (batch_size, seq_len, embed_dim)
        q = self.separate_heads(
            q, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        k = self.separate_heads(
            k, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        v = self.separate_heads(
            v, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(q, k, v)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.att = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x_q, x_k, x_v):
        attn_output = self.att(x_q, x_k, x_v)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x_q + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class EmbeddingLayer(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, emded_dim):
        super(EmbeddingLayer, self).__init__()
        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=emded_dim
        )
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=emded_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


# prac 2
class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.self_att = MultiHeadAttention(embed_dim, num_heads)
        self.cross_att = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x_q, x_k, x_v):
        # 빈칸
        print('빈칸')

        # self_attention

        # cross_attention

        return out2


'''
class TransformerClassifier(tf.keras.Model):
    def __init__(self, maxlen, vocab_size, embed_dim, ff_dim, num_heads):
        super(TransformerClassifier, self).__init__()
        self.emb = EmbeddingLayer(maxlen, vocab_size, embed_dim)
        self.transformer = TransformerLayer(embed_dim, num_heads, ff_dim)
        self.prehead = keras.layers.Dense(20, activation="relu")
        self.dropout1 = keras.layers.Dropout(0.05)
        self.dropout2 = keras.layers.Dropout(0.05)
        self.head = keras.layers.Dense(2, activation="softmax")

    def call(self, x):
        x = self.emb(x)
        x_q, x_k, x_v = x, x, x
        x = self.transformer(x_q, x_k, x_v)
        x = tf.math.reduce_mean(x, axis=1)
        x = self.dropout1(x)
        x = self.prehead(x)
        x = self.dropout2(x)
        x = self.head(x)
        return x
'''

