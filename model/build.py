import tensorflow as tf
from model.module import embedder
from model.module import encoder
from model.module import decoder
from model.module import head
from model.module import transformer
from model.module import bert


def build_model(args):
    word_embed = build_word_embed(args)
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    head = build_head(args)

    assert word_embed is not None and encoder is not None and decoder is not None and head is not None
    if args.word_embed == 'keras_embedding' or args.word_embed == 'positional_encoding':
        input = tf.keras.layers.Input(shape=args.max_len)
        feature = word_embed(input)
        feature = encoder(feature)

    elif args.word_embed == 'bert':
        input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        feature = word_embed(input)
        feature = encoder(feature)['encoder_outputs'][-1]

    else:
        return None

    feature = decoder(feature)
    out = head(feature)
    model = tf.keras.Model(inputs=input, outputs=out)

    return model


def build_word_embed(args):
    if args.word_embed == 'keras_embedding':
        return embedder.KerasEmbedder(args)

    elif args.word_embed == 'positional_encoding':
        return transformer.EmbeddingLayer(maxlen=args.max_len, vocab_size=args.n_words, emded_dim=args.dim_embedding)

    elif args.word_embed == 'bert':
        return bert.get_bert_preprocess(args)

    else:
        return None


def build_encoder(args):
    if args.encoder == 'cnn':
        return encoder.CNNBasedEncoder(args)

    elif args.encoder == 'lstm':
        return encoder.LSTMBasedEncoder(args)

    elif args.encoder == 'bilstm':
        return encoder.BiLSTMBasedEncoder(args)

    elif args.encoder == 'transformer':
        return encoder.TransformerBasedEncoder(args)

    elif args.encoder == 'bert':
        return bert.get_bert_encoder(args)

    else:
        return None


def build_decoder(args):
    if args.decoder == 'average_pooling':
        return decoder.GlobalAveragePool(args)

    elif args.decoder == 'transformer':
        return decoder.TransformerBasedDecoder(args)

    else:
        return None


def build_head(args):
    if args.head == 'sentiment_head':
        return head.SentimentHead(args)

    else:
        return None


