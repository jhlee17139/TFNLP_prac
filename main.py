import argparse
from dataset import build as dataset_build
from model import build as model_build


def parse_args():
    parser = argparse.ArgumentParser(
        description='Machine Learning Homework 4 : NLP')

    # model setting
    '''
    word_embed : keras_embedding or positional_encoding or bert
    encoder : cnn or lstm or bilstm or transformer or bert
    decoder : average_pooling or transformer
    head : sentiment_head
    '''
    parser.add_argument('--word_embed', default='keras_embedding', type=str)
    parser.add_argument('--encoder', default='cnn', type=str)
    parser.add_argument('--decoder', default='average_pooling', type=str)
    parser.add_argument('--head', default='sentiment_head', type=str)

    # dataset setting
    # imdb or raw_imdb
    parser.add_argument('--dataset', default='imdb', type=str)

    # imdb dataset setting
    parser.add_argument('--n_words', default=10000, type=int)
    parser.add_argument('--max_len', default=128, type=int)
    # 256
    parser.add_argument('--dim_embedding', default=256, type=int)

    # encoder
    parser.add_argument('--encoder_n_layer', default=2, type=int)

    # cnn encoder
    parser.add_argument('--cnn_kernel_size', default=3, type=int)

    # transformer
    parser.add_argument('--encoder_n_head', default=8, type=int)

    # decoder
    parser.add_argument('--decoder_n_layer', default=2, type=int)
    parser.add_argument('--decoder_n_head', default=4, type=int)

    # bert
    parser.add_argument('--bert_model_name', default='small_bert/bert_en_uncased_L-4_H-256_A-4', type=str)

    # hyper parameter
    parser.add_argument('--drop_out', default=0.05, type=float)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--loss', default='binary_crossentropy', type=str)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--seed', default=42, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = model_build.build_model(args)
    model.summary()
    model.compile(optimizer=args.optimizer, loss=args.loss, metrics=["accuracy"])

    if args.dataset == 'imdb':
        (x_train, y_train), (x_test, y_test) = dataset_build.build_dataset(args)
        score = model.fit(x_train, y_train,
                          epochs=args.epochs,
                          batch_size=args.batch_size,
                          validation_data=(x_test, y_test))
        print("\nTest loss:", score.history['val_loss'][-1])
        print('Test accuracy:', score.history['val_accuracy'][-1])

    elif args.dataset == 'raw_imdb':
        train_ds, val_ds, test_ds = dataset_build.build_dataset(args)
        score = model.fit(x=train_ds,
                          validation_data=val_ds,
                          epochs=args.epochs)
        print("\nTest loss:", score.history['val_loss'][-1])
        print('Test accuracy:', score.history['val_accuracy'][-1])

    else:
        print("args.dataset error")
        exit()
