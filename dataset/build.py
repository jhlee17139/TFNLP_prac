import tensorflow as tf
import tensorflow_datasets as tfds
import os
import shutil


def build_dataset(args):
    if args.dataset == 'imdb':
        return load_imdb_data(args)

    elif args.dataset == 'raw_imdb':
        return load_raw_imdb_data(args)

    else:
        return None


def load_imdb_data(args):
    # load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=args.n_words)
    # Pad sequences with max_len
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=args.max_len)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=args.max_len)

    return (x_train, y_train), (x_test, y_test)


def load_raw_imdb_data(args):
    # load raw data
    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

    dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url,
                                      untar=True, cache_dir='.',
                                      cache_subdir='')
    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    train_dir = os.path.join(dataset_dir, 'train')

    # remove unused folders to make it easier to load the data
    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)
    AUTOTUNE = tf.data.AUTOTUNE
    seed = args.seed
    batch_size = args.batch_size

    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed)

    class_names = raw_train_ds.class_names
    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed)

    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/test',
        batch_size=batch_size)

    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds

