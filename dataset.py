import tensorflow as tf
import tensorflow_datasets as tfds


def prepare_dataset(unlabeled_batch_size, labeled_batch_size):
    unlabeled_dataset_size = 100000
    labeled_dataset_size = 5000

    unlabeled_train_dataset = (
        tfds.load("stl10", split="unlabelled", as_supervised=True, shuffle_files=True)
        .shuffle(buffer_size=5000)
        .batch(unlabeled_batch_size, drop_remainder=True)
    )

    num_repeats = (unlabeled_dataset_size // unlabeled_batch_size) // (
        labeled_dataset_size // labeled_batch_size
    )
    labeled_train_dataset = (
        tfds.load("stl10", split="train", as_supervised=True, shuffle_files=True)
        .shuffle(buffer_size=5000)
        .repeat(num_repeats)
        .batch(labeled_batch_size, drop_remainder=True)
    )
    print(f"The labeled dataset is repeated {num_repeats} times.")

    # labeled and unlabeled datasets are zipped together
    train_dataset = tf.data.Dataset.zip(
        (unlabeled_train_dataset, labeled_train_dataset)
    ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    eval_train_dataset = tfds.load(
        "stl10", split="train", as_supervised=True, shuffle_files=True
    ).batch(labeled_batch_size, drop_remainder=True)
    eval_test_dataset = (
        tfds.load("stl10", split="test", as_supervised=True)
        .batch(labeled_batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    return train_dataset, eval_train_dataset, eval_test_dataset