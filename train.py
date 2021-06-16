import os
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # suppress info-level logs
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from dataset import prepare_dataset
from augmentations import RandomResizedCrop, RandomColorJitter
from algorithms import InfoNCE, SuNCEt, PAWS
from models import KNNClassifier

tf.get_logger().setLevel("WARN")  # suppress info-level logs

# hyperparameters
num_epochs = 30
width = 128
k_values = (20, 200)
batch_sizes = {  # unlabeled, labeled
    InfoNCE: (500, 25),
    SuNCEt: (250, 250),
    PAWS: (250, 250),
}
hyperparams = {
    InfoNCE: {"temperature": 0.1},
    SuNCEt: {"temperature": 0.1, "supervised_loss_weight": 1.0},
    PAWS: {"temperature": 0.1, "sharpening": 0.25},
}

# select an algorithm
Algorithm = InfoNCE  # InfoNCE, SuNCEt, PAWS

# load STL10 dataset
train_dataset, eval_train_dataset, eval_test_dataset = prepare_dataset(
    *batch_sizes[Algorithm]
)

model = Algorithm(
    augmenter=keras.Sequential(
        [
            layers.Input(shape=(96, 96, 3)),
            preprocessing.Rescaling(1 / 255),
            preprocessing.RandomFlip("horizontal"),
            RandomResizedCrop(scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3)),
            RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        ],
        name="augmenter",
    ),
    encoder=keras.Sequential(
        [
            layers.Input(shape=(96, 96, 3)),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Flatten(),
            layers.Dense(width),
        ],
        name="encoder",
    ),
    **hyperparams[Algorithm]
)
evaluators = [
    KNNClassifier(
        train_dataset=eval_train_dataset,
        test_dataset=eval_test_dataset,
        model=model,
        k=k,
    )
    for k in k_values
]

# optimizers
model.compile(
    optimizer=keras.optimizers.Adam(),
)

# run training
history = model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=eval_test_dataset,
    callbacks=[
        keras.callbacks.LambdaCallback(on_epoch_end=evaluator.evaluate)
        for evaluator in evaluators
    ],
)

# save history
with open("{}.pkl".format(Algorithm.__name__), "wb") as write_file:
    pickle.dump(history.history, write_file)
