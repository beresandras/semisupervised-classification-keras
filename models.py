import tensorflow as tf

from abc import abstractmethod
from tensorflow import keras


class SemisupervisedModel(keras.Model):
    def __init__(self, augmenter, encoder):
        super().__init__()

        self.augmenter = augmenter
        self.encoder = encoder

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)

        self.optimizer = optimizer
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    @abstractmethod
    def semisupervised_loss(
        self, labels, labeled_features, unlabeled_features_1, unlabeled_features_2
    ):
        pass

    def call(self, images, training):
        augmented_images = self.augmenter(images, training=training)
        features = self.encoder(augmented_images, training=training)
        return features

    def train_step(self, data):
        (unlabeled_images, _), (labeled_images, labels) = data

        labeled_images = self.augmenter(labeled_images)
        unlabeled_images_1 = self.augmenter(unlabeled_images)
        unlabeled_images_2 = self.augmenter(unlabeled_images)
        with tf.GradientTape() as tape:
            labeled_features = self.encoder(labeled_images)
            unlabeled_features_1 = self.encoder(unlabeled_images_1)
            unlabeled_features_2 = self.encoder(unlabeled_images_2)
            loss = self.semisupervised_loss(
                labels, labeled_features, unlabeled_features_1, unlabeled_features_2
            )
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.loss_tracker.update_state(loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, labels = data

        labeled_images = self.augmenter(images, training=False)
        unlabeled_images_1 = self.augmenter(images)
        unlabeled_images_2 = self.augmenter(images)

        labeled_features = self.encoder(labeled_images, training=False)
        unlabeled_features_1 = self.encoder(unlabeled_images_1, training=False)
        unlabeled_features_2 = self.encoder(unlabeled_images_2, training=False)

        # the InfoNCE validation loss will be lower the the train one,
        # because the labeled batch sizes are smaller
        loss = self.semisupervised_loss(
            labels, labeled_features, unlabeled_features_1, unlabeled_features_2
        )
        self.loss_tracker.update_state(loss)

        return {m.name: m.result() for m in self.metrics}


class KNNClassifier:
    def __init__(self, train_dataset, test_dataset, model, k):

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.k = k

        self.feature_dimensions = self.model.encoder.output_shape[1]
        self.bank_size = 5000
        self.num_classes = 10

        self.feature_bank = tf.Variable(
            tf.zeros(shape=(self.bank_size, self.feature_dimensions), dtype=tf.float32),
            trainable=False,
        )
        self.label_bank = tf.Variable(
            tf.zeros(shape=(self.bank_size,), dtype=tf.int64), trainable=False
        )
        self.bank_index = 0

        self.accuracy = keras.metrics.SparseCategoricalAccuracy()

    def empty_bank(self):
        self.feature_bank.assign(
            tf.zeros(shape=(self.bank_size, self.feature_dimensions), dtype=tf.float32)
        )
        self.label_bank.assign(tf.zeros(shape=(self.bank_size,), dtype=tf.int64))
        self.bank_index = 0

    def append_bank(self, features, labels):
        batch_size = tf.shape(features)[0]

        features = tf.math.l2_normalize(features, axis=1)
        self.feature_bank[self.bank_index : self.bank_index + batch_size].assign(
            features
        )
        self.label_bank[self.bank_index : self.bank_index + batch_size].assign(labels)
        self.bank_index += batch_size

    def evaluate(self, epoch, logs):
        self.empty_bank()
        for images, labels in self.train_dataset:
            features = self.model(images, training=False)
            self.append_bank(features, labels)

        self.accuracy.reset_states()
        for images, labels in self.test_dataset:
            features = self.model(images, training=False)
            self.accuracy.update_state(
                labels,
                self.predict(features),
            )

        accuracy = self.accuracy.result().numpy()
        logs["val_knn_acc"] = accuracy
        tf.print(f" val_{self.k}-nn_acc: {accuracy:.4f}")

    def predict(self, query_features):
        query_features = tf.math.l2_normalize(query_features, axis=1)
        similarities = query_features @ tf.transpose(
            self.feature_bank
        )  # batch_size x bank_size

        _, top_indices = tf.math.top_k(similarities, self.k)  # batch_size x k
        top_labels = tf.gather(self.label_bank, top_indices)  # batch_size x k
        class_scores = tf.reduce_sum(
            tf.one_hot(top_labels, self.num_classes), axis=1
        )  # batch_size x num_classes
        return class_scores