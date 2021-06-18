import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from models import SemisupervisedModel


class CrossEntropy(SemisupervisedModel):
    def __init__(self, augmenter, encoder):
        super().__init__(augmenter, encoder)

        self.num_classes = 10
        self.classifier_head = keras.Sequential(
            [
                layers.Input(shape=encoder.output_shape[1]),
                layers.ReLU(),  # activation on the output of the encoder
                layers.Dense(self.num_classes),
            ],
            name="classifier_head",
        )

    def semisupervised_loss(
        self, labels, labeled_features, unlabeled_features_1, unlabeled_features_2
    ):
        # supervised baseline: uses only labeled images
        # features -> logits -> softmax + cross-entropy
        class_logits = self.classifier_head(labeled_features)
        return keras.losses.sparse_categorical_crossentropy(
            labels, class_logits, from_logits=True
        )


class InfoNCE(SemisupervisedModel):
    def __init__(self, augmenter, encoder, temperature):
        super().__init__(augmenter, encoder)

        self.temperature = temperature

    def semisupervised_loss(
        self, labels, labeled_features, unlabeled_features_1, unlabeled_features_2
    ):
        # self-supervised baseline: uses only unlabeled images
        # 2 x features -> pairwise similarity -> temperature scaling + softmax + cross-entropy
        # labels = the pairs of features from the same image should be most similar
        batch_size = tf.shape(unlabeled_features_1)[0]

        unlabeled_features_1 = tf.math.l2_normalize(unlabeled_features_1, axis=1)
        unlabeled_features_2 = tf.math.l2_normalize(unlabeled_features_2, axis=1)
        similarities = (
            unlabeled_features_1 @ tf.transpose(unlabeled_features_2) / self.temperature
        )
        nn_labels = tf.range(batch_size)
        loss = keras.losses.sparse_categorical_crossentropy(
            nn_labels, similarities, from_logits=True
        )

        return loss


class SuNCEt(InfoNCE):
    def __init__(self, augmenter, encoder, temperature, supervised_loss_weight):
        super().__init__(augmenter, encoder, temperature)

        self.supervised_loss_weight = supervised_loss_weight
        self.num_classes = 10

    def semisupervised_loss(
        self, labels, labeled_features, unlabeled_features_1, unlabeled_features_2
    ):
        # self-supervised contrastive loss (InfoNCE) + supervised contrastive loss (SuNCEt)
        self_supervised_loss = super().semisupervised_loss(
            labels, labeled_features, unlabeled_features_1, unlabeled_features_2
        )

        labeled_batch_size = tf.shape(labeled_features)[0]
        labeled_features = tf.math.l2_normalize(labeled_features, axis=1)
        similarities = (
            labeled_features @ tf.transpose(labeled_features) - 1.0
        ) / self.temperature
        nn_scores = tf.exp(similarities) * (1.0 - tf.eye(labeled_batch_size))
        nn_probabilities = nn_scores / tf.reduce_sum(nn_scores, axis=1, keepdims=True)
        class_probabilities = nn_probabilities @ tf.one_hot(labels, self.num_classes)
        supervised_loss = keras.losses.sparse_categorical_crossentropy(
            labels, class_probabilities
        )

        return self_supervised_loss + self.supervised_loss_weight * supervised_loss


class PAWS(SemisupervisedModel):
    def __init__(self, augmenter, encoder, temperature, sharpening):
        super().__init__(augmenter, encoder)

        self.temperature = temperature
        self.sharpening = sharpening

        self.num_classes = 10

    def soft_nn_predict(self, labels, labeled_features, unlabeled_features):
        labeled_features = tf.math.l2_normalize(labeled_features, axis=1)
        unlabeled_features = tf.math.l2_normalize(unlabeled_features, axis=1)
        similarities = unlabeled_features @ tf.transpose(labeled_features)
        nn_probabilities = keras.activations.softmax(similarities / self.temperature)
        class_probabilities = nn_probabilities @ tf.one_hot(labels, self.num_classes)
        return class_probabilities

    def sharpen(self, probabilities):
        probabilities = probabilities ** (1 / self.sharpening)  # sharpening
        probabilities /= tf.reduce_sum(
            probabilities, axis=1, keepdims=True
        )  # renormalization
        return probabilities

    def semisupervised_loss(
        self, labels, labeled_features, unlabeled_features_1, unlabeled_features_2
    ):
        # self-labeling with a soft-nn classifier using the labeled batch
        pred_probabilities_1 = self.soft_nn_predict(
            labels, labeled_features, unlabeled_features_1
        )
        pred_probabilities_2 = self.soft_nn_predict(
            labels, labeled_features, unlabeled_features_2
        )

        pred_probabilities = tf.concat(
            [pred_probabilities_2, pred_probabilities_1], axis=0
        )
        target_probabilities = tf.concat(
            [self.sharpen(pred_probabilities_1), self.sharpen(pred_probabilities_2)],
            axis=0,
        )
        mean_target_probabilities = tf.reduce_mean(target_probabilities, axis=0)

        # the predicted label distribution should be close to its sharper version
        # the average predicted label distribution should be uniform
        loss = keras.losses.categorical_crossentropy(
            tf.stop_gradient(target_probabilities),
            pred_probabilities,
        ) - keras.losses.categorical_crossentropy(
            mean_target_probabilities,
            mean_target_probabilities,
        )
        return loss