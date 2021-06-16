# Semi-supervised Representation Learning for Image Classification with Keras

This repository contains an implementation for 3 semi-supervised methods:
- [InfoNCE](https://arxiv.org/abs/1807.03748)
- [SuNCEt](https://arxiv.org/abs/2006.10803)
- [PAWS](https://arxiv.org/abs/2104.13963)

The trained encoders do not have a classification head, all methods are evaluated using the accuracy of a k-nearest neighbour classifier.