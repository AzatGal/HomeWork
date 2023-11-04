import numpy as np


def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    pass


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    # TODO calculate accuracy for multiclass
    targets = np.argmax(targets, axis=1)
    predictions = np.argmax(predictions, axis=1)
    acc = np.sum(predictions == targets) / np.shape(targets)[0]
    return acc


def confusion_matrix(predictions: np.ndarray, targets: np.ndarray, *args, **kwargs):
    size = np.shape(targets)[1]
    conf_matrix = np.zeros((size, size))
    size = np.shape(targets)[0]
    targets = np.argmax(targets, axis=1)
    predictions = np.argmax(predictions, axis=1)
    for i in range(size):
        conf_matrix[predictions[i], targets[i]] += 1
    return conf_matrix
