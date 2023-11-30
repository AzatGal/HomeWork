import numpy as np
import torch

def accuracy(predicted_class, ground_truth, *args, **kwargs):
    """
        Вычисление точности:
            accuracy = sum( predicted_class == ground_truth ) / N, где N - размер набора данных
        TODO: реализуйте подсчет accuracy
    """
    # raise NotImplementedError
    acc = torch.sum(predicted_class == ground_truth) / predicted_class.size(0)
    return acc


def balanced_accuracy(tp, n, N=37, *args, **kwargs):
    """
        Вычисление точности:
            balanced accuracy = sum( TP_i / N_i ) / N, где
                TP_i - кол-во изображений класса i, для которых предсказан класс i
                N_i - количество изображений набора данных класса i
                N - количество классов в наборе данных
        TODO: реализуйте подсчет balanced accuracy
    """
    # raise NotImplementedError
    balanced_acc = 0
    for i in range(len(n)):
        if n[i] != 0:
            balanced_acc += tp[i] / n[i]
    # balanced_acc = np.sum(TP_i / N_i) / N
    return balanced_acc / N
