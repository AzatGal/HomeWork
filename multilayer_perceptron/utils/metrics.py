import torch


def accuracy(predicted_class, ground_truth, *args, **kwargs):
    """
        Вычисление точности:
            accuracy = sum( predicted_class == ground_truth ) / N, где N - размер набора данных
        TODO: реализуйте подсчет accuracy
    """
    # raise NotImplementedError
    predicted_class = torch.argmax(predicted_class, dim=1)
    _accuracy = torch.sum(predicted_class == ground_truth) / ground_truth.size(0)
    return _accuracy
