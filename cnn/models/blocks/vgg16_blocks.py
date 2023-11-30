import torch
import torch.nn as nn


def conv_block(in_channels: [], out_channels: [], conv_params=None, maxpool_params=None):
    """
        Функция построения одного сверточного блока нейронной сети VGG-16. Списки in_channels и out_channels задают
        последовательность сверточных слоев с соответствующими параметрами фильтров. После каждого сверточного слоя
        используется nn.BatchNorm2d и функция активации nn.RelU(inplace=True). В конце сверточных слоев необходимо применить Max Pooling

        :param in_channels: List - глубина фильтров в каждом слое
        :param out_channels: List - количество сверточных фильтров в каждом слое
        :param conv_params: None or dict - дополнительные параметры сверточных слоев
        :param maxpool_params: None or dict - параметры max pooling слоя
        :return: nn.Sequential - последовательность слоев

        # TODO: реализуйте данную функцию
    """

    assert len(in_channels) == len(out_channels)

    if conv_params is None:
        conv_params = dict(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    if maxpool_params is None:
        maxpool_params = dict(kernel_size=2, stride=2, padding=0)

    # raise NotImplementedError
    layers = []

    for i in range(len(in_channels)):
        layers.append(nn.Conv2d(in_channels[i], out_channels[i], **conv_params))
        layers.append(nn.BatchNorm2d(out_channels[i]))
        layers.append(nn.ReLU(inplace=True))

    layers.append(nn.MaxPool2d(**maxpool_params))

    return nn.Sequential(*layers)


def classifier_block(in_features: [], out_features: [], linear_params=None):
    """
        Функция построения блока полносвязных слоев нейронной сети VGG-16 (последний слой инициализируется отдельно).
        Списки in_features и out_features задают последовательность полносвязных слоев с соответствующими параметрами.
        После каждого полносвязного слоя используется функция активации nn.RelU(inplace=True) и
        nn.Dropout(p=0.5, inplace=False).

        :param in_features: List
        :param out_features: List
        :param linear_params: None or dict - дополнительные параметры nn.Linear
        :return: nn.Sequential - последовательность слоев

        # TODO: реализуйте данную функцию
    """

    assert len(in_features) == len(out_features)

    if linear_params is None:
        linear_params = dict(bias=True)

    # raise NotImplementedError
    layers = []

    for i in range(len(in_features)):
        layers.append(nn.Linear(in_features[i], out_features[i], **linear_params))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(p=0.5, inplace=False))

    return nn.Sequential(*layers)
