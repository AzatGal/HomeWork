import torch
import torch.nn as nn

from models.blocks.vgg16_blocks import conv_block, classifier_block


class VGG16(nn.Module):
    def __init__(self, cfg, nrof_classes):
        """https://arxiv.org/pdf/1409.1556.pdf"""
        super(VGG16, self).__init__()

        self.cfg = cfg
        self.nrof_classes = nrof_classes

        # TODO: инициализируйте сверточные слои модели, используя функцию conv_block
        """
        _conv_layers = []

        size = len(self.cfg.conv_blocks)
        for i in range(size):
            _conv_layers.append(
                conv_block(self.cfg.conv_blocks['in_channels'][i],
                           self.cfg.conv_blocks['out_channels'][i])
            )

        self.conv_layers = nn.Sequential(*_conv_layers)
        """

        self.conv1 = conv_block([3, 64], [64, 64])
        self.conv2 = conv_block([64, 128], [128, 128])
        self.conv3 = conv_block([128, 256, 256], [256, 256, 256])
        self.conv4 = conv_block([256, 512, 512], [512, 512, 512])
        self.conv5 = conv_block([512, 512, 512], [512, 512, 512])


        # TODO: инициализируйте полносвязные слои модели, используя функцию classifier_block
        #  (последний слой инициализируется отдельно)
        """
        self.linears = classifier_block(self.cfg.full_conn_blocks["in_features"],
                                        self.cfg.full_conn_blocks["out_features"])
        """
        self.linears = classifier_block([512 * 7 * 7, 4096], [4096, 4096])

        # TODO: инициализируйте последний полносвязный слой для классификации с помощью
        #  nn.Linear(in_features=4096, out_features=nrof_classes)
        self.classifier = nn.Linear(in_features=4096, out_features=nrof_classes)

        # raise NotImplementedError

    def forward(self, inputs):
        """
           Forward pass нейронной сети, все вычисления производятся для батча
           :param inputs: torch.Tensor(batch_size, channels, height, weight)
           :return output of the model: torch.Tensor(batch_size, nrof_classes)

           TODO: реализуйте forward pass
        """
        # raise NotImplementedError

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.linears(x)
        outputs = self.classifier(x)
        """
        print("inputs size:", inputs.size())
        x = self.conv_layers(inputs)
        # x = x.view(x.size(0), -1)
        # x = nn.AdaptiveAvgPool2d((7, 7))(x)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        # x = x.reshape(x.size(0), -1)
        x = self.linears(x)
        outputs = self.classifier(x)
        """
        return outputs
