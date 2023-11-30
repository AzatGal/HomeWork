import torch
import torch.nn as nn
from models.blocks.resnet_blocks import InputStem, StageD


class ResNetD(nn.Module):
    def __init__(self, cfg, nrof_classes):
        """ https://arxiv.org/pdf/1512.03385.pdf """
        super(ResNetD, self).__init__()

        self.cfg = cfg
        self.nrof_classes = nrof_classes

        # TODO: инициализируйте слои модели, используя классы InputStem, Stage
        self.input_block = InputStem()
        self.stages = nn.Sequential(
            StageD(nrof_blocks=3, in_channels=64, out_channels=64, stride=1),
            StageD(nrof_blocks=4, in_channels=128, out_channels=128, stride=2),
            StageD(nrof_blocks=6, in_channels=256, out_channels=256, stride=2),
            StageD(nrof_blocks=3, in_channels=512, out_channels=512, stride=2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # TODO: инициализируйте выходной слой модели
        self.linear = nn.Linear(2048, nrof_classes)

        self.apply(self._init_weights)
        # raise NotImplementedError

    def _init_weights(self, m):
        """
            Cверточные и полносвязные веса инициализируются согласно xavier_uniform
            Все bias инициализируются 0
            В слое batch normalization вектор gamma инициализируется 1, вектор beta – 0 (в базовой модели)

            # TODO: реализуйте этот метод
        """
        # raise NotImplementedError
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def weight_decay_params(self):
        """
            Сбор параметров сети, для которых применяется (веса сверточных и полносвязных слоев)
            и не применяется L2-регуляризация (все остальные параметры, включая bias conv и linear)
            :return: wo_decay, w_decay

            # TODO: реализуйте этот метод
        """
        wo_decay, w_decay = [], []
        for name, param in self.named_parameters():
            if 'weight' in name and ('conv' in name or 'linear' in name):
                w_decay.append(param)
            else:
                wo_decay.append(param)
        return wo_decay, w_decay
        # raise NotImplementedError

    def forward(self, inputs):
        """
           Forward pass нейронной сети, все вычисления производятся для батча
           :param inputs: torch.Tensor(batch_size, channels, height, weight), channels = 3, height = weight = 224
           :return output of the model: torch.Tensor(batch_size, nrof_classes)

           TODO: реализуйте forward pass
       """
        x = self.input_block(inputs)
        x = self.stages(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
        # raise NotImplementedError
