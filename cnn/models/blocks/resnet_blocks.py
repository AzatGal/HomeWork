import torch
import torch.nn as nn


class InputStem(nn.Module):
    def __init__(self):
        """
            Входной блок нейронной сети ResNet, содержит свертку 7x7 c количеством фильтров 64 и шагом 2, затем
            следует max-pooling 3x3 с шагом 2.
            
            TODO: инициализируйте слои входного блока
        """
        super().__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # raise NotImplementedError

    def forward(self, inputs):
        # TODO: реализуйте forward pass
        # raise NotImplementedError
        x = self.conv(inputs)
        x = self.max_pool(x)
        return x


class InputStemC(nn.Module):
    def __init__(self):
        """
            Входной блок нейронной сети ResNet, содержит свертку 7x7 c количеством фильтров 64 и шагом 2, затем
            следует max-pooling 3x3 с шагом 2.

            TODO: инициализируйте слои входного блока
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # raise NotImplementedError

    def forward(self, inputs):
        # TODO: реализуйте forward pass
        # raise NotImplementedError
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, stride=1, down_sampling=False):
        """
            Остаточный блок, состоящий из 3 сверточных слоев (path A) и shortcut connection (path B).
            Может быть двух видов:
                1. Down sampling (только первый Bottleneck блок в Stage)
                2. Residual (последующие Bottleneck блоки в Stage)

            Path A:
                Cостоит из 3-x сверточных слоев (1x1, 3x3, 1x1), после каждого слоя применяется BatchNorm,
                после первого и второго слоев - ReLU. Количество фильтров для первого слоя - out_channels,
                для второго слоя - out_channels, для третьего слоя - out_channels * expansion.

            Path B:
                1. Down sampling: path B = Conv (1x1, stride) и  BatchNorm
                2. Residual: path B = nn.Identity

            Выход Bottleneck блока - path_A(inputs) + path_B(inputs)

            :param in_channels: int - количество фильтров во входном тензоре
            :param out_channels: int - количество фильтров в промежуточных слоях
            :param expansion: int = 4 - множитель на количество фильтров в выходном слое
            :param stride: int
            :param down_sampling: bool
            TODO: инициализируйте слои Bottleneck
        """
        super().__init__()
        # raise NotImplementedError

        self.expansion = expansion
        self.down_sampling = down_sampling

        if self.down_sampling:
            in_channels *= stride
            self.path_B = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * expansion, kernel_size=(1, 1), stride=(stride, stride)),  # , padding=1),
                nn.BatchNorm2d(out_channels * expansion)
            )
        else:
            in_channels *= expansion
            self.path_B = nn.Identity()
        """
        self.path_B = []
        if self.down_sampling:
            self.path_B.append(nn.Conv2d(in_channels, out_channels * expansion, kernel_size=(1, 1),
                                         stride=(stride, stride), padding=1))
            self.path_B.append(nn.BatchNorm2d(out_channels * expansion))
        else:
            self.path_B.append(nn.Identity())
        """

        self.path_A = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, stride)),  # , padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3)),  # , padding=1),  # , stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * expansion, kernel_size=(1, 1), padding=1),
            nn.BatchNorm2d(out_channels * expansion)
        )
        """
        self.path_A = []
        self.path_A.append(nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, stride), padding=1))
        self.path_A.append(nn.BatchNorm2d(out_channels))
        self.path_A.append(nn.ReLU(inplace=True))
        self.path_A.append(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1))  # , stride=stride, padding=1),
        self.path_A.append(nn.BatchNorm2d(out_channels))
        self.path_A.append(nn.ReLU(inplace=True))
        self.path_A.append(nn.Conv2d(out_channels, out_channels * expansion, kernel_size=(1, 1)))
        self.path_A.append(nn.BatchNorm2d(out_channels * expansion))
        """

    def forward(self, inputs):
        # TODO: реализуйте forward pass
        # raise NotImplementedError
        # print("inputs botl", inputs.size())
        x = self.path_A(inputs)
        # print("path A x size", x.size())
        y = self.path_B(inputs)
        # print("path B x size", y.size())
        x += y
        x = nn.ReLU(inplace=True)(x)
        """
        x = self.path_A[0](inputs)
        y = self.path_B[0](inputs)
        for i in range(1, len(self.path_A)):
            x = self.path_A[i](x)
        for i in range(1, len(self.path_B)):
            y = self.path_B[i](y)
        x = nn.ReLU(inplace=True)(x + y)
        """
        return x


class BottleneckD(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, stride=1, down_sampling=False):
        """
            Остаточный блок, состоящий из 3 сверточных слоев (path A) и shortcut connection (path B).
            Может быть двух видов:
                1. Down sampling (только первый Bottleneck блок в Stage)
                2. Residual (последующие Bottleneck блоки в Stage)

            Path A:
                Cостоит из 3-x сверточных слоев (1x1, 3x3, 1x1), после каждого слоя применяется BatchNorm,
                после первого и второго слоев - ReLU. Количество фильтров для первого слоя - out_channels,
                для второго слоя - out_channels, для третьего слоя - out_channels * expansion.

            Path B:
                1. Down sampling: path B = Conv (1x1, stride) и  BatchNorm
                2. Residual: path B = nn.Identity

            Выход Bottleneck блока - path_A(inputs) + path_B(inputs)

            :param in_channels: int - количество фильтров во входном тензоре
            :param out_channels: int - количество фильтров в промежуточных слоях
            :param expansion: int = 4 - множитель на количество фильтров в выходном слое
            :param stride: int
            :param down_sampling: bool
            TODO: инициализируйте слои Bottleneck
        """
        super().__init__()
        # raise NotImplementedError

        self.expansion = expansion
        self.down_sampling = down_sampling

        if self.down_sampling:
            in_channels *= stride
            self.path_B = nn.Sequential(
                nn.AvgPool2d(kernel_size=(1, 1), stride=stride),  # kernel_size=(2, 2)
                nn.Conv2d(in_channels, out_channels * expansion, kernel_size=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(out_channels * expansion)
            )
        else:
            in_channels *= expansion
            self.path_B = nn.Identity()

        self.path_A = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, stride)),  # , padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3)),  # , padding=1),  # , stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * expansion, kernel_size=(1, 1), padding=1),
            nn.BatchNorm2d(out_channels * expansion)
        )

    def forward(self, inputs):
        # TODO: реализуйте forward pass
        # raise NotImplementedError
        # print("inputs size", inputs.size())
        x = self.path_A(inputs)
        # print("x size", x.size())
        y = self.path_B(inputs)
        # print("y size", y.size())
        x += y
        x = nn.ReLU(inplace=True)(x)
        return x


class BottleneckB(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, stride=1, down_sampling=False):
        """
            Остаточный блок, состоящий из 3 сверточных слоев (path A) и shortcut connection (path B).
            Может быть двух видов:
                1. Down sampling (только первый Bottleneck блок в Stage)
                2. Residual (последующие Bottleneck блоки в Stage)

            Path A:
                Cостоит из 3-x сверточных слоев (1x1, 3x3, 1x1), после каждого слоя применяется BatchNorm,
                после первого и второго слоев - ReLU. Количество фильтров для первого слоя - out_channels,
                для второго слоя - out_channels, для третьего слоя - out_channels * expansion.

            Path B:
                1. Down sampling: path B = Conv (1x1, stride) и  BatchNorm
                2. Residual: path B = nn.Identity

            Выход Bottleneck блока - path_A(inputs) + path_B(inputs)

            :param in_channels: int - количество фильтров во входном тензоре
            :param out_channels: int - количество фильтров в промежуточных слоях
            :param expansion: int = 4 - множитель на количество фильтров в выходном слое
            :param stride: int
            :param down_sampling: bool
            TODO: инициализируйте слои Bottleneck
        """
        super().__init__()
        # raise NotImplementedError

        self.expansion = expansion
        self.down_sampling = down_sampling

        if self.down_sampling:
            in_channels *= stride
            self.path_B = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * expansion, kernel_size=(1, 1), stride=(stride, stride)),
                nn.BatchNorm2d(out_channels * expansion)
            )
        else:
            in_channels *= expansion
            self.path_B = nn.Identity()

        self.path_A = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),  # , padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride), padding=1),  # , padding=1),  # , stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * expansion, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels * expansion)
        )

    def forward(self, inputs):
        # TODO: реализуйте forward pass
        # raise NotImplementedError

        x = self.path_A(inputs)
        # print("A size", x.size())
        y = self.path_B(inputs)
        # print("B size", y.size())
        x += y
        x = nn.ReLU(inplace=True)(x)

        return x


class Stage(nn.Module):
    def __init__(self, nrof_blocks: int, in_channels: int, out_channels: int, stride):
        """
            Последовательность Bottleneck блоков, первый блок Down sampling, остальные - Residual

            :param nrof_blocks: int - количество Bottleneck блоков
            TODO: инициализируйте слои, используя класс Bottleneck
        """
        super().__init__()
        # raise NotImplementedError
        self.blocks = nn.ModuleList([
            Bottleneck(in_channels=in_channels, out_channels=out_channels, stride=stride, down_sampling=True)
        ])

        self.blocks.extend([
            Bottleneck(in_channels=in_channels, out_channels=out_channels, stride=1, down_sampling=False) for _ in
            range(nrof_blocks - 2)
        ])

        self.blocks.append(
            Bottleneck(in_channels=in_channels, out_channels=out_channels, stride=1, down_sampling=False))

    def forward(self, inputs):
        # TODO: реализуйте forward pass
        # raise NotImplementedError
        x = inputs
        for block in self.blocks:
            # print("size x", x.size())
            x = block(x)
        return x


class StageB(nn.Module):
    def __init__(self, nrof_blocks: int, in_channels: int, out_channels: int, stride):
        """
            Последовательность Bottleneck блоков, первый блок Down sampling, остальные - Residual

            :param nrof_blocks: int - количество Bottleneck блоков
            TODO: инициализируйте слои, используя класс Bottleneck
        """
        super().__init__()
        # raise NotImplementedError
        self.blocks = nn.ModuleList([
            BottleneckB(in_channels=in_channels, out_channels=out_channels, stride=stride, down_sampling=True)
        ])

        self.blocks.extend([
            BottleneckB(in_channels=in_channels, out_channels=out_channels, stride=1, down_sampling=False) for _ in
            range(nrof_blocks - 1)
        ])
        """
        self.blocks.append(
            Bottleneck(in_channels=in_channels, out_channels=out_channels, stride=1, down_sampling=False))"""

    def forward(self, inputs):
        # TODO: реализуйте forward pass
        # raise NotImplementedError
        x = inputs
        for block in self.blocks:
            # print("size x", x.size())
            x = block(x)
        return x


class StageD(nn.Module):
    def __init__(self, nrof_blocks: int, in_channels: int, out_channels: int, stride):
        """
            Последовательность Bottleneck блоков, первый блок Down sampling, остальные - Residual

            :param nrof_blocks: int - количество Bottleneck блоков
            TODO: инициализируйте слои, используя класс Bottleneck
        """
        super().__init__()
        # raise NotImplementedError
        self.blocks = nn.ModuleList([
            BottleneckD(in_channels=in_channels, out_channels=out_channels, stride=stride, down_sampling=True)
        ])

        self.blocks.extend([
            BottleneckD(in_channels=in_channels, out_channels=out_channels, stride=1, down_sampling=False) for _ in
            range(nrof_blocks - 1)
        ])

    def forward(self, inputs):
        # TODO: реализуйте forward pass
        # raise NotImplementedError
        x = inputs
        for block in self.blocks:
            # print("size x", x.size())
            x = block(x)
        return x
