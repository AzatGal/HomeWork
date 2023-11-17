import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()

        self.cfg = cfg

        self.layers = self._init_layers()
        self.apply(self._init_weights)

    def _init_layers(self):
        """
            Инициализация слоев нейронной сети.

            Описание необходимых слоев задается в self.cfg.layers в следующем виде:
                 [
                    (name_1, params_1),
                    ...,
                    (name_N, params_N)
                ],
                где name_i (str) - название класса из nn (Linear, ReLU и т.д.), params_i (dict) - параметры этого слоя
            :return: список инициализированных слоев

            TODO: необходимо инициализировать слои, заданные с помощью self.cfg.layers, данная функция должна быть
                универсальной (описание модели вынесено в конфиг, основной код не должен меняться в зависимости от
                эксперимента), можно использовать getattr(nn, name) и nn.Sequential/nn.ModuleList
        """
        # raise NotImplementedError
        layers = []
        for name, params in self.cfg.layers:
            layer = getattr(nn, name)(**params)
            layers.append(layer)
        return nn.Sequential(*layers)

    @torch.no_grad()
    def _init_weights(self, m):
        """
            Инициализация параметров линейный слоев согласно заданному типу self.cfg.init_type.
            # TODO: реализуйте этот метод, можно использовать getattr(nn.init, self.cfg.init_type)
        """
        # raise NotImplementedError
        init_func = getattr(nn.init, self.cfg.init_type)
        if isinstance(m, nn.Linear):
            init_func(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, inputs):
        """
            Forward pass нейронной сети, все вычисления производятся для батча
            :param inputs: torch.Tensor(batch_size, height, weight)
            :return output of the model: torch.Tensor(batch_size, nrof_classes)

            TODO: реализуйте этот метод
        """
        # raise NotImplementedError
        x = inputs.view(inputs.size(0), -1)
        return self.layers(x)


if __name__ == '__main__':
    from Perceptron.configs.mlp_cfg import cfg

    model = MLP(cfg)

    # TODO: вывести количество обучаемых параметров нейронной сети
    # nrof_params = None

    nrof_params = 0
    for name, params in model.cfg.layers:
        for item in params:
            nrof_params += params[item]
    print(f'number of trainable parameters: {nrof_params}')
