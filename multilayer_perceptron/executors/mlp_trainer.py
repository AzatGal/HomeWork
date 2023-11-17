import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from multilayer_perceptron.datasets.KMNIST import KMNIST
from multilayer_perceptron.models.MLP import MLP
from multilayer_perceptron.datasets.utils.prepare_transforms import prepare_transforms
from multilayer_perceptron.utils.metrics import accuracy
from multilayer_perceptron.utils.visualization import confusion_matrix
from multilayer_perceptron.logs.Logger import Logger


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        # TODO: настройте логирование с помощью класса Logger
        #  (пример: https://github.com/KamilyaKharisova/mllib_f2023/blob/master/logginig_example.py)

        # TODO: залогируйте используемые гиперпараметры в neptune.ai через метод log_hyperparameters

        self.logger = Logger(env_path="/Users/azatgalautdinov/Desktop/ML_Homework2/Perceptron/api_token.env",
                             project='azat.galyautdinov161002/ML-Homework1')
        _params = {
            "batch_size": self.cfg.batch_size,
            "learning rate": self.cfg.lr,
            "optimizer name": self.cfg.optimizer_name,
            "number epoch": self.cfg.num_epochs
        }
        for i in range(len(self.cfg.model_cfg.layers)):
            _params[self.cfg.model_cfg.layers[i][0]+f' {i + 1}'] = self.cfg.model_cfg.layers[i][1]
            if self.cfg.model_cfg.layers[i][1] == {}:
                _params[f'activation function {i + 1}'] = self.cfg.model_cfg.layers[i][0]

        self.logger.log_hyperparameters(params=_params)
        '''
            params={
                "batch_size": self.cfg.batch_size,
                "learning rate": self.cfg.lr,
                "optimizer name": self.cfg.optimizer_name,
                "number epoch": self.cfg.num_epochs,
                self.cfg.model_cfg.layers[0][0] + '1': self.cfg.model_cfg.layers[0][1],
                self.cfg.model_cfg.layers[1][0]: 0,
                self.cfg.model_cfg.layers[2][0] + '2': self.cfg.model_cfg.layers[2][1]
                # "model cfg": self.cfg.model_cfg,
            })
        '''
        self.__prepare_data(self.cfg.dataset_cfg)
        self.__prepare_model(self.cfg.model_cfg)

    def __prepare_data(self, dataset_cfg):
        """ Подготовка обучающих и тестовых данных """
        self.train_dataset = KMNIST(dataset_cfg, 'train',
                                    transforms=prepare_transforms(dataset_cfg.transforms['train']))
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True)

        self.test_dataset = KMNIST(dataset_cfg, 'test', transforms=prepare_transforms(dataset_cfg.transforms['test']))
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False)

    def __prepare_model(self, model_cfg):
        """ Подготовка нейронной сети"""
        self.model = MLP(model_cfg)
        self.criterion = nn.CrossEntropyLoss()

        # TODO: инициализируйте оптимайзер через getattr(torch.optim, self.cfg.optimizer_name)
        # self.optimizer = None

        optimizer_class = getattr(torch.optim, self.cfg.optimizer_name)
        self.optimizer = optimizer_class(self.model.parameters(), lr=self.cfg.lr)

    def save_model(self, filename):
        """
            Сохранение весов модели с помощью torch.save()
            :param filename: str - название файла
            TODO: реализовать сохранение модели по пути os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        """
        # raise NotImplementedError

        save_path = os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, filename):
        """
            Загрузка весов модели с помощью torch.load()
            :param filename: str - название файла
            TODO: реализовать выгрузку весов модели по пути os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        """
        # raise NotImplementedError

        load_path = os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        self.model.load_state_dict(torch.load(load_path))

    def make_step(self, batch, update_model=True):
        """
            Этот метод выполняет один шаг обучения, включая forward pass, вычисление целевой функции,
            backward pass и обновление весов модели (если update_model=True).

            :param batch: dict of data with keys ["image", "label"]
            :param update_model: bool - если True, необходимо сделать backward pass и обновить веса модели
            :return: значение функции потерь, выход модели
            # TODO: реализуйте инференс модели для данных batch, посчитайте значение целевой функции
        """
        # raise NotImplementedError
        images = batch['image']
        labels = batch['label']

        # Forward pass
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        if update_model:
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item(), outputs

    def train_epoch(self, *args, **kwargs):
        """
            Обучение модели на self.train_dataloader в течение одной эпохи. Метод проходит через все обучающие данные и
            вызывает метод self.make_step() на каждом шаге.

            TODO: реализуйте функцию обучения с использованием метода self.make_step(batch, update_model=True),
                залогируйте на каждом шаге значение целевой функции и accuracy на batch
        """
        # raise NotImplementedError
        self.model.train()

        for batch_idx, batch in enumerate(self.train_dataloader):
            loss, outputs = self.make_step(batch, update_model=True)
            acc = accuracy(outputs, batch["label"])
            # Log loss and accuracy
            self.logger.save_param('train', 'loss', loss)   # ('Train Loss', loss, step=batch_idx)
            self.logger.save_param('train', 'accuracy', acc)

        # print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}")

    def evaluate(self, *args, **kwargs):
        """
            Метод используется для проверки производительности модели на обучающих/тестовых данных. Сначала модель
            переводится в режим оценки (model.eval()), затем данные последовательно подаются на вход модели, по
            полученным выходам вычисляются метрики производительности, такие как значение целевой функции, accuracy

            TODO: реализуйте функцию оценки с использованием метода self.make_step(batch, update_model=False),
                залогируйте значения целевой функции и accuracy, постройте confusion_matrix
        """
        # raise NotImplementedError

        self.model.eval()

        conf_matrix = torch.zeros(self.cfg.dataset_cfg.nrof_classes, self.cfg.dataset_cfg.nrof_classes)

        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_dataloader):
                loss, outputs = self.make_step(batch, update_model=False)
                for i in range(len(batch["label"])):
                    conf_matrix[batch["label"][i], torch.argmax(outputs, dim=1)[i]] += 1
                total_loss += loss * len(batch['image'])
                total_correct += accuracy(outputs, batch["label"]) * len(batch['image'])
                total_samples += len(batch['image'])
            # Log loss and accuracy

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples
        self.logger.save_param('test', 'loss', avg_loss)
        self.logger.save_param('test', 'accuracy', avg_accuracy)
        self.logger.save_param('test', 'confusion matrix', confusion_matrix(conf_matrix))
        # Confusion matrix
        #f = confusion_matrix(conf_matrix)

    def fit(self, *args, **kwargs):
        """
            Основной цикл обучения модели. Данная функция должна содержать один цикл на заданное количество эпох.
            На каждой эпохе сначала происходит обучение модели на обучающих данных с помощью метода self.train_epoch(),
            а затем оценка производительности модели на тестовых данных с помощью метода self.evaluate()

            # TODO: реализуйте основной цикл обучения модели, сохраните веса модели с лучшим значением accuracy на
                тестовой выборке
        """
        # raise NotImplementedError

        best_accuracy = 0.0

        for epoch in range(self.cfg.num_epochs):  # self.cfg.num_epochs):
            print(f"Epoch {epoch + 1}/{self.cfg.num_epochs}")  # self.cfg.num_epochs}")
            self.train_epoch()
            self.evaluate()
        # Save the model if it has the best accuracy so far
        self.evaluate()

    def overfitting_on_batch(self, max_step=100):
        """
            Оверфиттинг на одном батче. Эта функция может быть полезна для отладки и оценки способности вашей
            модели обучаться и обновлять свои веса в ответ на полученные данные.
        """
        batch = next(iter(self.train_dataloader))
        for step in range(max_step):
            loss, output = self.make_step(batch, update_model=True)
            if step % 10 == 0:
                acc = accuracy(output, batch['label'])
                print('[{:d}]: loss - {:.4f}, {:.4f}'.format(step, loss, acc))


if __name__ == '__main__':
    from multilayer_perceptron.configs.train_cfg import cfg

    trainer = Trainer(cfg)

    # оверффитинг на одном батче
    trainer.overfitting_on_batch()

    # обучение нейронной сети
    trainer.fit()

    # оценка сети на обучающей/валидационной/тестовой выборке
    trainer.evaluate()


    """
    cfg.dataset_cfg.transforms.train = [
        ('ToTensor', ()),
        ('Normalize', ([0.5], [1])),
        ('Pad', (5, 5)),
        ('RandomCrop', (28, 28))
    ]

    trainer = Trainer(cfg)
    
    # оверффитинг на одном батче
    trainer.overfitting_on_batch()

    # обучение нейронной сети
    trainer.fit()

    # оценка сети на обучающей/валидационной/тестовой выборке
    trainer.evaluate()
    """

    """ от аугментации результат не поменялся """

    """
    cfg.model_cfg.layers = [
        ('Linear', {'in_features': 28 * 28, 'out_features': 200}),
        ('Tanh', {}),
        ('Linear', {'in_features': 200, 'out_features': 10}),
    ]
    cfg.dataset_cfg.transforms.train = [
        ('ToTensor', ()),
        ('Normalize', ([0.5], [1]))
    ]
    trainer = Trainer(cfg)

    # оверффитинг на одном батче
    trainer.overfitting_on_batch()

    # обучение нейронной сети
    trainer.fit()

    # оценка сети на обучающей/валидационной/тестовой выборке
    trainer.evaluate()
    """

    """ tanh хуже relu"""

    """
    cfg.model_cfg.layers = [
        ('Linear', {'in_features': 28 * 28, 'out_features': 300}),
        ('ReLU', {}),
        ('Linear', {'in_features': 300, 'out_features': 10}),
    ]
    cfg.dataset_cfg.transforms.train = [
        ('ToTensor', ()),
        ('Normalize', ([0.5], [1]))
    ]
    trainer = Trainer(cfg)

    # оверффитинг на одном батче
    trainer.overfitting_on_batch()

    # обучение нейронной сети
    trainer.fit()

    # оценка сети на обучающей/валидационной/тестовой выборке
    trainer.evaluate()
    """

    """ небольшое улучшение """

    """
    cfg.model_cfg.layers = [
        ('Linear', {'in_features': 28 * 28, 'out_features': 200}),
        ('ReLU', {}),
        ('Linear', {'in_features': 200, 'out_features': 200}),
        ('ReLU', {}),
        ('Linear', {'in_features': 200, 'out_features': 10}),
    ]
    cfg.dataset_cfg.transforms.train = [
        ('ToTensor', ()),
        ('Normalize', ([0.5], [1]))
    ]
    trainer = Trainer(cfg)

    # оверффитинг на одном батче
    trainer.overfitting_on_batch()

    # обучение нейронной сети
    trainer.fit()

    # оценка сети на обучающей/валидационной/тестовой выборке
    trainer.evaluate()
    """

    """ Нет улучшений """

    """
    cfg.model_cfg.layers = [
        ('Linear', {'in_features': 28 * 28, 'out_features': 200}),
        ('ReLU', {}),
        ('Linear', {'in_features': 200, 'out_features': 10}),
    ]

    cfg.optimizer_name = 'Adam'

    trainer = Trainer(cfg)

    # оверффитинг на одном батче
    trainer.overfitting_on_batch()

    # обучение нейронной сети
    trainer.fit()

    # оценка сети на обучающей/валидационной/тестовой выборке
    trainer.evaluate()
    """

    """ сильно ухудшились результаты, на последних эпохах сильный перекос в предсказание 1 класса """

    """
    cfg.model_cfg.layers = [
        ('Linear', {'in_features': 28 * 28, 'out_features': 300}),
        ('ReLU', {}),
        ('Linear', {'in_features': 300, 'out_features': 10}),
    ]

    cfg.optimizer_name = 'Adam'

    trainer = Trainer(cfg)

    # оверффитинг на одном батче
    trainer.overfitting_on_batch()

    # обучение нейронной сети
    trainer.fit()

    # оценка сети на обучающей/валидационной/тестовой выборке
    trainer.evaluate()
    """
    """ попрежнему низкие результаты, сильный перекос в предсказание 3 класса на послежних эпохах"""
