# TODO: Реализуйте класс для обучения моделей, минимальный набор функций:
#  1. Подготовка обучающих и тестовых данных
#  2. Подготовка модели, оптимайзера, целевой функции
#  3. Обучение модели на обучающих данных
#  4. Эвалюэйшен модели на тестовых данных, для оценки точности можно рассмотреть accuracy, balanced accuracy
#  5. Сохранение и загрузка весов модели
#  6. Добавить возможность обучать на gpu
#  За основу данного класса можно взять https://github.com/pkhanzhina/mllib_f2023_mlp/blob/master/executors/mlp_trainer.py

import gc
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.oxford_pet_dataset import OxfordIIITPet

from datasets.utils.prepare_transforms import prepare_transforms
from logs.Logger import Logger
from models.vgg16 import VGG16
from models.resnet50 import ResNet50
from models.resnetB import ResNetB
from models.resnetC import ResNetC
from models.resnetD import ResNetD
from utils.metrics import accuracy, balanced_accuracy
from utils.visualization import show_batch
from utils.utils import set_seed
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, outputs, targets):
        num_classes = outputs.size(1)
        smoothed_targets = torch.full_like(outputs, self.smoothing / (num_classes - 1))
        smoothed_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        loss = nn.KLDivLoss()(torch.log_softmax(outputs, dim=1), smoothed_targets)
        return loss


class Trainer:
    def __init__(self, cfg):
        set_seed(cfg.seed)

        self.cfg = cfg
        if self.cfg.model_name in ("ResNetB", "ResNetC", "ResNetD"):
            self.cfg.lr = 0.1 * (self.cfg.batch_size / 256)

        self.cfg.device = torch.device("cuda" if (self.cfg.device == "cuda" and
                                                  torch.cuda.is_available()) else "cpu")

        self.__prepare_data(self.cfg.dataset_cfg)
        self.__prepare_model(self.cfg.model_cfg)

        self.logger = Logger(env_path="/Users/azatgalautdinov/Desktop/ML_Homework2/Perceptron/api_token.env",
                             project='azat.galyautdinov161002/ML-Homework1', experiment_name=self.cfg.experiment_name)
        _params = {
            "batch_size": self.cfg.batch_size,
            "learning rate": self.cfg.lr,
            "optimizer name": self.cfg.optimizer_name,
            "number epoch": self.cfg.num_epochs,
            "model name": self.cfg.model_name
        }
        if self.cfg.model_name == "VGG16":
            _params['momentum'] = self.cfg.momentum

        self.logger.log_hyperparameters(params=_params)

    def __prepare_data(self, dataset_cfg):
        self.train_dataset = OxfordIIITPet(dataset_cfg, "train",
                                           transform=prepare_transforms(dataset_cfg.transforms['train']))
        self.test_dataset = OxfordIIITPet(dataset_cfg, 'test',
                                          transform=prepare_transforms(dataset_cfg.transforms['test']))

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False)

    def __prepare_model(self, model_cfg):
        if self.cfg.model_name == "VGG16":
            self.model = VGG16(model_cfg, self.cfg.dataset_cfg.nrof_classes)
        if self.cfg.model_name == "ResNet50":
            self.model = ResNet50(model_cfg, self.cfg.dataset_cfg.nrof_classes)
        if self.cfg.model_name == "ResNetB":
            self.model = ResNetB(model_cfg, self.cfg.dataset_cfg.nrof_classes)
        if self.cfg.model_name == "ResNetC":
            self.model = ResNetC(model_cfg, self.cfg.dataset_cfg.nrof_classes)
        if self.cfg.model_name == "ResNetD":
            self.model = ResNetD(model_cfg, self.cfg.dataset_cfg.nrof_classes)

        self.model.to(self.cfg.device)
        # Определение функции потерь и оптимизатора
        if self.cfg.model_name in ("ResNetB", "ResNetC", "ResNetD"):
            self.criterion = LabelSmoothingLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        optim_class = getattr(torch.optim, self.cfg.optimizer_name)

        if self.cfg.model_name == "VGG16":
            if self.cfg.optimizer_name == "SGD":
                self.optimizer = optim_class(
                    self.model.parameters(),
                    lr=self.cfg.lr,
                    momentum=self.cfg.momentum,
                    weight_decay=self.cfg.weight_decay
                )
            else:
                self.optimizer = optim_class(self.model.parameters(), lr=self.cfg.lr)
        else:
            self.optimizer = optim_class(
                [
                    {'params': self.model.weight_decay_params()[0], 'weight_decay': 0},
                    {'params': self.model.weight_decay_params()[1]}
                ],
                lr=self.cfg.lr
            )

        if self.cfg.model_name in ("ResNetB", "ResNetC", "ResNetD"):
            self.scheduler_wu = LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: (epoch + 1) * self.cfg.lr / 10 if epoch < 5 else self.cfg.lr
            )
            self.scheduler_cd = CosineAnnealingLR(self.optimizer, T_max=self.cfg.num_epochs)

    def save_model(self, filename):
        save_path = os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, filename):
        load_path = os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        self.model.load_state_dict(torch.load(load_path))

    def make_step(self, batch, update_model=True):
        images = batch['image'].to(self.cfg.device)
        labels = batch['label'].to(self.cfg.device)

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
        self.model.train()
        for batch_idx, batch in enumerate(self.train_dataloader):

            loss, outputs = self.make_step(batch, update_model=True)

            outputs = torch.argmax(outputs, dim=1).to(self.cfg.device)
            batch['label'] = batch['label'].to(self.cfg.device)

            acc = accuracy(outputs, batch['label'])

            op = [0] * self.cfg.dataset_cfg.nrof_classes
            p = [0] * self.cfg.dataset_cfg.nrof_classes
            for i in range(len(outputs)):
                p[batch["label"][i]] += 1
                if batch["label"][i] == outputs[i]:
                    op[outputs[i]] += 1
            b_acc = balanced_accuracy(op, p)

            self.logger.save_param('train', 'loss', loss)  # ('Train Loss', loss, step=batch_idx)
            self.logger.save_param('train', 'accuracy', acc)
            self.logger.save_param('train', 'balanced accuracy', b_acc)
            self.logger.save_param('train', 'learning rate', self.optimizer.param_groups[0]['lr'])

        if self.cfg.model_name in ("ResNetB", "ResNetC", "ResNetD"):
            self.scheduler_wu.step()
            self.scheduler_cd.step()

    def evaluate(self, *args, **kwargs):
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0
        total_b_acc = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_dataloader):
                loss, outputs = self.make_step(batch, update_model=False)
                show_batch(batch["image"])
                outputs = torch.argmax(outputs, dim=1).to(self.cfg.device)
                batch['label'] = batch['label'].to(self.cfg.device)

                total_loss += loss * len(batch['image'])
                total_correct += accuracy(outputs, batch["label"]) * len(batch['image'])
                total_samples += len(batch['image'])

                op = [0] * self.cfg.dataset_cfg.nrof_classes
                p = [0] * self.cfg.dataset_cfg.nrof_classes
                for i in range(len(outputs)):
                    p[batch["label"][i]] += 1
                    if batch["label"][i] == outputs[i]:
                        op[outputs[i]] += 1
                total_b_acc += balanced_accuracy(op, p) * len(batch['image'])

                del batch
                del outputs
                gc.collect()  # import gc
                torch.cuda.empty_cache()

        avg_ba = total_b_acc / total_samples
        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples
        self.logger.save_param('test', 'loss', avg_loss)
        self.logger.save_param('test', 'accuracy', avg_accuracy)
        self.logger.save_param('test', 'balanced accuracy', avg_ba)

    def fit(self, *args, **kwargs):
        for epoch in range(self.cfg.num_epochs):  # self.cfg.num_epochs):
            print(f"Epoch {epoch + 1}/{self.cfg.num_epochs}")  # self.cfg.num_epochs}")
            self.train_epoch()
            self.evaluate()
        self.evaluate()


if __name__ == "__main__":
    from configs.train_cfg import cfg

    trainer = Trainer(cfg)

    trainer.fit()

    trainer.evaluate()
