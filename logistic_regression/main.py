#TODO
# инициализация класса набора данных, стандартизация данных, разделение на выборки, построение onehot encoding вектора
# инициализация класса логистической регрессии
# обучение модели, логирование в Нептун
# сохранение модели
import numpy as np
from datasets.base_dataset_classes import BaseClassificationDataset
from datasets.digits_dataset import Digits
from config.logistic_regression_config import cfg
from models.logistic_regression_model import LogReg
from utils.enums import DataProcessTypes, WeightsInitType


cfg.data_preprocess_type = DataProcessTypes.standardization
cfg.weights_init_type = WeightsInitType.xavier_normal  # normal
cfg.weights_init_kwargs = {'sigma': 1}
data = Digits(cfg)
model = LogReg(cfg, data.k, data.d, "logistic regression", np.size(data.inputs))
model.weights_init_xavier_normal()
model.train(data.inputs_train, BaseClassificationDataset.onehotencoding(data.targets_train, data.k),
            data.inputs_valid, BaseClassificationDataset.onehotencoding(data.targets_valid, data.k))
