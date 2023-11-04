from easydict import EasyDict

from logistic_regression.utils.enums import DataProcessTypes, WeightsInitType, GDStoppingCriteria

cfg = EasyDict()

# data
cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1
cfg.data_preprocess_type = DataProcessTypes.standardization

# training
cfg.weights_init_type = WeightsInitType.xavier_normal  # normal
cfg.weights_init_kwargs = {'sigma': 1}

cfg.env_path = '/Users/azatgalautdinov/Desktop/mllib_f2023_logreg/api_token.env'  # Путь до файла .env где будет храниться api_token.
cfg.project_name = 'azat.galyautdinov161002/ML-Homework1'

cfg.gamma = 0.01
cfg.gd_stopping_criteria = GDStoppingCriteria.epoch
cfg.nb_epoch = 100
