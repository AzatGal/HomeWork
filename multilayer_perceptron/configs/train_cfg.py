import os
from easydict import EasyDict
from multilayer_perceptron.configs.kmnist_cfg import cfg as dataset_cfg
from multilayer_perceptron.configs.mlp_cfg import cfg as model_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()

cfg.batch_size = 90  # 64
cfg.lr = 1e-1  # 1e-3
cfg.optimizer_name = 'SGD'  # ['SGD', 'Adam']
cfg.num_epochs = 10

cfg.model_cfg = model_cfg
cfg.dataset_cfg = dataset_cfg

cfg.exp_dir = os.path.join(ROOT_DIR, 'mlp_model')
