import os
from easydict import EasyDict
from configs.oxford_pet_cfg import cfg as dataset_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()
cfg.seed = 0

cfg.batch_size = 64
cfg.lr = 1e-3

cfg.model_name = 'VGG16'  # ['VGG16', 'ResNet50']
cfg.optimizer_name = 'Adam'  # ['SGD', 'Adam']

cfg.device = 'cpu'  # ['cpu', 'cuda']

cfg.model_cfg = ...
cfg.dataset_cfg = dataset_cfg

cfg.exp_dir = os.path.join(ROOT_DIR, 'train_vgg16')

cfg.num_epoch = 5













model_cfg = EasyDict()
ROOT_DIR = '/kaggle/working/'

train_cfg = EasyDict()
train_cfg.seed = 0

train_cfg.experiment_name = ""

train_cfg.batch_size = 64
train_cfg.lr = 1e-3

train_cfg.model_name = 'VGG16'  # ['VGG16', 'ResNet50']
train_cfg.optimizer_name = 'SGD'  # ['SGD', 'Adam']
train_cfg.momentum = 0.9
train_cfg.num_epochs = 15
train_cfg.weight_decay=5e-4

train_cfg.device = 'cuda'  # ['cpu', 'cuda']
train_cfg.exp_dir = os.path.join(ROOT_DIR, 'exp_name')
train_cfg.dataset_cfg = dataset_cfg
train_cfg.model_cfg = model_cfg