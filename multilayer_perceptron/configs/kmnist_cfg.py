from easydict import EasyDict
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()

cfg.nrof_classes = 10

cfg.path = os.path.join(ROOT_DIR, 'data/kmnist/')

cfg.filename = 'kmnist.pkl'

cfg.base_url = "http://codh.rois.ac.jp/kmnist/dataset/kmnist/"
cfg.raw_filename = [
    ["train_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["train_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]

cfg.transforms = EasyDict()
cfg.transforms.train = [
    ('ToTensor', ()),
    ('Normalize', ([0.5], [1]))
]
cfg.transforms.test = cfg.transforms.train
