from easydict import EasyDict
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()

cfg.path = os.path.join(ROOT_DIR, 'data/oxford-iiit-pet/')
cfg.nrof_classes = 37

cfg.annotation_filenames = {
    'train': 'trainval.txt',
    'test': 'test.txt'
}

cfg.transforms = EasyDict()
cfg.transforms.train = [
    ('RandomResizedCrop', ((224, 224),)),
    ('ToTensor', ()),
    ('Normalize', ([0.485, 0.456, 0.405], [0.229, 0.224, 0.225]))
]
cfg.transforms.test = [
    ('Resize', ((256, 256),)),
    ('CenterCrop', ((224, 224),)),
    ('ToTensor', ()),
    ('Normalize', ([0.485, 0.456, 0.405], [0.229, 0.224, 0.225]))
]