from easydict import EasyDict
from multilayer_perceptron.utils.enums import InitWeightType

cfg = EasyDict()

cfg.layers = [
    ('Linear', {'in_features': 28 * 28, 'out_features': 200}),
    ('ReLU', {}),
    ('Linear', {'in_features': 200, 'out_features': 10}),
]

cfg.init_type = InitWeightType.xavier_normal_.name
