from enum import IntEnum

SetType = IntEnum('SetType', ('train', 'valid', 'test'))
TrainType = IntEnum('TrainType', ('train', 'gradient_descent', 'normal_equation'))
