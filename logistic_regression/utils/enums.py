from enum import IntEnum

DataProcessTypes = IntEnum('DataProcessTypes', ('standardization', 'normalization'))
SetType = IntEnum('SetType', ('train', 'valid', 'test'))
WeightsInitType = IntEnum('WeightsInitType', ('xavier_normal', 'xavier_uniform', 'he_normal', 'he_uniform'))
GDStoppingCriteria = IntEnum('GDStoppingCriteria', ('epoch', 'gradient_norm', 'difference_norm', 'metric_value'))
