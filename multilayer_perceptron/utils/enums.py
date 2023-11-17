from enum import IntEnum


class InitWeightType(IntEnum):
    xavier_uniform_ = 1
    xavier_normal_ = 2
    kaiming_uniform_ = 3
    kaiming_normal_ = 4
    uniform_ = 5
    normal_ = 6
    zeros_ = 7

