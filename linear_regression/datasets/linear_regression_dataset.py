import numpy as np
from linear_regression.utils.common_functions import read_dataframe_file
from easydict import EasyDict
from linear_regression.datasets.base_dataset import BaseDataset
from linear_regression.utils.enums import SetType


class LinRegDataset(BaseDataset):

    def __init__(self, cfg: EasyDict, inputs_cols='inputs', target_cols='targets'):
        super(LinRegDataset, self).__init__(cfg.train_set_percent, cfg.valid_set_percent)

        dataframe = read_dataframe_file(cfg.dataframe_path)

        # define properties
        self.inputs = np.asarray(dataframe[inputs_cols])
        self.targets = np.asarray(dataframe[target_cols])

        # divide into sets
        self._divide_into_sets()

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        if isinstance(value, np.ndarray):
            self._inputs = value

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, value):
        self._targets = value

    def __call__(self, set_type: SetType) -> dict:
        return {'inputs': getattr(self, f'inputs_{set_type.name}'),
                'targets': getattr(self, f'targets_{set_type.name}'), }


if __name__ == '__main__':
    from configs.linear_regression_cfg import cfg

    lin_reg_dataset = LinRegDataset(cfg, ['x_0', 'x_1', 'x_2'])
