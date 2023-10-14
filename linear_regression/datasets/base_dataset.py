import random

import numpy as np
from abc import ABC, abstractmethod


class BaseDataset(ABC):

    def __init__(self, train_set_percent, valid_set_percent):
        self.train_set_percent = train_set_percent
        self.valid_set_percent = valid_set_percent

    @property
    @abstractmethod
    def targets(self):
        # targets variables
        pass

    @property
    @abstractmethod
    def inputs(self):
        # inputs variables
        pass

    def _divide_into_sets(self):
        # TODO define self.inputs_train, self.targets_train, self.inputs_valid, self.targets_valid,
        #  self.inputs_test, self.targets_test
        size_of_df = np.size(self.inputs)
        list_df = np.array(list(zip(self.inputs, self.targets)))
        np.random.shuffle(list_df)
        shuffled_inputs, shuffled_targets = zip(*list_df)
        size_of_train = int(self.train_set_percent * size_of_df)
        size_of_valid = int((self.train_set_percent + self.valid_set_percent) * size_of_df)
        self.inputs_train = np.array(shuffled_inputs[: size_of_train])
        self.targets_train = np.array(shuffled_targets[: size_of_train])
        self.inputs_valid = np.array(shuffled_inputs[size_of_train: size_of_valid])
        self.targets_valid = np.array(shuffled_targets[size_of_train: size_of_valid])
        self.inputs_test = np.array(shuffled_inputs[size_of_valid:])
        self.targets_test = np.array(shuffled_targets[size_of_valid:])


