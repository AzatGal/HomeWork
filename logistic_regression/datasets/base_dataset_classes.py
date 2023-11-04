from abc import ABC, abstractmethod

import numpy as np


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

    @property
    @abstractmethod
    def d(self):
        # inputs variables
        pass

    def divide_into_sets(self):
        # TODO define self.inputs_train, self.targets_train, self.inputs_valid, self.targets_valid,
        #  self.inputs_test, self.targets_test; you can use your code from previous homework
        size_of_df = np.shape(self.inputs)[0]
        list_df = np.array(list(zip(self.inputs, self.targets)), dtype="object")
        np.random.shuffle(list_df)
        shuffled_inputs, shuffled_targets = zip(*list_df)
        self.size_of_train = int(self.train_set_percent * size_of_df)
        size_of_valid = int((self.train_set_percent + self.valid_set_percent) * size_of_df)
        self.inputs_train = np.array(shuffled_inputs[: self.size_of_train])
        self.targets_train = np.array(shuffled_targets[: self.size_of_train])
        self.inputs_valid = np.array(shuffled_inputs[self.size_of_train: size_of_valid])
        self.targets_valid = np.array(shuffled_targets[self.size_of_train: size_of_valid])
        self.inputs_test = np.array(shuffled_inputs[size_of_valid:])
        self.targets_test = np.array(shuffled_targets[size_of_valid:])

    def normalization(self):
        # TODO write normalization method BONUS TASK
        min_inputs = np.min(self.inputs, axis=0)
        max_inputs = np.max(self.inputs, axis=0)
        s = lambda x: 2 * (x - min_inputs) / (max_inputs - min_inputs) - 1
        self.inputs = np.vectorize(s)(self.inputs)

    def get_data_stats(self):
        # TODO calculate mean and std of inputs vectors of training set by each dimension
        self.means = np.mean(self.inputs, axis=0)
        self.stds = np.std(self.inputs, axis=0)

    def standartization(self):
        # TODO write standardization method (use stats from __get_data_stats)
        #   DON'T USE LOOP
        s = lambda x: (x - self.means) / self.stds
        self.inputs = np.vectorize(s)(self.inputs)

# я добавил, мб нужно убрать
    @inputs.setter
    def inputs(self, value):
        self._inputs = value


class BaseClassificationDataset(BaseDataset):

    @property
    @abstractmethod
    def k(self):
        # number of classes
        pass

    @staticmethod
    def onehotencoding(targets: np.ndarray, number_classes: int) -> np.ndarray:
        # TODO create matrix of onehot encoding vactors for input targets
        # it is possible to do it without loop try make it without loop
        size = np.size(targets)
        one_hot = np.zeros((size, number_classes))
        one_hot[np.arange(size), targets] = 1
        return one_hot
