from easydict import EasyDict
from linear_regression.utils.enums import TrainType

cfg = EasyDict()

# Path to the dataframe
cfg.dataframe_path = '/Users/azatgalautdinov/PycharmProjects/ML_Homework/linear_regression_dataset.csv'

# cfg.base_functions contains callable functions to transform input features.
# E.g., for polynomial regression: [lambda x: x, lambda x: x**2]
# TODO You should populate this list with suitable functions based on the requirements.
#cfg.base_functions = [lambda x: x**copy.copy(i) for i in range(cfg.degree + 1)]

cfg.base_functions = [lambda x, power=i: x ** power for i in range(1, 1)]

cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1

# Specifies the type of training algorithm to be used
cfg.train_type = TrainType.gradient_descent

# how many times the algorithm will process the entire dataset for gradient descent algorithm
cfg.epoch = 100

# cfg.exp_name = ''
cfg.env_path = '/Users/azatgalautdinov/PycharmProjects/ML_Homework/api_token.env'  # Путь до файла .env где будет храниться api_token.
cfg.project_name = 'azat.galyautdinov161002/ML-Homework1'
