# TODO:
#  1) Load the dataset using pandas read_csv function.
#  2) Split the dataset into training, validation, and test sets.
#  Consider using a split ratio such as 80:10:10 or another appropriate ratio.
#  Use class from datasets.linear_regression_dataset.py
#  3) Define hyperparameters space
#  4) Use loop where you randomly choose hypeparameter from space and train model
#  5) Create experiment name using code from logging_example.py
#  6) Initialize the Linear Regression model using the provided `LinearRegression` class
#  7) Log hyperparameters to neptune
#  8) Train the model using the training data and gradient descent,
#  log MSE and cost function on validation and trainig sets
#  9) Log final mse on validation set after trainig
#  10) Save model if it is showing best mse on validation set
import hashlib
import inspect
import random
import numpy as np
from models.linear_regression_model import LinearRegression
from datasets.linear_regression_dataset import LinRegDataset
from configs.linear_regression_cfg import cfg
from utils.enums import TrainType
from utils.visualisation import Visualisation
from utils.metrics import MSE
from logs.Logger import Logger


def generate_experiment_name(base_functions: list, reg_coeff: float, lr: float) -> (str, str):
    # Convert base functions to string representation and hash them
    function_strings = [inspect.getsource(f).strip() for f in base_functions]
    concatenated = "\n".join(function_strings)
    hash_id = hashlib.md5(concatenated.encode()).hexdigest()[:6]  # taking the first 6 characters for brevity

    # Construct the name
    name = f"Reg{reg_coeff}_LR{lr}_FuncHash{hash_id}"

    return name, concatenated


M_min, M_max = 1, 53
rc_min, rc_max = 0, 1
lr_min, lr_max = 0, 1

base_functions = [lambda x, power=i: x ** power for i in range(1, 50 + 1)] + \
                 [lambda x: np.sin(x), lambda x: np.exp(x), lambda x: np.cos(x)]

data = LinRegDataset(cfg)
base_functions_str = []
models = []

for i in range(30):
    cfg.base_functions = np.random.choice(base_functions, size=random.randint(M_min, M_max), replace=False)
    reg_coeff = random.uniform(rc_min, rc_max)
    learning_rate = random.uniform(lr_min, lr_max)
    cfg.train_type = TrainType.gradient_descent

    experiment_name, base_function_str = generate_experiment_name(base_functions, reg_coeff, learning_rate)
    base_functions_str.append(base_function_str)
    model = LinearRegression(cfg.base_functions, learning_rate, reg_coeff, experiment_name)
    model.train(data.inputs_train, data.targets_train)

    model.neptune_logger.log_hyperparameters(params={
        'base_function': base_function_str,
        'regularisation_coefficient': reg_coeff,
        'learning_rate': learning_rate
    })

    model.neptune_logger.save_param('train', 'mse', MSE(model.__call__(data.inputs_train), data.targets_train))
    for item in model.cost_functions:
        model.neptune_logger.save_param('train', 'loss', item)
    model.neptune_logger.save_param('valid', 'mse', MSE(model.__call__(data.inputs_valid), data.targets_valid))
    model.neptune_logger.log_final_val_mse(MSE(model.__call__(data.inputs_valid), data.targets_valid))

    models.append(model)

final_mse = [MSE(item.__call__(data.inputs_valid), data.targets_valid) for item in models]
index_of_best_model = np.argmin(final_mse)
best_model = models[index_of_best_model]

best_model.neptune_logger = None
best_model.save('model.pkl')



