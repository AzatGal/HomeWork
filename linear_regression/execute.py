# TODO:
#  1. Load the dataset using pandas' read_csv function.
#  2. Split the dataset into training, validation, and test sets. Consider using a split ratio such as 80:10:10 or another appropriate ratio.
#  3. Initialize the Linear Regression model using the provided `LinearRegression` class
#  4. Train the model using the training data.
#  5. Evaluate the trained model on the validation set,train set, test set. You might consider metrics like Mean Squared Error (MSE) for evaluation.
#  6. Plot the model's predictions against the actual values from the validation set using the `Visualisation` class.

import numpy as np
from models.linear_regression_model import LinearRegression
from datasets.linear_regression_dataset import LinRegDataset
from configs.linear_regression_cfg import cfg
from utils.enums import TrainType
from utils.visualisation import Visualisation
from utils.metrics import MSE

degree = [1, 8, 100]
train_type = {"gradient descent": TrainType.gradient_descent,
              "normal equation": TrainType.normal_equation}
data = LinRegDataset(cfg)
predictions = {"gradient descent": {},
               "normal equation": {}}
mse = {"gradient descent": {},
       "normal equation": {}}

for i in degree:
    for j in train_type:
        cfg.degree = i
        cfg.base_functions = [lambda x, power=i: x ** power for i in range(1, cfg.degree + 1)]
        cfg.train_type = train_type[j]
        model = LinearRegression(cfg.base_functions, 0.01, 0, 'W')
        model.train(data.inputs_train, data.targets_train)
        predictions[j] = {"train": model.__call__(data.inputs_train),
                          "valid": model.__call__(data.inputs_valid),
                          "test": model.__call__(data.inputs_test)}
        mse[j] = {"train": MSE(predictions[j]["train"], data.targets_train),
                  "valid": MSE(predictions[j]["valid"], data.targets_valid),
                  "test": MSE(predictions[j]["test"], data.targets_test)}
    graph = Visualisation()
    graph.compare_model_predictions(data.inputs_valid, predictions, data.targets_valid,
                                    f"polynomial degree: {i}, normal equation MSE: {mse['normal equation']['test']}, "
                                    f"gradient decent MSE: {mse['gradient descent']['test']}")
