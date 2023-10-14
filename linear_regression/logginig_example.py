import hashlib
import inspect
from logs.Logger import Logger
import random


def generate_experiment_name(base_functions: list, reg_coeff: float, lr: float) -> (str, str):
    # Convert base functions to string representation and hash them
    function_strings = [inspect.getsource(f).strip() for f in base_functions]
    concatenated = "\n".join(function_strings)
    hash_id = hashlib.md5(concatenated.encode()).hexdigest()[:6]  # taking the first 6 characters for brevity

    # Construct the name
    name = f"Reg{reg_coeff}_LR{lr}_FuncHash{hash_id}"

    return name, concatenated


for i in range(3):
    x_0, x_1 = random.randint(0, 4), random.randint(1, 4)
    f1 = lambda x: x + x_0
    f2 = lambda x: x * x_1

    reg_coeff = random.uniform(0, 1)
    learning_rate = random.uniform(0, 1)
    base_function = [f1, f2]

    experiment_name, base_function_str = generate_experiment_name(base_function, reg_coeff, learning_rate)
    logger = Logger(env_path='env.env', project="kkmle/Linear-Regression", experiment_name=experiment_name)

    logger.log_hyperparameters(params={
        'base_function': base_function_str,
        'regularisation_coefficient': reg_coeff,
        'learning_rate': learning_rate
    })

    for j in range(100):
        logger.save_param('train', 'mse', random.uniform(0, 0.001))
        logger.save_param('train', 'loss', random.uniform(0, 0.001))
        logger.save_param('valid', 'mse', random.uniform(0, 0.001))
        logger.save_param('valid', 'loss', random.uniform(0, 0.001))
    logger.log_final_val_mse(random.uniform(0, 0.001))
