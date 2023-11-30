import neptune
from dotenv import load_dotenv
import os
from typing import Union, List


class Logger:
    def __init__(self, env_path, project, experiment_name=None):

        load_dotenv(env_path)
        self.run = neptune.init_run(
            project=project,
            api_token=os.environ['NEPTUNE_APIKEY'],
            name=experiment_name
        )

    def log_hyperparameters(self, params: dict):
        # сохранение гиперпараметов модели
        for param, value in params.items():
            self.run[f'hyperparameters/{param}'] = value

    def save_param(self, type_set, metric_name: Union[List[str], str], metric_value: Union[List[float], float]):
        if isinstance(metric_name, List):
            for p_n, p_v in zip(metric_name, metric_value):
                self.run[f"{type_set}/{p_n}"].append(p_v)
        else:
            self.run[f"{type_set}/{metric_name}"].append(metric_value)

    def save_plot(self, type_set, plot_name, plt_fig):
        self.run[f"{type_set}/{plot_name}"].append(plt_fig)
