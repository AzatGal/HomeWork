import plotly.graph_objects as go
from linear_regression.configs.linear_regression_cfg import cfg
import numpy as np


class Visualisation:
    def __init__(self):
        pass

    # maybe do the method static?
    def compare_model_predictions(self, x_values, y_values_list, y_actual, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_values, y=y_actual, mode='markers', name='Target values'))
        for item in y_values_list:
            coefficients = np.polyfit(x_values, y_values_list[item]['valid'], len(x_values))
            polynomial = np.poly1d(coefficients)
            x_new = np.linspace(np.min(x_values), np.max(x_values), np.size(x_values))
            y_new = polynomial(x_new)
            fig.add_trace(go.Scatter(x=x_new, y=y_new, mode='lines', name=item))

        fig.update_layout(title=title)

        fig.show()

    """
        A class for visualizing results using plotly.

        This class provides methods for creating various visual representations,
        such as plots, charts, etc., based on the plotly library.


        TODO:
            - Write a function `compare_model_predictions` in the class `Visualisation` that takes in:
                  1. x_values: Array-like data for the x-axis - our inputs .
                  2. y_values_list: A list of array-like data containing predictions from different models.
                  3. y_actual: Array-like data containing actual y-values - our targets.
                  4. title: A string to be used as the plot title.
              The function should generate a plot comparing model predictions from gradient descent and normal equation methods against actual data.


        Example:
            To create a simple line chart with additional traces and a title using plotly:

            >>> import numpy as np
            >>> x = np.arange(10)
            >>> y1 = np.sin(x)
            >>> y2 = np.cos(x)

            # Create an initial plot with the sine curve
            >>> fig = go.Figure(data=go.Scatter(x=x, y=y1, mode='lines', name='sin(x)'))

            # Add a trace for the cosine curve
            >>> fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='cos(x)'))

            # Add a title to the figure
            >>> fig.update_layout(title='Sine and Cosine Curves')

            # Display the figure
            >>> fig.show()
        """

    pass
