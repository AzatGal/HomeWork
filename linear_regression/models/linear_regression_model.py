import numpy as np
from linear_regression.configs.linear_regression_cfg import cfg
from linear_regression.utils.enums import TrainType
import sys
from linear_regression.logs.Logger import Logger
import cloudpickle


class LinearRegression:
    def __init__(self, base_functions: list, learning_rate: float, reg_coefficient: float, experiment_name: str):
        self.cost_functions = [0]*cfg.epoch
        self.weights = np.random.randn(len(base_functions) + 1)  # init weights using np.random.randn (normal distribution with mean=0 and variance=1).
        self.base_functions = np.array(base_functions)
        self.learning_rate = learning_rate
        self.reg_coefficient = reg_coefficient
        self.neptune_logger = Logger(cfg.env_path, cfg.project_name, experiment_name)

    # Methods related to the Normal Equation

    def _pseudoinverse_matrix(self, matrix: np.ndarray) -> np.ndarray:
        u, s, vh = np.linalg.svd(matrix)  # , full_matrices=False)  # False
        eps = sys.float_info.epsilon
        max_s = np.max(s)
        s_s = np.zeros_like(matrix)
        s_sh = np.where(s > eps * max(self.n, self.m + 1) * max_s, s / (s**2 + self.reg_coefficient), 0)
        s_s[:len(s_sh), :len(s_sh)] = np.diag(s_sh)
        # s_s[0][0] = 1 / s[0][0]
        """Compute the pseudoinverse of a matrix using SVD.

        The pseudoinverse (Φ^+) of the design matrix Φ can be computed using the formula:

        Φ^+ = V * Σ^+ * U^T

        Where:
        - U, Σ, and V are the matrices resulting from the SVD of Φ.

        The Σ^+ is computed as:

        Σ'_{i,j} =
        | 1/Σ_{i,j}, if Σ_{i,j} > ε * max(N, M+1) * max(Σ)
        | 0, otherwise

        and then:
        Σ^+ = Σ'^T

        where:
        - ε is the machine epsilon, which can be obtained in Python using:
            ε = sys.float_info.epsilon
        - N is the number of rows in the design matrix.
        - M is the number of base functions (without φ_0(x_i)=1).

        For regularisation

        Σ'_{i,j} =
        | Σ_{i,j}/(Σ_{i,j}ˆ2 + λ) , if Σ_{i,j} > ε * max(N, M+1) * max(Σ)
        | 0, otherwise

        Note that Σ'_[0,0] = 1/Σ_{i,j}

        TODO: Add regularisation
        """
        return vh.T @ s_s.T @ u.T
        pass

    def _calculate_weights(self, pseudoinverse_plan_matrix: np.ndarray, targets: np.ndarray) -> None:
        self.weights = pseudoinverse_plan_matrix @ targets  # @
        """Calculate the optimal weights using the normal equation.

            The weights (w) can be computed using the formula:

            w = Φ^+ * t

            Where:
            - Φ^+ is the pseudoinverse of the design matrix and can be defined as:
                Φ^+ = (Φ^T * Φ)^(-1) * Φ^T

            - t is th e target vector.

            TODO: Implement this method. Calculate  Φ^+ using _pseudoinverse_matrix function
        """
        pass

    # General methods
    def _plan_matrix(self, inputs: np.ndarray) -> np.ndarray:
        self.n = np.size(inputs)
        self.m = np.size(self.base_functions)
        phi = np.ones((self.n, self.m + 1))
        for i in range(self.m):
            phi[:, i + 1] = np.vectorize(self.base_functions[i])(inputs)
        """Construct the design matrix (Φ) using base functions.

            The structure of the matrix Φ is as follows:

            Φ = [ [ φ_0(x_1), φ_1(x_1), ..., φ_M(x_1) ],
                  [ φ_0(x_2), φ_1(x_2), ..., φ_M(x_2) ],
                  ...
                  [ φ_0(x_N), φ_1(x_N), ..., φ_M(x_N) ] ]

            where:
            - x_i denotes the i-th input vector.
            - φ_j(x_i) represents the j-th base function applied to the i-th input vector.
            - M is the total number of base functions (without φ_0(x_i)=1).
            - N is the total number of input vectors.

            TODO: Implement this method using one loop over the base functions.

        """
        return phi
        pass

    def calculate_model_prediction(self, plan_matrix: np.ndarray) -> np.ndarray:
        w_t = self.weights.T
        y_pred = plan_matrix @ w_t
        """Calculate the predictions of the model.

            The prediction (y_pred) can be computed using the formula:

            y_pred = Φ * w^T

            Where:
            - Φ is the design matrix.
            - w^T is the transpose of the weight vector.

            To compute multiplication in Python using numpy, you can use:
            - `numpy.dot(a, b)`
            OR
            - `a @ b`

        TODO: Implement this method without using loop

        """
        return y_pred
        pass

    # Methods related to Gradient Descent
    def _calculate_gradient(self, plan_matrix: np.ndarray, targets: np.ndarray) -> np.ndarray:
        grad_w_e = (2 / self.n) * plan_matrix.transpose() @ (plan_matrix @ self.weights - targets) \
                   + self.reg_coefficient * self.weights
        """Calculate the gradient of the cost function with respect to the weights.

            The gradient of the error with respect to the weights (∆w E) can be computed using the formula:

            ∆w E = (2/N) * Φ^T * (Φ * w - t)

            Where:
            - Φ is the design matrix.
            - w is the weight vector.
            - t is the vector of target values.
            - N is the number of data points.

            This formula represents the partial derivative of the mean squared error with respect to the weights.

            For regularisation
            ∆w E = (2/N) * Φ^T * (Φ * w - t)  + λ * w

            TODO: Implement this method using matrix operations in numpy. a.T - transpose. Do not use loops
            TODO: Add regularisation
        """
        return grad_w_e
        pass

    def calculate_cost_function(self, plan_matrix, targets):
        pred = plan_matrix @ self.weights
        e = (1 / self.n) * np.sum(np.square(targets - pred)) + \
            self.reg_coefficient * self.weights.T @ self.weights
        """Calculate the cost function value for the current weights.

        The cost function E(w) represents the mean squared error and is given by:

        E(w) = (1/N) * ∑(t - Φ * w^T)^2

        Where:
        - Φ is the design matrix.
        - w is the weight vector.
        - t is the vector of target values.
        - N is the number of data points.

        For regularisation
        E(w) = (1/N) * ∑(t - Φ * w^T)^2 + λ * w^T * w


        TODO: Implement this method using numpy operations to compute the mean squared error. Do not use loops
        TODO: Add regularisation

        """
        return e
        pass

    def train(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """Train the model using either the normal equation or gradient descent based on the configuration.
        TODO: Complete the training process.
        """
        plan_matrix = self._plan_matrix(inputs)
        if cfg.train_type == TrainType.normal_equation:  # .value
            pseudoinverse_plan_matrix = self._pseudoinverse_matrix(plan_matrix)
            # train process
            self._calculate_weights(pseudoinverse_plan_matrix, targets)
        else:
            """
            At each iteration of gradient descent, the weights are updated using the formula:
        
            w_{k+1} = w_k - γ * ∇_w E(w_k)
        
            Where:
            - w_k is the current weight vector at iteration k.
            - γ is the learning rate, determining the step size in the direction of the negative gradient.
            - ∇_w E(w_k) is the gradient of the cost function E with respect to the weights w at iteration k.
        
            This iterative process aims to find the weights that minimize the cost function E(w).
            """
            for e in range(cfg.epoch):
                gradient = self._calculate_gradient(plan_matrix, targets)
                self.weights = self.weights - self.learning_rate * gradient

                # update weights w_{k+1} = w_k - γ * ∇_w E(w_k)

                if e % 10 == 0:
                    t = self.calculate_cost_function(plan_matrix, targets)
                    self.cost_functions[e] = t
                    print(t)
                    # TODO: Print the cost function's value.
                    pass

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """return prediction of the model"""
        plan_matrix = self._plan_matrix(inputs)
        predictions = self.calculate_model_prediction(plan_matrix)

        return predictions

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            return cloudpickle.load(f)
