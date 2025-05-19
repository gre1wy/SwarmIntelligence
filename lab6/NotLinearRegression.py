import numpy as np
from DE_algorithm import DE_algorithm
from PSO_algorithm import PSO_algorithm
class NotLinearRegression:
    def __init__(self, model_func, bounds, x_data, y_data):
        """
        Constructor for NotLinearRegression class

        Parameters
        ----------
        model_func : function
            Nonlinear regression model function, should take two parameters: B and x,
            where B is a vector of parameters and x is a vector of input data.
        bounds : list of tuples
            List of tuples, where each tuple contains lower and upper bounds for
            corresponding parameter in B.
        x_data : array-like
            Array of input data.
        y_data : array-like
            Array of output data.

        Attributes
        ----------
        best_params_de : array-like
            Best parameters found by DE algorithm.
        best_MSE_de : float
            Best MSE found by DE algorithm.
        loss_history_de : list
            List of losses on each iteration of DE algorithm.
        best_individuals_de : list
            List of best individuals found by DE algorithm on each iteration.

        best_params_pso : array-like
            Best parameters found by PSO algorithm.
        best_MSE_pso : float
            Best MSE found by PSO algorithm.
        loss_history_pso : list
            List of losses on each iteration of PSO algorithm.
        best_individuals_pso : list
            List of best individuals found by PSO algorithm on each iteration.
        """
        self.model_func = model_func        # функция вида: f(B, x)
        self.bounds = np.array(bounds)
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)

        # Для DE
        self.best_params_de = None
        self.best_MSE_de = None
        self.loss_history_de = []
        self.best_individuals_de = []

        # Для PSO
        self.best_params_pso = None
        self.best_MSE_pso = None
        self.loss_history_pso = []
        self.best_individuals_pso = []

    def _MSE(self, B):
        """
        Calculates Mean Squared Error (MSE) between real and predicted values

        Parameters
        ----------
        B : array-like
            Vector of parameters

        Returns
        -------
        mse : float
            Mean Squared Error
        """

        y_pred = self.model_func(B, self.x_data)
        mse = np.mean((self.y_data - y_pred) ** 2)
        return mse

    def fit_DE(self, **kwargs):
        """
        Fits the model using Differential Evolution algorithm.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to DE_algorithm.

        Returns
        -------
        None
        """
        
        optimizer = DE_algorithm(self._MSE, self.bounds, **kwargs)
        best_params, best_MSE, best_inds, loss_hist = optimizer.optimize()
        self.best_params_de = best_params
        self.best_MSE_de = best_MSE
        self.best_individuals_de = best_inds
        self.loss_history_de = loss_hist

    def fit_PSO(self, **kwargs):
        """
        Fits the model using Particle Swarm Optimization algorithm.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to PSO_algorithm.

        Returns
        -------
        None
        """
        optimizer = PSO_algorithm(self._MSE, self.bounds, **kwargs)
        best_params, best_MSE, best_inds, loss_hist = optimizer.optimize()
        self.best_params_pso = best_params
        self.best_MSE_pso = best_MSE
        self.best_individuals_pso = best_inds
        self.loss_history_pso = loss_hist

    def predict(self, x_new, method='DE'):
        """
        Predicts the output values for given input x_new using the best parameters found by the selected optimization method.

        Parameters
        ----------
        x_new : array-like
            Input data to predict the output for.
        method : str, {'DE', 'PSO'}, optional
            Optimization method to use. Defaults to 'DE'.

        Returns
        -------
        y_pred : array-like
            Predicted output values.

        Raises
        ------
        Exception
            If the model has not been trained with the selected method yet.
        ValueError
            If method is not 'DE' or 'PSO'.
        """
        if method == 'DE':
            if self.best_params_de is None:
                raise Exception("Model has not been trained with DE yet. Call fit_DE() first.")
            return self.model_func(self.best_params_de, x_new)
        elif method == 'PSO':
            if self.best_params_pso is None:
                raise Exception("Model has not been trained with PSO yet. Call fit_PSO() first.")
            return self.model_func(self.best_params_pso, x_new)
        else:
            raise ValueError("Unsupported method for prediction. Use 'DE' or 'PSO'.")
