import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from NotLinearRegression import NotLinearRegression
from matplotlib.animation import FuncAnimation

class RegressionPipeline:
    def __init__(self, model_func, bounds, x_data, y_data, test_size=0.2, random_state=None):
        """
        Initialize the RegressionPipeline with provided model function, data, and configuration.

        Parameters:
        model_func (callable): The model function to be used for regression.
        bounds (array-like): Bounds for the model parameters.
        x_data (array-like): Input feature data.
        y_data (array-like): Target data corresponding to x_data.
        test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int, optional): Controls the shuffling applied to the data before splitting. Default is None.

        Attributes:
        model_func (callable): Stored model function.
        bounds (array-like): Stored bounds for the model parameters.
        x_data (np.ndarray): Stored input feature data.
        y_data (np.ndarray): Stored target data.
        x_train (np.ndarray): Training input data.
        x_test (np.ndarray): Testing input data.
        y_train (np.ndarray): Training target data.
        y_test (np.ndarray): Testing target data.
        train_mask (np.ndarray): Boolean mask indicating training samples.
        model (NotLinearRegression): Instance of NotLinearRegression for the training data.
        trained_methods (set): Set to store names of trained methods.
        """

        self.model_func = model_func
        self.bounds = bounds
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)

        self.x_train, self.x_test, self.y_train, self.y_test, train_idx, _ = train_test_split(
            self.x_data, self.y_data, np.arange(len(self.x_data)),
            test_size=test_size, random_state=random_state
        )
        self.train_mask = np.zeros_like(self.y_data, dtype=bool)
        self.train_mask[train_idx] = True

        self.model = NotLinearRegression(model_func, bounds, self.x_train, self.y_train)
        self.trained_methods = set()

    def train(self, method='DE', **kwargs):
        
        """
        Trains the model using the specified optimization method.

        Parameters:
        method (str, {'DE', 'PSO'}): Optimization method to use. Defaults to 'DE'.
        **kwargs : dict
            Additional keyword arguments to pass to the optimization algorithm.

        Raises:
        ValueError
            If the specified method is not supported. Use 'DE' or 'PSO'.
        """
        method = method.upper()
        if method == 'DE':
            self.model.fit_DE(**kwargs)
            self.trained_methods.add('DE')
        elif method == 'PSO':
            self.model.fit_PSO(**kwargs)
            self.trained_methods.add('PSO')
        else:
            raise ValueError("Unsupported training method. Use 'DE' or 'PSO'.")

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
        method = method.upper()
        if method not in self.trained_methods:
            raise Exception(f"Model not trained with method '{method}'. Call train() first with this method.")
        return self.model.predict(x_new, method=method)

    def get_loss_history(self, method='DE'):
        """
        Returns the loss history for the specified optimization method.

        Parameters
        ----------
        method : str, {'DE', 'PSO'}, optional
            Optimization method to use. Defaults to 'DE'.

        Returns
        -------
        loss_history : list
            Loss values at each iteration of the optimization algorithm.

        Raises
        ------
        ValueError
            If method is not 'DE' or 'PSO'.
        """
        method = method.upper()
        if method == 'DE':
            return self.model.loss_history_de
        elif method == 'PSO':
            return self.model.loss_history_pso
        else:
            raise ValueError("Unsupported method for loss history. Use 'DE' or 'PSO'.")

    def get_best_params(self, method='DE'):
        """
        Returns the best parameters found by the specified optimization method.

        Parameters
        ----------
        method : str, {'DE', 'PSO'}, optional
            Optimization method to use. Defaults to 'DE'.

        Returns
        -------
        best_params : array-like
            The best parameters found by the optimization algorithm.

        Raises
        ------
        ValueError
            If method is not 'DE' or 'PSO'.
        """
        method = method.upper()
        if method == 'DE':
            return self.model.best_params_de
        elif method == 'PSO':
            return self.model.best_params_pso
        else:
            raise ValueError("Unsupported method for best params. Use 'DE' or 'PSO'.")

    def plot_results(self, methods_to_plot=None):
        """
        Plots the results of the optimization.

        Parameters
        ----------
        methods_to_plot : list of str, optional
            List of methods to plot. If not provided, all trained methods are plotted.

        Notes
        -----
        The plots are split into two rows and three columns. The first row shows the overall regression, regression
        on the train set, and regression on the test set. The second row shows the convergence of the mean squared
        error (MSE) on the train and test sets for each method, as well as the final MSE value in the legend.
        """
        if methods_to_plot is None:
            methods_to_plot = self.trained_methods

        x_plot = np.linspace(np.min(self.x_data), np.max(self.x_data), 200)
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))

        for ax in axs.flat:
            ax.grid(True)

        # --- (0, 0): Overall Regression ---
        axs[0, 0].scatter(self.x_data[self.train_mask], self.y_data[self.train_mask],
                        label='Train', color='black', alpha=0.8)
        axs[0, 0].scatter(self.x_data[~self.train_mask], self.y_data[~self.train_mask],
                        label='Test', color='green', alpha=0.8)

        for method in methods_to_plot:
            y_pred = self.predict(x_plot, method=method)
            axs[0, 0].plot(x_plot, y_pred, label=f'Regression {method}', linewidth=2)

        axs[0, 0].set_title('Overall Regression')
        axs[0, 0].set_ylabel('y')
        axs[0, 0].legend()

        # --- (0, 1): Regression on Train Set ---
        axs[0, 1].scatter(self.x_data[self.train_mask], self.y_data[self.train_mask],
                        label='Train Points', color='black', alpha=0.8)
        for method in methods_to_plot:
            y_pred = self.predict(x_plot, method=method)
            axs[0, 1].plot(x_plot, y_pred, label=f'{method} Regression', linewidth=2)
        axs[0, 1].set_title('Regression on Training Set')
        axs[0, 1].legend()

        # --- (0, 2): Regression on Test Set ---
        axs[0, 2].scatter(self.x_data[~self.train_mask], self.y_data[~self.train_mask],
                        label='Test Points', color='green', alpha=0.8)
        for method in methods_to_plot:
            y_pred = self.predict(x_plot, method=method)
            axs[0, 2].plot(x_plot, y_pred, label=f'{method} Regression', linewidth=2)
        axs[0, 2].set_title('Regression on Test Set')
        axs[0, 2].legend()

        # --- (1, 0): Convergence ---
        for method in methods_to_plot:
            best_individuals = getattr(self.model, f"best_individuals_{method.lower()}")
            color = 'blue' if method == 'DE' else 'red'

            train_losses = [
                np.mean((self.y_data[self.train_mask] - self.model_func(p, self.x_data)[self.train_mask]) ** 2)
                for p in best_individuals
            ]
            test_losses = [
                np.mean((self.y_data[~self.train_mask] - self.model_func(p, self.x_data)[~self.train_mask]) ** 2)
                for p in best_individuals
            ]

            axs[1, 0].plot(train_losses, label=f'{method} Train MSE', linestyle='-', color=color)
            axs[1, 0].plot(test_losses, label=f'{method} Test MSE', linestyle='--', color=color)

        axs[1, 0].set_title('MSE Convergence (Train/Test)')
        axs[1, 0].set_xlabel('Iterations')
        axs[1, 0].set_ylabel('MSE')
        axs[1, 0].legend()

        # --- (1, 1): Train MSE with final value in legend ---
        for method in methods_to_plot:
            best_individuals = getattr(self.model, f"best_individuals_{method.lower()}")
            color = 'blue' if method == 'DE' else 'red'

            train_losses = [
                np.mean((self.y_data[self.train_mask] - self.model_func(p, self.x_data)[self.train_mask]) ** 2)
                for p in best_individuals
            ]
            final_train_loss = train_losses[-1]
            axs[1, 1].plot(train_losses, label=f'{method} Train MSE (final={final_train_loss:.6f})',
                        color=color)

        axs[1, 1].set_title('Train MSE')
        axs[1, 1].set_xlabel('Iterations')
        axs[1, 1].set_ylabel('Train MSE')
        axs[1, 1].legend()

        # --- (1, 2): Test MSE with final value in legend ---
        for method in methods_to_plot:
            best_individuals = getattr(self.model, f"best_individuals_{method.lower()}")
            color = 'blue' if method == 'DE' else 'red'

            test_losses = [
                np.mean((self.y_data[~self.train_mask] - self.model_func(p, self.x_data)[~self.train_mask]) ** 2)
                for p in best_individuals
            ]
            final_test_loss = test_losses[-1]
            axs[1, 2].plot(test_losses, label=f'{method} Test MSE (final={final_test_loss:.6f})',
                        color=color)

        axs[1, 2].set_title('Test MSE')
        axs[1, 2].set_xlabel('Iterations')
        axs[1, 2].set_ylabel('Test MSE')
        axs[1, 2].legend()

        plt.tight_layout()
        plt.show()
        
    def create_animation(self, methods_to_plot=None, save_path=None, interval=100):
        """
        Creates an animation of the optimization process.

        Parameters
        ----------
        methods_to_plot : list of str, optional
            List of methods to plot. If not provided, all trained methods are plotted.
        save_path : str, optional
            Path to save the animation. If not provided, the animation is not saved.
        interval : int, optional
            Interval between frames in milliseconds.

        Returns
        -------
        None
        """
    
        if methods_to_plot is None:
            methods_to_plot = list(self.trained_methods)
        else:
            methods_to_plot = [m.upper() for m in methods_to_plot]

        x_plot = np.linspace(np.min(self.x_data), np.max(self.x_data), 200)
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))

        # Статичный scatter для Train/Test на верхних графиках
        axs[0, 0].scatter(self.x_data[self.train_mask], self.y_data[self.train_mask],
                        label='Train', color='black', alpha=0.8)
        axs[0, 0].scatter(self.x_data[~self.train_mask], self.y_data[~self.train_mask],
                        label='Test', color='green', alpha=0.8)
        axs[0, 1].scatter(self.x_data[self.train_mask], self.y_data[self.train_mask],
                        label='Train Points', color='black', alpha=0.8)
        axs[0, 2].scatter(self.x_data[~self.train_mask], self.y_data[~self.train_mask],
                        label='Test Points', color='green', alpha=0.8)

        for ax in axs.flat:
            ax.grid(True)

        # Словари для линий, которые будем анимировать
        line_preds = {}         # (0,0) — overall regression
        line_preds_train = {}   # (0,1) — regression on train set
        line_preds_test = {}    # (0,2) — regression on test set
        lines_train_mse = {}    # (1,0) — mse convergence train
        lines_test_mse = {}     # (1,0) — mse convergence test
        line_train_mse_final = {}  # (1,1) — train mse final legend
        line_test_mse_final = {}   # (1,2) — test mse final legend

        colors = {'DE': 'blue', 'PSO': 'red'}

        for method in methods_to_plot:
            # Линии регрессии
            line_preds[method], = axs[0, 0].plot([], [], label=f'Regression {method}', linewidth=2)
            line_preds_train[method], = axs[0, 1].plot([], [], label=f'{method} Regression', linewidth=2)
            line_preds_test[method], = axs[0, 2].plot([], [], label=f'{method} Regression', linewidth=2)

            # Линии MSE (train/test convergence)
            lines_train_mse[method], = axs[1, 0].plot([], [], linestyle='-', color=colors[method], label=f'{method} Train MSE')
            lines_test_mse[method], = axs[1, 0].plot([], [], linestyle='--', color=colors[method], label=f'{method} Test MSE')

            # Линии Train MSE и Test MSE с финальным значением
            line_train_mse_final[method], = axs[1, 1].plot([], [], color=colors[method])
            line_test_mse_final[method], = axs[1, 2].plot([], [], color=colors[method])

        # Настроим заголовки, легенды и подписи
        axs[0, 0].set_title('Overall Regression')
        axs[0, 0].set_ylabel('y')
        axs[0, 0].legend()

        axs[0, 1].set_title('Regression on Training Set')
        axs[0, 1].legend()

        axs[0, 2].set_title('Regression on Test Set')
        axs[0, 2].legend()

        axs[1, 0].set_title('MSE Convergence (Train/Test)')
        axs[1, 0].set_xlabel('Iterations')
        axs[1, 0].set_ylabel('MSE')
        axs[1, 0].legend()

        axs[1, 1].set_title('Train MSE')
        axs[1, 1].set_xlabel('Iterations')
        axs[1, 1].set_ylabel('Train MSE')

        axs[1, 2].set_title('Test MSE')
        axs[1, 2].set_xlabel('Iterations')
        axs[1, 2].set_ylabel('Test MSE')

        # Получаем лучших индивидов для каждого метода
        best_inds_dict = {}
        max_iters = 0
        for method in methods_to_plot:
            best_inds = getattr(self.model, f"best_individuals_{method.lower()}", [])
            best_inds_dict[method] = best_inds
            max_iters = max(max_iters, len(best_inds))

        def init():
            # Обнуляем все линии
            for d in [line_preds, line_preds_train, line_preds_test,
                    lines_train_mse, lines_test_mse,
                    line_train_mse_final, line_test_mse_final]:
                for line in d.values():
                    line.set_data([], [])
            return list(line_preds.values()) + list(line_preds_train.values()) + list(line_preds_test.values()) + \
                list(lines_train_mse.values()) + list(lines_test_mse.values()) + \
                list(line_train_mse_final.values()) + list(line_test_mse_final.values())

        def update(frame):
            for method in methods_to_plot:
                best_inds = best_inds_dict[method]
                if frame < len(best_inds):
                    params = best_inds[frame]
                else:
                    params = best_inds[-1]

                y_pred = self.model_func(params, x_plot)

                # Обновляем линии регрессии
                line_preds[method].set_data(x_plot, y_pred)
                line_preds_train[method].set_data(x_plot, y_pred)
                line_preds_test[method].set_data(x_plot, y_pred)

                # Считаем MSE по train и test для всех итераций до frame включительно
                train_losses = [
                    np.mean((self.y_data[self.train_mask] - self.model_func(p, self.x_data)[self.train_mask]) ** 2)
                    for p in best_inds[:frame+1]
                ]
                test_losses = [
                    np.mean((self.y_data[~self.train_mask] - self.model_func(p, self.x_data)[~self.train_mask]) ** 2)
                    for p in best_inds[:frame+1]
                ]

                x_range = np.arange(len(train_losses))

                # Обновляем линии MSE convergence (1,0)
                lines_train_mse[method].set_data(x_range, train_losses)
                lines_test_mse[method].set_data(x_range, test_losses)

                # Обновляем линии Train MSE (1,1) и Test MSE (1,2)
                line_train_mse_final[method].set_data(x_range, train_losses)
                line_test_mse_final[method].set_data(x_range, test_losses)

            max_x = max_iters if max_iters > 0 else 1
            axs[1, 0].set_xlim(0, max_x)
            axs[1, 1].set_xlim(0, max_x)
            axs[1, 2].set_xlim(0, max_x)

            y_min = -0.01
            y_max = max(train_losses)*1.1
            axs[1, 0].set_ylim(y_min, y_max)
            axs[1, 1].set_ylim(y_min, y_max)
            axs[1, 2].set_ylim(y_min, y_max)
            
            # Обновим легенду с финальными значениями MSE для всех методов (после цикла)
            axs[1, 1].legend([
                f'{m} Train MSE (final={np.mean((self.y_data[self.train_mask] - self.model_func(best_inds_dict[m][frame if frame < len(best_inds_dict[m]) else -1], self.x_data)[self.train_mask]) ** 2):.6f})'
                for m in methods_to_plot
            ])

            axs[1, 2].legend([
                f'{m} Test MSE (final={np.mean((self.y_data[~self.train_mask] - self.model_func(best_inds_dict[m][frame if frame < len(best_inds_dict[m]) else -1], self.x_data)[~self.train_mask]) ** 2):.6f})'
                for m in methods_to_plot
            ])

            return list(line_preds.values()) + list(line_preds_train.values()) + list(line_preds_test.values()) + \
                list(lines_train_mse.values()) + list(lines_test_mse.values()) + \
                list(line_train_mse_final.values()) + list(line_test_mse_final.values())

        ani = FuncAnimation(fig, update, frames=max_iters, init_func=init, blit=True, interval=interval)

        if save_path:
            ani.save(save_path, writer='pillow')

        plt.show()