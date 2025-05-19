import numpy as np
class DE_algorithm:
    def __init__(self, loss_func, bounds, pop_size=100, F=0.5, CR=0.7, max_iter=100):
        """
        Initialize the Differential Evolution algorithm.

        Parameters
        ----------
        loss_func : callable
            The loss function to be minimized.
        bounds : list of tuples
            The bounds for each dimension of the optimization problem.
        pop_size : int, optional (default=100)
            The size of the population.
        F : float, optional (default=0.5)
            The mutation factor.
        CR : float, optional (default=0.7)
            The crossover rate.
        max_iter : int, optional (default=100)
            The maximum number of generations to run the optimization for.

        Returns
        -------
        None
        """
        self.loss_func = loss_func
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.F = F  
        self.CR = CR  
        self.max_iter = max_iter
        self.dim = self.bounds.shape[0]
    

    def optimize(self):
        """
        Perform optimization using the Differential Evolution algorithm.

        This method initializes a population of candidate solutions and iteratively
        improves them through mutation, crossover, and selection processes. The goal
        is to minimize the loss function provided during initialization.

        Returns
        -------
        tuple
            A tuple containing:
            - The best solution found.
            - The fitness value of the best solution.
            - A list of the best solutions found at each iteration.
            - A list of the fitness values of the best solutions found at each iteration.
        """

        pop = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.pop_size, self.dim))
        fitness = np.array([self.loss_func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx].copy()

        best_individuals = [best.copy()]
        loss_history = [fitness[best_idx]]

        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                # mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.bounds[:, 0], self.bounds[:, 1])

                # crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # selection
                trial_fitness = self.loss_func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best = trial.copy()

            best_individuals.append(best.copy()) 
            loss_history.append(fitness[best_idx]) 


        return best, fitness[best_idx], best_individuals, loss_history