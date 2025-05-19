import numpy as np
class PSO_algorithm:
    def __init__(self, loss_func, bounds, num_particles=100, w=0.5, c1=1.5, c2=1.5, max_iter=100):
        """
        Parameters
        ----------
        loss_func : callable
            function to optimize
        bounds : list of tuples
            bounds for each dimension, e.g. [(0, 10), (0, 10)]
        num_particles : int, optional
            number of particles in the swarm (default is 100)
        w : float, optional
            inertia weight (default is 0.5)
        c1 : float, optional
            cognitive weight (default is 1.5)
        c2 : float, optional
            social weight (default is 1.5)
        max_iter : int, optional
            maximum number of iterations (default is 100)
        """
        
        self.loss_func = loss_func
        self.bounds = np.array(bounds)
        self.num_particles = num_particles
        self.w = w  # inertia
        self.c1 = c1  # cognitive
        self.c2 = c2  # social
        self.max_iter = max_iter
        self.dim = self.bounds.shape[0]

    def optimize(self):
        """
        Optimize the loss function using Particle Swarm Optimization.

        Returns
        -------
        best_individual : array-like
            the best individual found
        best_fitness : float
            the best fitness found
        best_individuals : list of arrays
            a list of the best individuals found at each iteration
        loss_history : list of floats
            a list of the best fitnesses found at each iteration
        """
        X = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.num_particles, self.dim))
        V = np.zeros_like(X)
        personal_best = X.copy()
        personal_best_fitness = np.array([self.loss_func(x) for x in X])
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]

        best_individuals = [global_best.copy()]
        loss_history = [global_best_fitness]

        for _ in range(self.max_iter):
            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                V[i] = (
                    self.w * V[i]
                    + self.c1 * r1 * (personal_best[i] - X[i])
                    + self.c2 * r2 * (global_best - X[i])
                )
                X[i] = np.clip(X[i] + V[i], self.bounds[:, 0], self.bounds[:, 1])

                f = self.loss_func(X[i])
                if f < personal_best_fitness[i]:
                    personal_best[i] = X[i]
                    personal_best_fitness[i] = f
                    if f < global_best_fitness:
                        global_best = X[i].copy()
                        global_best_fitness = f

            best_individuals.append(global_best.copy())
            loss_history.append(global_best_fitness)

        return global_best, global_best_fitness, best_individuals, loss_history