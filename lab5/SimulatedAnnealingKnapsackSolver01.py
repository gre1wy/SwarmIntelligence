import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
class SimulatedAnnealingKnapsackSolver01:
    def __init__(self, weights, values, capacity, t_max=100, t_min=0.1, alpha=0.95, iterations_per_temp=100):
        """
        Constructor for SimulatedAnnealingKnapsackSolver01

        Parameters
        ----------
        weights : list
            list of weights of items
        values : list
            list of values of items
        capacity : int
            maximum capacity of the knapsack
        t_max : int, optional
            initial temperature, by default 100
        t_min : float, optional
            minimal temperature, by default 0.1
        alpha : float, optional
            rate of cooling, by default 0.95
        iterations_per_temp : int, optional
            number of iterations per temperature, by default 100

        Attributes
        ----------
        weights : list
            list of weights of items
        values : list
            list of values of items
        capacity : int
            maximum capacity of the knapsack
        n : int
            number of items
        t_max : int
            initial temperature
        t_min : float
            minimal temperature
        alpha : float
            rate of cooling
        iterations_per_temp : int
            number of iterations per temperature
        best_values : list
            list of best values
        best : list
            the best individual
        best_fitness : int
            the best value
        """
        self.weights = weights
        self.values = values
        self.capacity = capacity
        self.n = len(weights)
        self.t_max = t_max
        self.t_min = t_min
        self.alpha = alpha
        self.iterations_per_temp = iterations_per_temp

        self.best_values = []
        self.best = []
        self.best_fitness = 0
         

    def fitness(self, individual):
        """
        Calculate the fitness of an individual

        Parameters
        ----------
        individual : list
            individual

        Returns
        -------
        int
            fitness of the individual
        """
        total_weight = sum(w for w, bit in zip(self.weights, individual) if bit)
        total_value = sum(p for p, bit in zip(self.values, individual) if bit)
        return total_value if total_weight <= self.capacity else 0

    def neighbor(self, individual: list):
        """
        Create a neighbor of the given individual.

        A neighbor is an individual which is different from the given one in one position.
        The position is randomly chosen and the value is flipped.

        Parameters
        ----------
        individual : list
            individual

        Returns
        -------
        list
            neighbor of the individual
        """
        neighbor = individual.copy()
        i = random.randint(0, self.n - 1)
        neighbor[i] = 1 - neighbor[i]
        return neighbor

    def solve(self):
        """
        Solve the knapsack problem using simulated annealing.

        This method initializes a random solution and iteratively improves it
        using the simulated annealing algorithm. At each temperature level, 
        a neighbor solution is generated and its fitness is evaluated. If the 
        new solution is better, it is accepted. Otherwise, it is accepted with 
        a probability that depends on the difference in fitness and the current 
        temperature. The temperature is reduced by a factor of alpha after each 
        iteration. The process continues until the temperature falls below the 
        minimum threshold.

        Returns
        -------
        list
            The best solution found and its corresponding fitness value.
        """

        current = [random.choice([0, 1]) for _ in range(self.n)] 
        self.best = current.copy()
        current_fitness = self.fitness(current)
        self.best_fitness = current_fitness
        t = self.t_max

        while t > self.t_min:
            for _ in range(self.iterations_per_temp):
                candidate = self.neighbor(current)
                candidate_fitness = self.fitness(candidate)
                delta_fitness = candidate_fitness - current_fitness

                if delta_fitness > 0:
                    current = candidate
                    current_fitness = candidate_fitness
                else:
                    prob = np.exp(delta_fitness / t)  
                    if random.random() < prob:
                        current = candidate
                        current_fitness = candidate_fitness

                if current_fitness > self.best_fitness:
                    self.best = current.copy()
                    self.best_fitness = current_fitness

                self.best_values.append(self.best_fitness)

            t *= self.alpha
        return self.best, self.best_fitness
    
    def plot(self):
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # Графік збіжності
        axs[0].plot(self.best_values, color='red')
        axs[0].set_title("Збіжність алгоритму імітації відпалу")
        axs[0].set_xlabel("Ітерація")
        axs[0].set_ylabel("Найкраща цінність")
        axs[0].grid(True)

        # Вибрані предмети
        item_indices = list(range(self.n))
        axs[1].bar(item_indices, self.best, color='orange')
        axs[1].set_xticks([i for i in item_indices if i % 2 == 1])  # непарні індекси
        axs[1].set_title(f"Вибрані предмети (Total value = {self.best_fitness})")
        axs[1].set_xlabel("Номер предмета")
        axs[1].set_ylabel("Вибрано (1 — так, 0 — ні)")
        axs[1].set_ylim(0, 1.2)
        axs[1].grid(axis='y')

        plt.tight_layout()
        plt.show()
