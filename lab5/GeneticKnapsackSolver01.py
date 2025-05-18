import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
class GeneticKnapsackSolver01:
    def __init__(self, weights, values, capacity, population_size=100, mutation_rate=0.05, generations=300):
        """
        Initialize the GeneticKnapsackSolver01 class.

        Parameters
        ----------
        weights : list
            List of item weights.
        values : list
            List of item values.
        capacity : int
            Maximum capacity of the knapsack.
        population_size : int, optional
            Size of the population, by default 100.
        mutation_rate : float, optional
            Rate of mutation, by default 0.05.
        generations : int, optional
            Number of generations, by default 300.

        Attributes
        ----------
        weights : list
            List of item weights.
        values : list
            List of item values.
        capacity : int
            Maximum capacity of the knapsack.
        population_size : int
            Size of the population.
        mutation_rate : float
            Rate of mutation.
        generations : int
            Number of generations.
        N : int
            Number of items.
        best : list
            Best individual found.
        best_value : int
            Value of the best individual.
        all_population : list
            Collection of all populations over generations.
        best_values : list
            Record of the best values found in each generation.
        """

        self.weights = weights
        self.values = values
        self.capacity = capacity
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.N = len(weights)

        self.best = []
        self.best_value = 0
        self.all_population = []
        self.best_values = []


    def _fitness(self, individual):
        """
        Calculate the fitness of an individual.

        Parameters
        ----------
        individual : list
            A binary list representing the presence (1) or absence (0) of each item in the knapsack.

        Returns
        -------
        int
            The total value of the selected items if the total weight does not exceed the capacity,
            otherwise 0.
        """

        total_weight = sum(w for w, bit in zip(self.weights, individual) if bit)
        total_value = sum(p for p, bit in zip(self.values, individual) if bit)
        return total_value if total_weight <= self.capacity else 0

    def _selection_tournament(self, population):
        """
        Perform tournament selection on a population.

        This method selects two individuals from the given population using
        tournament selection. For each selection, a random sample of 5 individuals
        is taken, and the individual with the highest fitness is selected as the
        best.

        Parameters
        ----------
        population : list
            The current population from which individuals will be selected.

        Returns
        -------
        list
            A list containing the two selected individuals.
        """
        selected = []
        for _ in range(2):
            tournament = random.sample(population, 5)
            best = max(tournament, key=lambda r: self._fitness(r))
            selected.append(best)
        return selected

    def _crossover(self, parent1, parent2):
        
        """
        Perform a single-point crossover between two parent individuals.

        This method takes two parent individuals and creates a single child
        individual by combining parts of both parents. The crossover point is
        chosen randomly between the first and last elements of the individuals.

        Parameters
        ----------
        parent1 : list
            The first parent individual.
        parent2 : list
            The second parent individual.

        Returns
        -------
        list
            The offspring individual created by crossing over the two parents.
        """

        point = random.randint(1, self.N - 1)
        return parent1[:point] + parent2[point:]

    def _mutate(self, individual):
        """
        Perform a mutation on an individual.

        This method takes an individual and randomly selects positions to
        flip the bit at that position. The probability of flipping a bit is
        given by the mutation rate.

        Parameters
        ----------
        individual : list
            The individual to be mutated.

        Returns
        -------
        list
            The mutated individual.
        """
        return [bit if random.random() > self.mutation_rate else 1 - bit for bit in individual]

    def solve(self):
        """
        Execute the genetic algorithm to solve the knapsack problem.

        This method initializes a population of potential solutions and iteratively
        evolves the population over a number of generations to maximize the fitness
        function, which evaluates the total value of items in the knapsack without
        exceeding its capacity. In each generation, the algorithm selects an elite
        individual, applies tournament selection, crossover, and mutation operations
        to generate new offspring, and updates the population. The best individual
        found during the process is tracked and returned.

        Returns
        -------
        tuple
            A tuple containing the best individual found, its fitness value, a list
            of all populations over the generations, and a list of the best fitness
            values for each generation.
        """

        population = [random.choices([0, 1], k=self.N) for _ in range(self.population_size)]
        best = max(population, key=self._fitness)

        all_population = []
        best_values = []

        for _ in range(self.generations):
            new_population = []

            elite = max(population, key=self._fitness)
            new_population.append(elite)

            for _ in range(self.population_size - 1):
                p1, p2 = self._selection_tournament(population)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_population.append(child)

            population = new_population

            all_population.append(deepcopy(population))
            current_best = max(population, key=self._fitness)
            best_values.append(self._fitness(current_best))

            if self._fitness(current_best) > self._fitness(best):
                best = current_best

        best_value = self._fitness(best)

        self.best = best
        self.best_value = best_value
        self.all_population = all_population
        self.best_values = best_values
        return best, best_value, all_population, best_values

    def plot(self):
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        item_indices = list(range(self.N))
        plt.xticks([i for i in item_indices if i % 2 == 1])  
        # Графік збіжності
        axs[0].plot(self.best_values, color='blue')
        axs[0].set_title("Збіжність генетичного алгоритму")
        axs[0].set_xlabel("Покоління")
        axs[0].set_ylabel("Найкраще значення")
        axs[0].grid(True)

        # Графік вибраних предметів
        item_indices = list(range(self.N))
        selected_items = self.best  # бінарний вектор
        axs[1].bar(item_indices, selected_items, color='green')
        axs[1].set_title(f"Вибрані предмети (Total value = {self.best_value})")
        axs[1].set_xlabel("Номер предмета")
        axs[1].set_ylabel("Вибрано (1 — так, 0 — ні)")
        axs[1].set_ylim(0, 1.2)
        axs[1].grid(axis='y')

        plt.tight_layout()
        plt.show()