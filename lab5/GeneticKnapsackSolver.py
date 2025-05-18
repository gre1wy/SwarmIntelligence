import random
from copy import deepcopy
import matplotlib.pyplot as plt
class GeneticKnapsackSolver:
    def __init__(self, weights, values, max_counts, capacity, population_size=100, mutation_rate=0.05, generations=300):
        """
        Constructor for GeneticKnapsackSolver

        Parameters
        ----------
        weights : list
            list of weights of items
        values : list
            list of values of items
        max_counts : list
            list of maximum counts of each item
        capacity : int
            maximum capacity of the knapsack
        population_size : int, optional
            size of the population, by default 100
        mutation_rate : float, optional
            rate of mutation, by default 0.05
        generations : int, optional
            number of generations, by default 300

        Attributes
        ----------
        weights : list
            list of weights of items
        values : list
            list of values of items
        max_counts : list
            list of maximum counts of each item
        capacity : int
            maximum capacity of the knapsack
        population_size : int
            size of the population
        mutation_rate : float
            rate of mutation
        generations : int
            number of generations
        N : int
            number of items
        best : list
            the best individual
        best_value : int
            the best value
        all_population : list
            list of all populations
        best_values : list
            list of best values
        """
        self.weights = weights
        self.values = values
        self.max_counts = max_counts
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
        total_weight = sum(w * count for w, count in zip(self.weights, individual))
        total_value = sum(v * count for v, count in zip(self.values, individual))
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
        Perform crossover between two parent individuals.

        This method takes two parent individuals and performs single-point
        crossover between them. The crossover point is chosen randomly between
        the first and last elements of the individuals (inclusive).

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
        Perform mutation on an individual.

        This method takes an individual and mutates its genes with a
        probability given by the mutation rate. The mutation consists of
        randomly selecting a new value for the gene from the range given by
        the maximum count for that gene.

        Parameters
        ----------
        individual : list
            The individual to be mutated.

        Returns
        -------
        list
            The mutated individual.
        """
        mutant = individual.copy()
        for i in range(self.N):
            if random.random() < self.mutation_rate:
                mutant[i] = random.randint(0, self.max_counts[i])
        return mutant

    def solve(self):
        """
        Run the genetic algorithm optimization.

        This method runs the genetic algorithm optimization for the specified
        number of generations. At each generation, it selects a new population
        using the `_selection_tournament` method, applies crossover and mutation
        to the selected individuals to generate a new population of offspring,
        and then replaces the existing population with the new offspring. It
        keeps track of the best route found so far and the lengths of the best
        routes found so far.

        Returns
        -------
        tuple
            A tuple containing the best route found, the length of the best route
            found, a list of all the best routes found, and a list of the lengths
            of all the best routes found.
        """
        population = [
            [random.randint(0, self.max_counts[i]) for i in range(self.N)]
            for _ in range(self.population_size)
        ]
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
        plt.xticks([i for i in item_indices if i % 2 == 0]) 
        # Графік збіжності
        axs[0].plot(self.best_values, color='blue')
        axs[0].set_title("Збіжність генетичного алгоритму")
        axs[0].set_xlabel("Покоління")
        axs[0].set_ylabel("Найкраще значення")
        axs[0].grid(True)

        # Графік вибраних предметів (кількість кожного)
        selected_counts = self.best  
        axs[1].bar(item_indices, selected_counts, color='green')
        axs[1].set_title(f"Вибрані предмети (Total value = {self.best_value})")
        axs[1].set_xlabel("Номер предмета")
        axs[1].set_ylabel("Кількість предметів")
        axs[1].set_ylim(0, max(self.max_counts) + 1)
        axs[1].grid(axis='y')

        plt.tight_layout()
        plt.show()