import random
import numpy as np
import matplotlib.pyplot as plt
class GeneticTSP:
    def __init__(self, points, pop_size=100, generations=500, mutation_rate=0.1, elitism=True):
        """
        Initialize the GeneticTSP class.

        Parameters
        ----------
        points : list
            A list of (x, y) coordinates of the points in the TSP.
        pop_size : int, optional
            The size of the population. Defaults to 100.
        generations : int, optional
            The number of generations to run the optimization. Defaults to 500.
        mutation_rate : float, optional
            The mutation rate for the genetic algorithm. Defaults to 0.1.
        elitism : bool, optional
            Whether to use elitism in the genetic algorithm. Defaults to True.

        Attributes
        ----------
        points : list
            A list of (x, y) coordinates of the points in the TSP.
        num_points : int
            The number of points in the TSP.
        pop_size : int
            The size of the population.
        generations : int
            The number of generations to run the optimization.
        mutation_rate : float
            The mutation rate for the genetic algorithm.
        elitism : bool
            Whether to use elitism in the genetic algorithm.
        population : list
            The current population of routes.
        best_route : list
            The best route found so far.
        best_length : float
            The length of the best route found so far.
        best_routes : list
            A list of all the best routes found so far.
        best_lengths : list
            A list of the lengths of all the best routes found so far.
        """
        self.points = points
        self.num_points = len(points)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.population = self._create_initial_population()
        self.best_route = None
        self.best_length = float('inf')
        self.best_routes = []  
        self.best_lengths = []

    def _distance(self, p1, p2):
        """
        Calculate the Euclidean distance between two points.

        Parameters
        ----------
        p1 : tuple
            A tuple representing the (x, y) coordinates of the first point.
        p2 : tuple
            A tuple representing the (x, y) coordinates of the second point.

        Returns
        -------
        float
            The Euclidean distance between the two points.
        """

        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _route_length(self, route):
        
        """
        Calculate the length of a route.

        Parameters
        ----------
        route : list
            A list of indices of the points in the route.

        Returns
        -------
        float
            The length of the route.
        """
        return sum(self._distance(self.points[route[i]], self.points[route[(i + 1) % self.num_points]]) 
                   for i in range(self.num_points))

    def _create_initial_population(self):
        
        """
        Create an initial population of random routes.

        Returns
        -------
        list
            A list of lists, where each sublist is a route represented as a list of indices of the points in the route.
        """
        return [random.sample(range(self.num_points), self.num_points) for _ in range(self.pop_size)]

    def _fitness(self, route):
        """
        Calculate the fitness of a route.

        The fitness is the inverse of the length of the route plus a small value to avoid division by zero.

        Parameters
        ----------
        route : list
            A list of indices of the points in the route.

        Returns
        -------
        float
            The fitness of the route.
        """
        return 1 / (self._route_length(route) + 1e-6)

    def _selection(self):
        """
        Select a subset of the population using tournament selection.

        This method performs tournament selection to choose a subset of routes
        from the current population. For each selection, a random sample of routes
        is taken, and the route with the shortest length is selected as the best.

        Returns
        -------
        list
            A list of routes selected from the population.
        """

        selected = []
        for _ in range(self.pop_size):
            tournament = random.sample(self.population, 5)
            best = min(tournament, key=lambda r: self._route_length(r))
            selected.append(best)
        return selected

    def _crossover_parity(self, parent1, parent2):
        """
        Perform a parity-based crossover between two parent routes to produce a child route.

        This method takes two parent routes and combines them by selecting cities from
        the first parent at even indices and filling in the remaining positions with cities
        from the second parent, maintaining their order and ensuring no city is duplicated.

        Parameters
        ----------
        parent1 : list
            The first parent route represented as a list of city indices.
        parent2 : list
            The second parent route represented as a list of city indices.

        Returns
        -------
        list
            A new child route represented as a list of city indices.
        """

        size = len(parent1)
        child = [-1] * size
        used = set()

        for i in range(0, size, 2):
            city = parent1[i]
            child[i] = city
            used.add(city)

        pointer = 1
        for city in parent2:
            if city not in used:
                if pointer < size:
                    child[pointer] = city
                    used.add(city)
                    pointer += 2

        return child

    def _mutate_inversion(self, route):
        """
        Perform an inversion mutation on a route.

        This method takes a route and selects two random indices, reversing the
        subsequence of the route between those indices.

        Parameters
        ----------
        route : list
            The route to be mutated, represented as a list of city indices.

        Returns
        -------
        list
            The mutated route, represented as a list of city indices.
        """
        start, end = sorted(random.sample(range(len(route)), 2))
        route[start:end] = reversed(route[start:end])
        return route

    def _mutate_swap(self, route):
        """
        Perform a swap mutation on a route.

        This method takes a route and performs a series of swaps between randomly
        selected city indices. A random number of swaps is determined, and for each
        swap, two cities are chosen and their positions are exchanged.

        Parameters
        ----------
        route : list
            The route to be mutated, represented as a list of city indices.

        Returns
        -------
        list
            The mutated route, represented as a list of city indices.
        """

        num_swaps = random.randint(2, len(route) // 2)
        indices = random.sample(range(len(route)), num_swaps)
        random.shuffle(indices)
        for i in range(0, len(indices) - 1, 2):
            route[indices[i]], route[indices[i + 1]] = route[indices[i + 1]], route[indices[i]]
        return route

    def _mutate(self, route):
        """
        Perform a mutation on a route.

        This method takes a route and randomly selects between two different mutation
        methods. With a probability of 0.5, it will use the inversion mutation method;
        otherwise, it will use the swap mutation method.

        Parameters
        ----------
        route : list
            The route to be mutated, represented as a list of city indices.

        Returns
        -------
        list
            The mutated route, represented as a list of city indices.
        """
        if random.random() < 0.5:
            return self._mutate_inversion(route)
        else:
            return self._mutate_swap(route)

    def optimize(self):
        """
        Run the genetic algorithm optimization.

        This method runs the genetic algorithm optimization for the specified number
        of generations. At each generation, it selects a new population using the
        `_selection` method, applies crossover and mutation to the selected individuals
        to generate a new population of offspring, and then replaces the existing
        population with the new offspring. It keeps track of the best route found so far
        and the lengths of the best routes found so far.

        Returns
        -------
        tuple
            A tuple containing the best route found, the length of the best route found,
            a list of all the best routes found, and a list of the lengths of all the
            best routes found.
        """
        for generation in range(self.generations):
            new_population = self._selection()
            offspring = []

            if self.elitism:
                elite = min(self.population, key=lambda r: self._route_length(r))
                offspring.append(elite)

            for _ in range(self.pop_size - len(offspring)):
                parent1, parent2 = random.sample(new_population, 2)
                child = self._crossover_parity(parent1, parent2)

                if random.random() < self.mutation_rate:
                    child = self._mutate(child)

                offspring.append(child)

            self.population = offspring

            current_best = min(self.population, key=lambda r: self._route_length(r))
            current_length = self._route_length(current_best)

            self.best_lengths.append(self.best_length)
            self.best_routes.append(current_best)

            if current_length < self.best_length:
                self.best_route, self.best_length = current_best, current_length

            print(f"Generation {generation+1}: Best Length = {self.best_length}")

        return self.best_route, self.best_length, self.best_routes, self.best_lengths

    def plot_route(self, route):
        ordered_points = [self.points[i] for i in route] + [self.points[route[0]]]
        x, y = zip(*ordered_points)
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.plot(x, y, marker='o', linestyle='-', color='blue')
        ax1.set_title("Optimal Route")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.grid(True)

        ax2.plot(self.best_lengths)
        ax2.set_title("Best Length Over Generations")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Best Length")
        ax2.grid(True)

        plt.show()