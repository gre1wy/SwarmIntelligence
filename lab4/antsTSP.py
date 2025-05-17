import numpy as np
import matplotlib.pyplot as plt
import random
class AntColonyTSP:
    def __init__(self, points, num_ants=20, generations=100, alpha=1.0, beta=2.0, evaporation=0.5, Q=100):
        """
        Initialize the AntColonyTSP class.

        Parameters
        ----------
        points : list
            A list of (x, y) coordinates of the points in the TSP.
        num_ants : int, optional
            The number of ants to use in optimization. Defaults to 20.
        generations : int, optional
            The number of generations to run the optimization. Defaults to 100.
        alpha : float, optional
            The alpha parameter for the ant colony optimization. Defaults to 1.0.
        beta : float, optional
            The beta parameter for the ant colony optimization. Defaults to 5.0.
        evaporation : float, optional
            The rate at which pheromones evaporate. Defaults to 0.5.
        Q : float, optional
            The constant used for calculating the pheromone strength. Defaults to 100.

        Attributes
        ----------
        points : list
            A list of (x, y) coordinates of the points in the TSP.
        num_points : int
            The number of points in the TSP.
        num_ants : int
            The number of ants to use in optimization.
        generations : int
            The number of generations to run the optimization.
        alpha : float
            The alpha parameter for the ant colony optimization.
            Controls the influence of pheromone concentration on the decision-making process.
        beta : float
            The beta parameter for the ant colony optimization.
            Controls the influence of heuristic information (distance) on the decision-making process.
        evaporation : float
            The rate at which pheromones evaporate.
        Q : float
            The constant used for calculating the pheromone strength.
        distances : numpy array
            A matrix of distances between all points in the TSP.
        pheromones : numpy array
            A matrix of pheromone strengths between all points in the TSP.
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
        self.num_ants = num_ants
        self.generations = generations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = Q

        self.distances = self._calculate_distances()
        self.pheromones = np.ones((self.num_points, self.num_points))
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

    def _calculate_distances(self):
        """
        Calculate the matrix of distances between all points in the TSP.

        Returns
        -------
        numpy array
            A matrix of distances between all points in the TSP.
        """

        distances = np.zeros((self.num_points, self.num_points))
        for i in range(self.num_points):
            for j in range(i, self.num_points):
                dist = self._distance(self.points[i], self.points[j])
                distances[i][j] = dist
                distances[j][i] = dist
        return distances

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
        return sum(self.distances[route[i], route[(i + 1) % self.num_points]] for i in range(self.num_points))

    def _probability(self, current_city, visited):
        """
        Calculate the probability of moving to each unvisited city from the current city.

        Parameters
        ----------
        current_city : int
            The index of the current city.
        visited : set
            A set of indices representing cities that have already been visited.

        Returns
        -------
        list of tuple
            A list of tuples where each tuple contains the index of an unvisited city
            and the normalized probability of moving to that city.
        """

        total = 0
        probabilities = []

        # Розрахунок ймовірності для кожного міста
        for j in range(self.num_points):
            if j not in visited:
                pheromone = self.pheromones[current_city][j] ** self.alpha
                heuristic = (1 / self.distances[current_city][j]) ** self.beta
                prob = pheromone * heuristic
                total += prob
                probabilities.append((j, prob))

        # Нормалізація ймовірностей
        normalized = [(city, prob / total) for city, prob in probabilities]
        return normalized

    def _select_next_city(self, current_city, visited):
        """
        Select the next city in the route based on probabilities.

        Parameters
        ----------
        current_city : int
            The index of the current city.
        visited : set
            A set of indices representing cities that have already been visited.

        Returns
        -------
        int
            The index of the next city in the route.
        """

        probabilities = self._probability(current_city, visited)

        # Вибір наступного міста на основі ймовірностей
        cities, probs = zip(*probabilities)
        next_city = random.choices(cities, weights=probs)[0]
        return next_city


    def _construct_route(self):
        """
        Construct a route by selecting the next city based on probabilities.

        Parameters
        ----------
        None

        Returns
        -------
        list
            A list of indices representing the route.
        """

        start = random.randint(0, self.num_points - 1)
        route = [start]
        visited = {start}

        for _ in range(self.num_points - 1):
            current_city = route[-1]
            next_city = self._select_next_city(current_city, visited)
            route.append(next_city)
            visited.add(next_city)

        return route

    def _update_pheromones(self, all_routes):
        """
        Update the pheromones based on all routes found so far.

        Parameters
        ----------
        all_routes : list of tuple
            A list of tuples where each tuple contains a route and its length.

        Returns
        -------
        None

        """
        # Випаровування феромонів
        self.pheromones *= (1 - self.evaporation)

        # Оновлення феромонів на основі всіх маршрутів
        for route, length in all_routes:
            delta_pheromone = self.Q / length
            for i in range(self.num_points):
                j = (i + 1) % self.num_points
                self.pheromones[route[i]][route[j]] += delta_pheromone
                self.pheromones[route[j]][route[i]] += delta_pheromone

    def optimize(self):
        """
        Run the ant colony optimization algorithm.

        This method performs ant colony optimization over a specified number of generations.
        Each generation, a set number of ants construct routes through the graph of points.
        Pheromone levels are updated based on the quality of the routes found, which guides
        the ants in future generations.

        The function keeps track of the best route and its length found over all generations.

        Returns
        -------
        tuple
            A tuple containing the best route found, the length of the best route, a list
            of the best routes found in each generation, and a list of the lengths of these
            best routes.
        """

        for generation in range(self.generations):
            all_routes = []

            for _ in range(self.num_ants):
                route = self._construct_route()
                length = self._route_length(route)
                all_routes.append((route, length))

                if length < self.best_length:
                    self.best_route = route
                    self.best_length = length

            self._update_pheromones(all_routes)
            
            self.best_lengths.append(self.best_length)
            self.best_routes.append(self.best_route)

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