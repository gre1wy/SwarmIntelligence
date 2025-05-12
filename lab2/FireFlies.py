from scipy.stats import qmc
import numpy as np
class Firefly:
    def __init__(self, dim, pos_bounds):
        self.dim = dim
        self.pos_bounds = np.array(pos_bounds)
        # Ініціалізація випадкової позиції світлячка у межах простору
        self.position = np.random.uniform(self.pos_bounds[:, 0], self.pos_bounds[:, 1])
        self.score = float('inf') 

class FireflyAlgorithm:
    def __init__(self, fitness_func, num_fireflies, dim, pos_bounds, max_iter, 
                 alpha=0.2, beta0=1.0, gamma=1.0,
                 target_score=None, epsilon_target=None,
                 epsilon_stagnation=1e-6, stagnation_iter=20):
        self.fitness_func = fitness_func
        # K — кількість світлячків 
        self.num_fireflies = num_fireflies
        # M — розмірність простору
        self.dim = dim
        # Ініціалізація меж пошуку (мінімальні і максимальні значення для кожної змінної)
        if np.array(pos_bounds).ndim == 1:
            self.pos_bounds = np.tile(np.array(pos_bounds), (dim, 1))
        else:
            self.pos_bounds = np.array(pos_bounds)
        # N — максимальна кількість ітерацій
        self.max_iter = max_iter

        self.alpha = alpha  # стохастичний коефіцієнт випадкового руху
        self.beta0 = beta0  # максимальна яскравість
        self.gamma = gamma  # коефіцієнт поглинання світла

        self.target_score = target_score
        self.epsilon_target = epsilon_target

        self.stagnation_iter = stagnation_iter
        self.epsilon_stagnation = epsilon_stagnation    

        # Ініціалізація популяції світлячків
        positions = self._latin_hypercube_init(num_fireflies, dim, pos_bounds)
        self.fireflies = []
        for pos in positions:
            firefly = Firefly(dim, pos_bounds)
            firefly.position = pos
            self.fireflies.append(firefly)
        self.best_position = None
        self.best_score = float('inf')

        self.history_best = []
        self.history_all_positions = []

    def _latin_hypercube_init(self, num_points, dim, bounds):
        sampler = qmc.LatinHypercube(d=dim)
        sample = sampler.random(n=num_points)
        lower_bounds = np.array(bounds)[:, 0]
        upper_bounds = np.array(bounds)[:, 1]
        scaled_sample = qmc.scale(sample, lower_bounds, upper_bounds)
        return scaled_sample


    def _attractiveness(self, distance):
        # Розрахунок привабливості залежно від відстані між світлячками
        return self.beta0 * np.exp(-self.gamma * distance**2)
    
    def _check_termination(self, iteration):
        if self.target_score is not None:
            if self.best_score <= self.target_score:
                print(f"Target global minimum {self.target_score} reached at iteration {iteration}. Stopping.")
                return True
            if self.epsilon_target is not None and abs(self.best_score - self.target_score) < self.epsilon_target:
                print(f"Global minimum is within epsilon {self.epsilon_target} of target at iteration {iteration}. Stopping.")
                return True
        return False
    
    def run(self):
        stagnation_counter = 0
        previous_best = float('inf')

        for iteration in range(self.max_iter):
            for firefly in self.fireflies:
                firefly.score = self.fitness_func(firefly.position)

            for i in range(self.num_fireflies):
                for j in range(self.num_fireflies):
                    if self.fireflies[j].score < self.fireflies[i].score:
                        distance = np.linalg.norm(self.fireflies[i].position - self.fireflies[j].position)
                        beta = self._attractiveness(distance)
                        rand = np.random.uniform(-0.5, 0.5, self.dim)
                        self.fireflies[i].position += beta * (self.fireflies[j].position - self.fireflies[i].position) + self.alpha * rand
                        self.fireflies[i].position = np.clip(self.fireflies[i].position, self.pos_bounds[:, 0], self.pos_bounds[:, 1])

            self.alpha *= 0.97
            self.gamma *= 1.02

            best_firefly = min(self.fireflies, key=lambda f: f.score)
            if best_firefly.score < self.best_score:
                self.best_score = best_firefly.score
                self.best_position = best_firefly.position.copy()

            self.history_best.append((self.best_position.copy(), self.best_score))
            self.history_all_positions.append([f.position.copy() for f in self.fireflies])

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Global best score = {self.best_score}")

            if self._check_termination(iteration):
                break

            # Перевірка стагнації
            if abs(previous_best - self.best_score) < self.epsilon_stagnation:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                previous_best = self.best_score

            if stagnation_counter >= self.stagnation_iter:
                print(f"Stagnation detected. Early stopping at iteration {iteration}.")
                break

        return self.best_position, self.best_score, self.history_best, self.history_all_positions
