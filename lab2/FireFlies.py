from scipy.stats import qmc
import numpy as np
class Firefly:
    def __init__(self, dim, pos_bounds):
        self.dim = dim
        self.pos_bounds = np.array(pos_bounds)
        # Ініціалізація випадкової позиції світлячка у межах простору (пункт 1.5 методички)
        self.position = np.random.uniform(self.pos_bounds[:, 0], self.pos_bounds[:, 1])
        self.score = float('inf')  # Значення цільової функції

class FireflyAlgorithm:
    def __init__(self, fitness_func, num_fireflies, dim, pos_bounds, max_iter, 
                 alpha=0.2, beta0=1.0, gamma=1.0,
                 target_score=None, epsilon_target=None):
        self.fitness_func = fitness_func
        # K — кількість світлячків (популяція)
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

        # Параметри алгоритму (пункт 1.1 методички)
        self.alpha = alpha  # стохастичний коефіцієнт випадкового руху
        self.beta0 = beta0  # максимальна яскравість
        self.gamma = gamma  # коефіцієнт поглинання світла

        self.target_score = target_score
        self.epsilon_target = epsilon_target

        # Ініціалізація популяції світлячків (пункт 1.5)
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
        # Розрахунок привабливості залежно від відстані між світлячками (пункт 5.1)
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
        for iteration in range(self.max_iter):  # Початок ітерації n (пункт 2)
            for firefly in self.fireflies:
                firefly.score = self.fitness_func(firefly.position)  # Оцінка значення цільової функції (пункт 5)

            for i in range(self.num_fireflies):  # Вибір світлячка k (пункт 3)
                for j in range(self.num_fireflies):  # Вибір світлячка l (пункт 4)
                    if self.fireflies[j].score < self.fireflies[i].score:
                        distance = np.linalg.norm(self.fireflies[i].position - self.fireflies[j].position)  # Розрахунок відстані d_kl
                        beta = self._attractiveness(distance)  # Розрахунок яскравості beta (пункт 5.1)
                        rand = np.random.uniform(-0.5, 0.5, self.dim)
                        # Формула оновлення позиції світлячка k за впливом l (пункт 5.2)
                        self.fireflies[i].position += beta * (self.fireflies[j].position - self.fireflies[i].position) + self.alpha * rand
                        # Обмеження позиції в межах простору (пункт 5.3)
                        self.fireflies[i].position = np.clip(self.fireflies[i].position, self.pos_bounds[:, 0], self.pos_bounds[:, 1])

            # Адаптивне зменшення alpha і збільшення gamma
            self.alpha *= 0.97
            self.gamma *= 1.02

            # Визначення найкращого світлячка (пункт 7)
            best_firefly = min(self.fireflies, key=lambda f: f.score)
            if best_firefly.score < self.best_score:
                self.best_score = best_firefly.score
                self.best_position = best_firefly.position.copy()  # Оновлення глобального найкращого рішення (пункт 8)

            # Збереження історії найкращого результату та всіх позицій
            self.history_best.append((self.best_position.copy(), self.best_score))
            self.history_all_positions.append([f.position.copy() for f in self.fireflies])

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Global best score = {self.best_score}")
            
            if self._check_termination(iteration):
                break

        # Завершення після N ітерацій (пункт 9)
        return self.best_position, self.best_score, self.history_best, self.history_all_positions