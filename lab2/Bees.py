import numpy as np
class Bee:
    def __init__(self, dim, pos_bounds):
        self.dim = dim
        self.pos_bounds = np.array(pos_bounds)
        # Ініціалізація випадкової позиції бджоли у межах простору
        self.position = np.random.uniform(self.pos_bounds[:, 0], self.pos_bounds[:, 1])
        self.score = float('inf')  # Значення цільової функції

class BeesAlgorithm:
    def __init__(self, fitness_func, num_bees, dim, pos_bounds, max_iter,
                 elite_sites, best_sites, elite_bees, recruited_bees,
                 delta_init=0.1, alpha=0.95, eta_max=0.9,
                 target_score=None, epsilon_target=None):
        
        self.fitness_func = fitness_func
        # K — розмір популяції
        self.num_bees = num_bees
        # M — розмірність простору пошуку
        self.dim = dim
        # Межі простору (мінімальні та максимальні координати для кожної змінної)
        if np.array(pos_bounds).ndim == 1:
            self.pos_bounds = np.tile(np.array(pos_bounds), (dim, 1))
        else:
            self.pos_bounds = np.array(pos_bounds)
        # N — максимальна кількість ітерацій
        self.max_iter = max_iter

        # L_es — кількість елітних ділянок
        self.elite_sites = elite_sites
        # L_s — кількість ділянок для пошуку
        self.best_sites = best_sites
        # Z_e — кількість бджіл на елітній ділянці
        self.elite_bees = elite_bees
        # Z_0 — кількість бджіл на звичайній ділянці
        self.recruited_bees = recruited_bees

        # Початковий розмір околу (діапазон локального пошуку)
        self.delta = np.abs(self.pos_bounds[:, 1] - self.pos_bounds[:, 0]) * delta_init
        # alpha — коефіцієнт зменшення околу після кожної ітерації
        self.alpha = alpha
        # eta_max — коефіцієнт стохастичного шуму
        self.eta_max = eta_max

        self.target_score = target_score
        self.epsilon_target = epsilon_target


        # Ініціалізація популяції
        self.bees = [Bee(dim, pos_bounds) for _ in range(num_bees)]
        # Краща знайдена позиція та її значення
        self.best_position = None
        self.best_score = float('inf')

        # Історія найкращих рішень та всіх позицій
        self.history_best = []
        self.history_all_positions = []

    def _local_search(self, site_position, delta, num_bees):
        candidates = []
        for _ in range(num_bees):
            offset = np.random.uniform(-delta * self.eta_max, delta * self.eta_max)
            candidate = np.clip(site_position + offset, self.pos_bounds[:, 0], self.pos_bounds[:, 1])
            candidates.append(candidate)
        return candidates
    
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
        # Початкова оцінка всіх бджіл
        for bee in self.bees:
            bee.score = self.fitness_func(bee.position)

        for iteration in range(self.max_iter):
            # Сортування бджіл за якістю (зростання score)
            self.bees.sort(key=lambda b: b.score)

            new_positions = []
            new_scores = []

            for i, site in enumerate(self.bees[:self.best_sites]):
                # Визначення кількості бджіл для локального пошуку
                n_recruited = self.elite_bees if i < self.elite_sites else self.recruited_bees
                candidates = self._local_search(site.position, self.delta, n_recruited)

                # Вибір кращого кандидата серед локального пошуку
                candidates = np.array(candidates)
                scores = np.array([self.fitness_func(candidate) for candidate in candidates])
                best_idx = np.argmin(scores)

                best_candidate = candidates[best_idx]
                best_candidate_score = scores[best_idx]

                new_positions.append(best_candidate)
                new_scores.append(best_candidate_score)

            # Додавання розвідників (рандомні позиції)
            for _ in range(self.num_bees - len(new_positions)):
                random_position = np.random.uniform(self.pos_bounds[:, 0], self.pos_bounds[:, 1])
                random_score = self.fitness_func(random_position)
                new_positions.append(random_position)
                new_scores.append(random_score)

            # Оновлення популяції
            for bee, pos, score in zip(self.bees, new_positions, new_scores):
                bee.position = pos
                bee.score = score

            # Оновлення найкращої бджоли
            best_bee = min(self.bees, key=lambda b: b.score)
            if best_bee.score < self.best_score:
                self.best_score = best_bee.score
                self.best_position = best_bee.position.copy()

            # Зменшення розміру околу
            self.delta *= self.alpha

            # Збереження історії
            self.history_best.append((self.best_position.copy(), self.best_score))
            self.history_all_positions.append([bee.position.copy() for bee in self.bees])

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Global best score = {self.best_score}")

            if self._check_termination(iteration):
                break

        return self.best_position, self.best_score, self.history_best, self.history_all_positions
