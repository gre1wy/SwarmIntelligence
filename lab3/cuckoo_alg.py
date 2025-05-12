import numpy as np
# nest = [x_1, x_2, ..., x_n]
# egg = func(nest)
class Cuckoo:
    def __init__(self, func, bounds, max_iter=1000, 
                 n_nests=100, p_detect=0.9,
                 delta_min=0.1, delta_max=0.9,
                 eps=1e-6, target=None):
        self.func = func
        self.bounds = np.array(bounds)
        self.n_nests = n_nests
        self.p_detect = p_detect
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.max_iter = max_iter
        self.eps = eps
        self.target = target

    def simple_bounds(self, s):
        return np.clip(s, self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        M = self.bounds.shape[0]
        range_width = self.bounds[:, 1] - self.bounds[:, 0]

        # Ініціалізація популяції
        nests = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_nests, M))
        eggs = np.apply_along_axis(self.func, 1, nests)

        # Ініціалізація кращого рішення
        best_nest_idx = np.argmin(eggs)
        best_egg = eggs[best_nest_idx]
        best_nest = nests[best_nest_idx].copy()

        # Історія
        best_history = []
        all_history = []

        for n in range(1, self.max_iter + 1):
            delta = self.delta_max - (self.delta_max - self.delta_min) * (n / self.max_iter)

            # 3. Вибір випадкового гнізда
            random_nest_idx = np.random.randint(0, self.n_nests) 
            random_nest = nests[random_nest_idx]

            # 4. Генерація нового рішення
            perturbation = delta * range_width * np.random.uniform(-1, 1, M)
            new_nest = self.simple_bounds(random_nest + perturbation)
            new_nest_egg = self.func(new_nest)

            # 5. Оновлення рішення
            if new_nest_egg < eggs[random_nest_idx]:
                nests[random_nest_idx] = new_nest
                eggs[random_nest_idx] = new_nest_egg
                if new_nest_egg < best_egg:
                    best_nest = new_nest.copy()
                    best_egg = new_nest_egg

            # 6. Якщо яйце знайдене
            if np.random.rand() >= self.p_detect:
                worst_idx = np.argmax(eggs)
                worst_nest = nests[worst_idx]

                perturbation = delta * range_width * np.random.uniform(-1, 1, M)
                new_worst_nest = self.simple_bounds(worst_nest + perturbation)

                nests[worst_idx] = new_worst_nest
                eggs[worst_idx] = self.func(new_worst_nest)

            # 7. Збереження історії
            best_history.append((best_nest.copy(), best_egg))

            # Збереження всіх особин у форматі [(позиція1, значення1), (позиція2, значення2), ...]
            iteration_history = [(nests[i].copy(), eggs[i]) for i in range(self.n_nests)]
            all_history.append(iteration_history)
            
            if n % 100 == 0:
                print(f"Cuckoo Algorithm: Iteration {n}, Best fitness = {best_egg}")

            if self.target is not None and np.abs(best_egg - self.target) < self.eps:
                print(f"Early stopping: target reached with best fitness = {best_egg}")
                break

        return best_nest, best_egg, best_history, all_history
