import numpy as np
class Cuckoo_Improved:
    def __init__(self, func, bounds, n_nests=50, p_detect=0.25,
                  delta_min=0.1, delta_max=0.9, max_iter=1000, mass_update_fraction=0.1, 
                  eps=1e-6, target=None):
        self.func = func
        self.bounds = np.array(bounds)
        self.n_nests = n_nests
        self.p_detect = p_detect
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.max_iter = max_iter
        self.mass_update_fraction = mass_update_fraction  # скільки гнізд оновлювати масово
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

            # Modified Cuckoo Search: Оновлюємо кілька випадкових гнізд
            adaptive_mass_update = self.mass_update_fraction * (1 - n / self.max_iter)
            num_updates = max(1, int(adaptive_mass_update * self.n_nests))

            random_indices = np.random.choice(self.n_nests, size=num_updates, replace=False)

            for idx in random_indices:
                perturbation = delta * range_width * np.random.uniform(-1, 1, M)
                new_nest = self.simple_bounds(nests[idx] + perturbation)
                new_nest_egg = self.func(new_nest)

                if new_nest_egg < eggs[idx]:
                    nests[idx] = new_nest
                    eggs[idx] = new_nest_egg
                    if new_nest_egg < best_egg:
                        best_nest = new_nest.copy()
                        best_egg = new_nest_egg

            # Improved Cuckoo Search: Локальний пошук навколо найкращого гнізда
            local_perturbation = 0.1 * range_width * np.random.uniform(-1, 1, M)
            new_best_candidate = self.simple_bounds(best_nest + local_perturbation)
            new_best_candidate_egg = self.func(new_best_candidate)

            if new_best_candidate_egg < best_egg:
                best_nest = new_best_candidate.copy()
                best_egg = new_best_candidate_egg
                best_nest_idx = np.argmin(eggs)
                nests[best_nest_idx] = best_nest
                eggs[best_nest_idx] = best_egg

            # Видалення поганого яйця (як у класичному)
            if np.random.rand() >= self.p_detect:
                worst_idx = np.argmax(eggs)
                perturbation = delta * range_width * np.random.uniform(-1, 1, M)
                new_worst_nest = self.simple_bounds(nests[worst_idx] + perturbation)

                nests[worst_idx] = new_worst_nest
                eggs[worst_idx] = self.func(new_worst_nest)

            # Збереження історії
            best_history.append((best_nest.copy(), best_egg))

            # Формування списку всіх особин у форматі (позиція, значення)
            iteration_history = [(nests[i].copy(), eggs[i]) for i in range(self.n_nests)]
            all_history.append(iteration_history)

            if n % 100 == 0:
                print(f"Cuckoo Improved Algorithm: Iteration {n}, Best fitness = {best_egg}")

            if self.target is not None and abs(best_egg - self.target) < self.eps:
                print(f"Early stopping: target reached with best fitness = {best_egg}")
                break

        return best_nest, best_egg, best_history, all_history
