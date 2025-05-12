import numpy as np
# class BatAlgorithm:
#     def __init__(self, func, bounds, num_bats=20, max_iter=100, alpha=0.9, gamma=0.9, f_min=0.0, f_max=2.0, eps=1e-6, target=None):
#         self.func = func
#         self.bounds = np.array(bounds)
#         self.num_bats = num_bats
#         self.max_iter = max_iter
#         self.alpha = alpha
#         self.gamma = gamma
#         self.f_min = f_min
#         self.f_max = f_max
#         self.dim = len(bounds)
#         self.eps = eps
#         self.target = target

#     def initialize(self):
#         self.positions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.num_bats, self.dim))
#         self.velocities = np.zeros((self.num_bats, self.dim))
#         self.frequencies = np.zeros(self.num_bats)
#         self.pulse_rates = np.random.uniform(0, 1, self.num_bats)
#         self.loudness = np.random.uniform(1, 2, self.num_bats)
#         self.best_position = self.positions[np.argmin([self.func(pos) for pos in self.positions])]

#     def update_bat(self, i):
#         beta = np.random.rand()
#         self.frequencies[i] = self.f_min + (self.f_max - self.f_min) * beta
#         self.velocities[i] += (self.positions[i] - self.best_position) * self.frequencies[i]
#         new_position = self.positions[i] + self.velocities[i]
#         new_position = np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])

#         if np.random.rand() > self.pulse_rates[i]:
#             new_position = self.best_position + 0.001 * np.random.randn(self.dim)

#         if (np.random.rand() < self.loudness[i] and
#                 self.func(new_position) < self.func(self.positions[i])):
#             self.positions[i] = new_position
#             self.loudness[i] *= self.alpha
#             self.pulse_rates[i] = self.pulse_rates[i] * (1 - np.exp(-self.gamma))

#         if self.func(new_position) < self.func(self.best_position):
#             self.best_position = new_position

#     def optimize(self):
#         self.initialize()
#         best_history = []
#         all_history = []
#         for t in range(1, self.max_iter +1):
#             for i in range(self.num_bats):
#                 self.update_bat(i)
#             best_history.append((self.best_position.copy(), self.func(self.best_position)))
#             all_history.append(self.positions.copy())

#             if t % 100 == 0:
#                 print(f"Bat Algorithm: Iteration {t}, Best fitness = {self.func(self.best_position)}")

#             if self.target is not None and np.abs(self.func(self.best_position) - self.target) < self.eps:
#                 break
#         return self.best_position, self.func(self.best_position), best_history, all_history

class BatAlgorithm:
    def __init__(self, func, bounds, num_bats=50, max_iter=2000, alpha=0.95, gamma=0.8, f_min=0.0, f_max=2.0, eps=1e-6, target=None):
        self.func = func
        self.bounds = np.array(bounds)
        self.num_bats = num_bats
        self.max_iter = max_iter
        self.alpha = alpha
        self.gamma = gamma
        self.f_min = f_min
        self.f_max = f_max
        self.dim = len(bounds)
        self.eps = eps
        self.target = target

    def initialize(self):
        self.positions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.num_bats, self.dim))
        self.velocities = np.zeros((self.num_bats, self.dim))
        self.frequencies = np.zeros(self.num_bats)
        self.pulse_rates = np.random.uniform(0, 1, self.num_bats)
        self.loudness = np.random.uniform(1, 2, self.num_bats)
        fitness_values = np.array([self.func(pos) for pos in self.positions])
        self.best_position = self.positions[np.argmin(fitness_values)]
        self.best_fitness = np.min(fitness_values)

    def update_bat(self, i):
        beta = np.random.rand()
        self.frequencies[i] = self.f_min + (self.f_max - self.f_min) * beta
        self.velocities[i] += (self.positions[i] - self.best_position) * self.frequencies[i]
        
        # Нормалізація швидкості
        self.velocities[i] = np.clip(self.velocities[i], -1, 1)
        
        new_position = self.positions[i] + self.velocities[i]
        new_position = np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])

        # Локальне оновлення з випадковим шумом
        if np.random.rand() > self.pulse_rates[i]:
            new_position = self.best_position + 0.001 * np.random.randn(self.dim)

        new_fitness = self.func(new_position)

        # Адаптація гучності та пульсації
        if (np.random.rand() < self.loudness[i] and new_fitness < self.func(self.positions[i])):
            self.positions[i] = new_position
            self.loudness[i] *= self.alpha
            self.pulse_rates[i] = self.pulse_rates[i] * (1 - np.exp(-self.gamma))

        # Оновлення глобального найкращого рішення
        if new_fitness < self.best_fitness:
            self.best_position = new_position
            self.best_fitness = new_fitness

    def optimize(self):
        self.initialize()
        best_history = []
        all_history = []

        for t in range(1, self.max_iter + 1):
            iteration_history = []  # Збереження всіх особин на ітерації

            for i in range(self.num_bats):
                self.update_bat(i)
                # Додаємо кожну особину у форматі (позиція, значення)
                iteration_history.append((self.positions[i].copy(), self.func(self.positions[i])))

            # Зберігаємо найкращу позицію та значення
            best_history.append((self.best_position.copy(), self.best_fitness))
            # Зберігаємо всі позиції та значення на цій ітерації
            all_history.append(iteration_history)

            # Виведення прогресу кожні 100 ітерацій
            if t % 100 == 0:
                print(f"BatAlgorithm: Iteration {t}: Best fitness = {self.best_fitness}")

            # Умова завершення
            if self.target is not None and np.abs(self.best_fitness - self.target) < self.eps:
                print(f"Target reached at iteration {t}: Best fitness = {self.best_fitness}")
                break

        return self.best_position, self.best_fitness, best_history, all_history
