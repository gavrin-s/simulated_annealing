import numpy as np
from numpy.random import randint, random
import matplotlib.pyplot as plt
from copy import deepcopy


class SimulatedAnnealing():
    def __init__(self):
        self.length = 8
        self.initial_temperature = 100.0
        self.final_temperature = 0.5
        self.alpha = 0.99
        self.steps_per_chance = 100
        self.history_temperature = []
        self.history_best_energy = []
        self.history_accepted = []

        self.current_solution = self.initialize_solution()
        self.best_solution = self.initialize_solution()
        self.working_solution = self.initialize_solution()
        self.verbose = True
        self.solution = False

    def tweak_solution(self, solution):
        x = randint(self.length)
        y = randint(self.length)

        while x == y:
            y = randint(self.length)

        solution['board'][x], solution['board'][y] = solution['board'][y], solution['board'][x]
        solution['energy'] = self.compute_energy(solution)
        return solution

    def initialize_solution(self, solution=None):
        if solution is None:
            solution = {'board': np.array(range(self.length), dtype=int), 'energy': self.length * (self.length - 1)}

        #for _ in range(self.length):
        #    solution = self.tweak_solution(solution)
        return solution

    def compute_energy(self, solution):
        board = np.zeros((self.length, self.length))

        dxs = np.array([-1, 1, -1, 1])
        dys = np.array([-1, 1, 1, -1])

        for i in range(self.length):
            board[i, solution['board'][i]] = 1

        conflicts = 0

        for i in range(self.length):
            x = i
            y = solution['board'][i]
            for dx, dy in zip(dxs, dys):
                tempx = x
                tempy = y

                while True:
                    tempx += dx
                    tempy += dy
                    if (tempx < 0) or (self.length <= tempx) or (0 > tempy) or (tempy >= self.length):
                        break
                    if board[tempx, tempy] == 1:
                        conflicts += 1

        return conflicts

    def print_solution(self):
        board = np.zeros((self.length, self.length))

        for i in range(self.length):
            board[i, self.best_solution['board'][i]] = 1

        for string in board:
            print(''.join(['. ' if not s else 'Q' for s in string]))

    def fit(self):
        temperature = self.initial_temperature

        while temperature > self.final_temperature:
            accepted = 0

            for steps in range(self.steps_per_chance):
                use_new = 0

                self.working_solution = self.tweak_solution(self.working_solution)
                #self.working_solution['energy'] = self.compute_energy(self.working_solution)

                if self.working_solution['energy'] <= self.current_solution['energy']:
                    use_new = 1
                else:
                    test = random()
                    delta = self.working_solution['energy'] - self.current_solution['energy']
                    calc = np.exp(-delta / temperature)
                    if calc > test:
                        accepted += 1
                        use_new = 1

                if use_new:
                    self.current_solution = deepcopy(self.working_solution)

                    if self.current_solution['energy'] < self.best_solution['energy']:
                        self.best_solution = deepcopy(self.current_solution)
                        self.solution = True
                else:
                    self.working_solution = deepcopy(self.current_solution)
            temperature *= self.alpha

            self.history_accepted.append(accepted)
            self.history_best_energy.append(self.best_solution['energy'])
            self.history_temperature.append(temperature)

            if self.verbose:
                print('{}, {}, {}'.format(temperature, self.best_solution['energy'], accepted))

    def plot(self):
        plt.plot(self.history_temperature, label='Temperature')
        plt.plot(self.history_accepted, label='Accepted')
        plt.plot(self.history_best_energy, label='Best energy')
        plt.legend(loc='best', frameon=True)
        plt.xlim(0, len(self.history_accepted))
        plt.show()


if __name__ == "__main__":
    sim = SimulatedAnnealing()
    sim.fit()
    print(sim.best_solution)
    sim.print_solution()
    sim.plot()