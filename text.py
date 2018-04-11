import numpy as np
from numpy.random import randint, random
import matplotlib.pyplot as plt

max_length = 15
solution_type = np.zeros(max_length)

initial_temperature = 100.0
final_temperature = 0.2
alpha = 0.99
steps_per_chance = 100

t = []
e = []
a = []


def tweak_solution(solution):
    x = randint(max_length)
    y = randint(max_length)

    while x == y:
        y = randint(max_length)

    solution[x], solution[y] = solution[y], solution[x]
    return solution


def initialize_solution(solution=None):
    if solution is None:
        solution = np.array(range(max_length))

    for _ in range(max_length):
        solution = tweak_solution(solution)
    return solution
    

def compute_energy(solution):
    board = np.zeros((max_length, max_length))

    dxs = np.array([-1, 1, -1, 1])
    dys = np.array([-1, 1, 1, -1])
    
    for i in range(max_length):
        board[i, solution[i]] = 1

    conflicts = 0

    for i in range(max_length):
        x = i
        y = solution[i]
        for dx, dy in zip(dxs, dys):
            tempx = x
            tempy = y

            while True:
                tempx += dx
                tempy += dy
                if (tempx < 0) or (max_length <= tempx) or (0 > tempy) or (tempy >= max_length):
                    break
                if board[tempx, tempy] == 1:
                    conflicts += 1

    return conflicts


def print_solution(solution):
    board = np.zeros((max_length, max_length))

    for i in range(max_length):
        board[i, solution[i]] = 1

    for string in board:
        print(''.join(['. ' if not s else 'Q' for s in string]))


def main():
    temperature = initial_temperature

    sol = 0

    current = np.zeros(max_length, dtype=int)
    working = np.zeros(max_length, dtype=int)
    best = np.zeros(max_length, dtype=int)

    current = initialize_solution()
    current_energy = compute_energy(current)
    best_energy = 100.0

    working = current.copy()
    working_energy = current_energy

    while temperature > final_temperature:
        accepted = 0

        for steps in range(steps_per_chance):
            use_new = 0

            working = tweak_solution(working)
            working_energy = compute_energy(working)

            if working_energy <= current_energy:
                use_new = 1
            else:
                test = random()
                delta = working_energy - current_energy
                calc = np.exp(-delta / temperature)
                if calc > test:
                    accepted += 1
                    use_new = 1

            if use_new:
                use_new = 0
                current = working.copy()
                current_energy = working_energy

                if current_energy < best_energy:
                    best = current.copy()
                    best_energy = current_energy
                    sol = 1
                else:
                    working = current.copy()
                    working_energy = current_energy
        temperature *= alpha
        print('{}, {}, {}'.format(temperature, best_energy, accepted))
        t.append(temperature)
        e.append(best_energy)
        a.append(accepted)
        print('Best energy = {}'.format(best_energy))

    if sol:
        print_solution(best)
    return best

if __name__ == "__main__":
    print(main())
    #energy = compute_energy(np.array([2, 0, 6, 4, 7, 1, 3, 5]))
    #print(energy)
    #solution = initialize_solution()
    #print_solution(solution)
    plt.plot(t, label='Temperature')
    plt.plot(e, label='Best Energy')
    plt.plot(a, label='Accepted')
    plt.legend(loc='best', frameon=True)
    plt.show()