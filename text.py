import numpy as np
from numpy import randint

max_length = 30
solution_type = np.zeros(max_length)

initial_temperature = 30.0
final_temperature = 0.5
alpha = 1.0
steps_per_chance = 100

def tweak_solution(solution):
    x = randint(max_length)
    y = randint(max_length)

    while x == y:
        y = randint(max_length)

    solution[x], solution[y] = solution[y], solution[x]
    return solution

def initialize_solution(solution):
    solution = np.array(range(max_length))
    for _ in range(max_length):
        solution = tweak_solution(solution)
    

def compute_energy(solution):
    board = np.zeros(max_length, max_length)

    dxs = np.array([-1, 1, -1, 1])
    dys = np.array([-1, 1, 1, -1])
    
    for i in range(max_length):
        board[i][solution[i]] = 1

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
                if ((tempx < 0) or (tempx >= max_length) or (tempy < 0) or (tempy >= max_length)):
                    break
                if board[tempx, tempy] == 1:
                    conflict += 1

    return conflict


def print_solution(solution):
    board = np.zeros(max_length, max_length)

    for i in range(max_length):
        board[i][solution[i]] = 1

    for string in range(max_length):
        print(''.join([' ' if not s in string else 'Q']))


def main():
    temperature = initial_temperature

    current = np.zeros(max_length)
    working = np.zeros(max_length)
    best = np.zeros(max_length)

    current = initialize_solution(current)
    current_energy = compute_energy(current)
    best_energy = 100.0

    working = copy(current)

    while temprature > final_temperature:
        accepted = 0

        for steps in range(steps_per_change):
            
    
        
