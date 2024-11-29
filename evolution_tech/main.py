import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import benchmark_functions as bf

# funkce:
# 1-10 from benchmark_functions library
# levy not found - implemented
# shubert not found - implemented
# drop-wave not found - implemented
# cross-in-tray not found - implemented
# bohachevsky not found - implemented


func_dict = {
    0: bf.Rastrigin(),
    1: bf.Ackley(),
    2: bf.Griewank(),
    3: bf.DeJong5(),
    4: bf.Schwefel(),
    5: bf.Michalewicz(),
    6: bf.StyblinskiTang(),
    7: bf.EggHolder(),
    8: bf.GoldsteinAndPrice(),
    9: bf.PichenyGoldsteinAndPrice(),
    10: bf.Levy(),
    11: bf.Shubert(),
    12: bf.DropWave(),
    13: bf.CrossInTray(),
    14: bf.Bohachevsky()
}


# Kontrola hranic metodou odrazu (reflection)
def reflect_boundaries(x, bounds):
    for i in range(len(x)):
        if x[i] < bounds[0]:
            x[i] = bounds[0] + (bounds[0] - x[i])
        elif x[i] > bounds[1]:
            x[i] = bounds[1] - (x[i] - bounds[1])
    return x


def de_rand_1_bin(func, dim, bounds, population_size=30, F=0.8, CR=0.9, max_evals=1000):
    population = np.random.uniform(bounds[0], bounds[1], (population_size, dim))
    fitness = np.array([func(ind) for ind in population])
    eval_count = len(population)

    while eval_count < max_evals:
        for i in range(population_size):
            r1, r2, r3 = np.random.choice([j for j in range(population_size) if j != i], 3, replace=False)
            mutant = population[r1] + F * (population[r2] - population[r3])
            mutant = reflect_boundaries(mutant, bounds)

            trial = np.copy(population[i])
            for j in range(dim):
                if np.random.rand() < CR:
                    trial[j] = mutant[j]

            trial_fitness = func(trial)
            eval_count += 1
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness

        if eval_count >= max_evals:
            break

    best_index = np.argmin(fitness)
    return population[best_index], fitness[best_index]


def de_best_1_bin(func, dim, bounds, population_size=30, F=0.8, CR=0.9, max_evals=1000):
    population = np.random.uniform(bounds[0], bounds[1], (population_size, dim))
    fitness = np.array([func(ind) for ind in population])
    eval_count = len(population)

    while eval_count < max_evals:
        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        for i in range(population_size):
            r1, r2 = np.random.choice([j for j in range(population_size) if j != i and j != best_index], 2,
                                      replace=False)
            mutant = best_individual + F * (population[r1] - population[r2])
            mutant = reflect_boundaries(mutant, bounds)

            trial = np.copy(population[i])
            for j in range(dim):
                if np.random.rand() < CR:
                    trial[j] = mutant[j]

            trial_fitness = func(trial)
            eval_count += 1
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness

        if eval_count >= max_evals:
            break

    best_index = np.argmin(fitness)
    return population[best_index], fitness[best_index]


def pso(func, dim, bounds, swarm_size=30, w=0.5, c1=1.5, c2=1.5, max_evals=1000):
    positions = np.random.uniform(bounds[0], bounds[1], (swarm_size, dim))
    velocities = np.zeros((swarm_size, dim))
    personal_best_positions = np.copy(positions)
    personal_best_fitness = np.array([func(ind) for ind in positions])
    global_best_position = positions[np.argmin(personal_best_fitness)]
    global_best_fitness = np.min(personal_best_fitness)
    eval_count = len(positions)

    while eval_count < max_evals:
        for i in range(swarm_size):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = (
                    w * velocities[i]
                    + c1 * r1 * (personal_best_positions[i] - positions[i])
                    + c2 * r2 * (global_best_position - positions[i])
            )
            positions[i] += velocities[i]
            positions[i] = reflect_boundaries(positions[i], bounds)

            fitness = func(positions[i])
            eval_count += 1

            if fitness < personal_best_fitness[i]:
                personal_best_positions[i] = positions[i]
                personal_best_fitness[i] = fitness

            if fitness < global_best_fitness:
                global_best_position = positions[i]
                global_best_fitness = fitness

        if eval_count >= max_evals:
            break
    return global_best_position, global_best_fitness


if __name__ == '__main__':
    for i in range(0, 15):
        function = func_dict[i]
        function.show()
        function.show(asHeatMap=True)

        # DE/rand/1/bin
        # best_solution_de_rand, best_fitness_de_rand = de_rand_1_bin(function, dim=2, bounds=(-100, 100), max_evals=5000)
        # print("DE/rand/1/bin Best Solution:", best_solution_de_rand, "Fitness:", best_fitness_de_rand)
        #
        # # DE/best/1/bin
        # best_solution_de_best, best_fitness_de_best = de_rand_1_bin(function, dim=2, bounds=(-100, 100), max_evals=5000)
        # print("DE/rand/1/bin Best Solution:", best_solution_de_best, "Fitness:", best_fitness_de_best)
        #
        # # PSO
        # best_solution_pso, best_fitness_pso = pso(function, dim=2, bounds=(-100, 100), max_evals=5000)
        # print("PSO Best Solution:", best_solution_pso, "Fitness:", best_fitness_pso)