import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import libs.benchmark_functions as bf
import pandas as pd


func_classes = {
    0: bf.Rastrigin,        # 0,0 res:0
    1: bf.Ackley,           # 0,0 res:0
    2: bf.Griewank,         # 0,0 res:0
    3: bf.Schwefel,         # 420.9, 420.9 res:0
    4: bf.Michalewicz,      # 2.2, 1.57 res: −1.8013
    5: bf.StyblinskiTang,   # -2.9, -2.9 res: −39.166×n
    6: bf.EggHolder,        # 512, 404.23  res: −959.6407
    7: bf.Levy,             # 1, 1 res:0
    8: bf.Shubert,          # x, y res: −186.7309
    9: bf.DropWave,         # 0, 0 res: -1
    10: bf.Whitley,         # 1,1 res:1
    11: bf.Salomon,         # 0,0 res: 0
    12: bf.Alpine1,         # 0,0 res: 0
    13: bf.Trid,            # 2, 2 res: -2
    14: bf.Keane            # 0, 1.39325 res: 0.67366
}


def create_func_dict(D):
    return {
        func_id: func_classes[func_id](n_dimensions=D)
        for func_id in func_classes
    }


def clamp_boundaries(x, bounds):
    return np.clip(x, bounds[0], bounds[1])


def de_rand_1_bin(func, dim, bounds, population_size=30, F=0.8, CR=0.9, max_evals=1000):
    population = np.random.uniform(bounds[0], bounds[1], (population_size, dim))
    fitness = np.array([func(ind) for ind in population])
    eval_count = len(population)

    while eval_count < max_evals:
        for i in range(population_size):
            r1, r2, r3 = np.random.choice([j for j in range(population_size) if j != i], 3, replace=False)
            mutant = population[r1] + F * (population[r2] - population[r3])
            mutant = clamp_boundaries(mutant, bounds)

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


def de_best_1_bin(func, dim, bounds, population_size=30, F=0.5, CR=0.9, max_evals=1000):
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
            mutant = clamp_boundaries(mutant, bounds)

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


def pso(func, dim, bounds, swarm_size=30, w=0.7298, c1=1.49618, c2=1.49618, max_evals=1000):
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
            positions[i] = clamp_boundaries(positions[i], bounds)

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


def run_statistics():
    dim_list = [2, 10, 20]
    pop_dict = {2: 10, 10: 20, 20: 40}
    num_runs = 20
    results = []
    for D in dim_list:
        func_dict = create_func_dict(D)
        population_size = pop_dict[D]
        max_evals = 2000 * D

        for func_id in range(0, 15):
            function = func_dict[func_id]

            for run in range(num_runs):
                # 1) DE/rand/1/bin
                best_sol_rand, best_fit_rand = de_rand_1_bin(
                    func=function,
                    dim=D,
                    bounds=(-100, 100),
                    population_size=population_size,
                    F=0.8,
                    CR=0.9,
                    max_evals=max_evals
                )
                results.append([
                    D,
                    func_id,
                    run,
                    "DE/rand/1/bin",
                    best_sol_rand.tolist(),
                    best_fit_rand
                ])

                # 2) DE/best/1/bin
                best_sol_best, best_fit_best = de_best_1_bin(
                    func=function,
                    dim=D,
                    bounds=(-100, 100),
                    population_size=population_size,
                    F=0.5,
                    CR=0.9,
                    max_evals=max_evals
                )
                results.append([
                    D,
                    func_id,
                    run,
                    "DE/best/1/bin",
                    best_sol_best.tolist(),
                    best_fit_best
                ])

                # 3) PSO
                best_sol_pso, best_fit_pso = pso(
                    func=function,
                    dim=D,
                    bounds=(-100, 100),
                    swarm_size=population_size,
                    w=0.7298,
                    c1=1.49618,
                    c2=1.49618,
                    max_evals=max_evals
                )
                results.append([
                    D,
                    func_id,
                    run,
                    "PSO",
                    best_sol_pso.tolist(),
                    best_fit_pso
                ])

        with open("all_results.csv", mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Dimension", "FunctionID", "Run", "Algorithm", "BestSolution", "BestFitness"])

            for row in results:
                writer.writerow(row)

func_names = {
    0: "Rastrigin",
    1: "Ackley",
    2: "Griewank",
    3: "Schwefel",
    4: "Michalewicz",
    5: "StyblinskiTang",
    6: "EggHolder",
    7: "Levy",
    8: "Shubert",
    9: "DropWave",
    10: "Whitley",
    11: "Salomon",
    12: "Alpine1",
    13: "Trid",
    14: "Keane"
}

def process_data(data_file, dimension):
    df = pd.read_csv(data_file)

    df_grouped = df.groupby(["Dimension", "FunctionID", "Algorithm"], as_index=False).agg(meanFitness=("BestFitness", "mean"))

    df_dim = df_grouped[df_grouped["Dimension"] == dimension]
    df_dim_pivot = df_dim.pivot(index="FunctionID", columns="Algorithm", values="meanFitness")
    df_dim_pivot.index = df_dim_pivot.index.map(func_names)
    df_dim_rank = df_dim_pivot.rank(axis=1, method='dense')
    df_dim_final = df_dim_pivot.copy()
    for col in df_dim_final.columns:
        means = df_dim_pivot[col]
        ranks = df_dim_rank[col]
        combined = means.round(4).astype(str) + " (" + ranks.astype(int).astype(str) + ")"
        df_dim_final[col] = combined

    html_table = df_dim_final.to_html()
    print(html_table)
    return df_dim_final


if __name__ == '__main__':

    # for generating maps
    # for i in range(0, 15):
    #     function = func_dict[i]
    #     # function.show()
    #     # function.show(asHeatMap=True)

    #run_statistics()

    df = process_data('all_results.csv', 20)
    #print(df)



