import mlrose_hiive
import numpy as np

import matplotlib.pyplot as plt

IMAGE_DIR = 'images/'

schedule = mlrose_hiive.ExpDecay()

# Create list of city coordinates
coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]

# # Initialize fitness function object using coords_list
fitness_coords = mlrose_hiive.TravellingSales(coords = coords_list)

# Define optimization problem object. Instead of minimizing, we are maximizing travel distance
problem_fit = mlrose_hiive.TSPOpt(length = 8, fitness_fn = fitness_coords, maximize=True)


# Solve problem using the genetic algorithm
best_state, best_fitness, curv = mlrose_hiive.genetic_alg(problem_fit, mutation_prob = 0.2,
                                              max_attempts = 100, curve=True)
print(best_state)
print(best_fitness)
# print(curv)


best_state, best_fitness, curv = mlrose_hiive.random_hill_climb(problem_fit, 
                                              max_attempts = 100, curve=True)
print(best_state)
print(best_fitness)



# print(curv[:,0])

saSettings = []
saSettings.append({"schedule": 0.05})
saSettings.append({"schedule": 0.5})
saSettings.append({"schedule": 0.005})


def sa_plot(settings, problem_fit):
    plt.figure()
    plt.xlabel("Number of iterations")
    plt.ylabel("Fitness")
    
    for setup in settings:

        best_state, best_fitness, curv = mlrose_hiive.simulated_annealing(problem_fit, schedule = mlrose_hiive.ExpDecay(setup["schedule"]),
                                              max_attempts = 100, curve=True)
        legend = 'exp decay constant of {}'.format(setup["schedule"])
        
        plt.plot(curv[:,1], curv[:,0], label=legend)
        plt.legend(loc='lower left')


    plt.show() #TODO: change it to save

def ga_plot(problem_fit):
    plt.figure()
    plt.xlabel("Number of iterations")
    plt.ylabel("Fitness")
    mutations = (0.1, 0.3, 0.6, 0.9)
    for mut in mutations:
        # Have to keep max_attempts to 10. If I keep attempts to 100, iteration goes to 20000, but it starts flattening out at 3000
        best_state, best_fitness, curv = mlrose_hiive.genetic_alg(problem_fit, mutation_prob = mut,
                                                max_attempts = 10, curve=True, max_iters=3000)
        legend = 'mutation probability of {}'.format(mut)
        plt.plot(curv[:,1], curv[:,0], label=legend)
        plt.legend(loc='lower left')


    plt.show() #TODO: change it to save

if __name__ == '__main__':
    # sa_plot(saSettings, problem_fit=problem_fit)
    ga_plot(problem_fit)



#TODO: change max attempts size, and see where it calculates maximum