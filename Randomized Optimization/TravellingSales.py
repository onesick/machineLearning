import mlrose_hiive
import numpy as np

import matplotlib.pyplot as plt

IMAGE_DIR = 'images/'

# Create list of city coordinates
coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]

# # Initialize fitness function object using coords_list
fitness_coords = mlrose_hiive.TravellingSales(coords = coords_list)

# Define optimization problem object. Instead of minimizing, we are maximizing travel distance
problem_fit = mlrose_hiive.TSPOpt(length = 8, fitness_fn = fitness_coords, maximize=True)




saSettings = []
saSettings.append({"schedule": 0.05})
saSettings.append({"schedule": 0.5})
saSettings.append({"schedule": 0.005})


def sa_plot(settings, problem_fit, title='title'):
    plt.figure()
    plt.xlabel("Number of iterations")
    plt.ylabel("Fitness")
    best_fit = 0
    best_curve = []
    for setup in settings:

        best_state, best_fitness, curv = mlrose_hiive.simulated_annealing(problem_fit, schedule = mlrose_hiive.ExpDecay(setup["schedule"]),
                                              max_attempts = 100, curve=True)
        legend = 'exp decay constant of {}'.format(setup["schedule"])
        
        plt.plot(curv[:,1], curv[:,0], label=legend)
        plt.legend(loc='lower left')
        if best_fitness > best_fit:
            best_fit = best_fitness
            best_curve = curv


    plt.savefig(IMAGE_DIR + '{}_sa'.format(title))

    return best_fit, best_curve

def ga_plot(problem_fit, title='title'):
    best_fit = 0
    best_curve = []
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
        if best_fitness > best_fit:
            best_fit = best_fitness
            best_curve = curv

    plt.savefig(IMAGE_DIR + '{}_ga'.format(title))
    return best_fit, best_curve

def rhc_plot(problem_fit, title='title'):
    plt.figure()
    plt.xlabel("Number of iterations")
    plt.ylabel("Fitness")
    best_state, best_fitness, curv = mlrose_hiive.random_hill_climb(problem_fit, 
                                              max_attempts = 100, curve=True)
    plt.plot(curv[:,1], curv[:,0])
    plt.savefig(IMAGE_DIR + '{}_rhc'.format(title))      

    return best_fitness, curv

def mimic_plot(problem_fit, title='title'):
    best_fit = 0
    best_curve = []
    plt.figure()
    plt.xlabel("Number of iterations")
    plt.ylabel("Fitness")
    keeps = (0.1, 0.3, 0.6, 0.9)
    for keep in keeps:
        best_state, best_fitness, curv = mlrose_hiive.mimic(problem_fit, keep_pct=keep, curve=True)
        legend = 'keep ratio of {}'.format(keep)
        plt.plot(curv[:,1], curv[:,0], label=legend)
        plt.legend(loc='lower left')
        if best_fitness > best_fit:
            best_fit = best_fitness
            best_curve = curv
    plt.savefig(IMAGE_DIR + '{}_MIMIC'.format(title))
    return best_fit, best_curve

if __name__ == '__main__':
    
    
    # //TODO: add all the curve and plots
    comparison_x = []
    comparison_y = []
    problems = []
    problems.append((problem_fit, 'TSP'))
    
    for (problem_fit, title) in problems:
        sa_fitness, sa_curve = sa_plot(saSettings, problem_fit, title)
        
        ga_fitness, ga_curve = ga_plot(problem_fit, title)    

        rhc_fitness, rhc_curve = rhc_plot(problem_fit, title)
        
        mimic_fitness, mimic_curve = mimic_plot(problem_fit, title)
        
        
        plt.figure()
        plt.xlabel("Number of iterations")
        plt.ylabel("Fitness")

        legend = '{}'.format("mimic")
        plt.plot(mimic_curve[:,1], mimic_curve[:,0], label=legend)
        plt.legend(loc='lower left')
        comparison.append(['MIMIC', mimic_curve])

        legend = '{}'.format("sa")
        plt.plot(sa_curve[:,1], sa_curve[:,0], label=legend)
        plt.legend(loc='lower left')
        comparison.append(['SA', sa_fitness])
        
        legend = '{}'.format("ga")
        plt.plot(ga_curve[:,1], ga_curve[:,0], label=legend)
        plt.legend(loc='lower left')
        comparison.append(['GA', ga_fitness])

        legend = '{}'.format("rhc")
        plt.plot(rhc_curve[:,1], rhc_curve[:,0], label=legend)
        plt.legend(loc='lower left')
        comparison.append(['RHC', rhc_curve])

        plt.savefig(IMAGE_DIR + '{}_comparison_curve'.format(title))

        print(comparison)
        plt.figure()
        plt.ylabel("Fitness Score")
        plt.plot(comparison[:,0], comparison[:,1])
        plt.savefig(IMAGE_DIR + '{}_comparison_score'.format(title))

    
#TODO: change max attempts size, and see where it calculates maximum