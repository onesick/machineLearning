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


flip_flop_problem_fit = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=mlrose_hiive.FlipFlop())


sixPeak_problem_fit = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=mlrose_hiive.SixPeaks())


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
    
    
    comparison_x = []
    comparison_y = []
    problems = []
    problems.append((problem_fit, 'TSP'))
    problems.append((flip_flop_problem_fit, 'Flip'))
    problems.append((sixPeak_problem_fit, 'SixPeak'))
    # rhc_fitness, rhc_curve = rhc_plot(sixPeak_problem_fit, "title")
    # print(rhc_fitness)
    # print(rhc_curve)
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
        comparison_x.append('MIMIC')
        comparison_y.append(mimic_fitness)

        legend = '{}'.format("sa")
        plt.plot(sa_curve[:,1], sa_curve[:,0], label=legend)
        plt.legend(loc='lower left')
        comparison_x.append('SA')
        comparison_y.append(sa_fitness)
        
        legend = '{}'.format("ga")
        plt.plot(ga_curve[:,1], ga_curve[:,0], label=legend)
        plt.legend(loc='lower left')
        comparison_x.append('GA')
        comparison_y.append(ga_fitness)

        legend = '{}'.format("rhc")
        plt.plot(rhc_curve[:,1], rhc_curve[:,0], label=legend)
        plt.legend(loc='lower left')
        comparison_x.append('RHC')
        comparison_y.append(rhc_fitness)

        plt.savefig(IMAGE_DIR + '{}_comparison_curve'.format(title))

        plt.figure()
        plt.ylabel("Fitness Score")
        plt.bar(comparison_x, comparison_y)
        plt.savefig(IMAGE_DIR + '{}_comparison_score'.format(title))

    
