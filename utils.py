import math
import numpy as np

# import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import os
import matplotlib.colors as mcolors

def fermi(focal, neighbor, K=0.3):
    return 1 / (1 + pow(math.e, (focal - neighbor) / K))

def calculate_fitness(population, payoff_matrix):
    size_x, size_y = population.shape
    fitnesses = np.zeros((size_x, size_y))

    for i in range(size_x):
        for j in range(size_y):
            strategy = population[i, j]
            total_payoff = 0
            neighbors = []
            if i > 0:
                neighbors.append((i - 1, j))
            if i < size_x - 1:
                neighbors.append((i + 1, j))
            if j > 0:
                neighbors.append((i, j - 1))
            if j < size_y - 1:
                neighbors.append((i, j + 1))

            for x, y in neighbors:
                neighbor_strategy = population[x, y]
                total_payoff += payoff_matrix[strategy][neighbor_strategy]

            fitnesses[i, j] = total_payoff

    return fitnesses

def calculate_fitness_pgg(population, r=3.0):
    """
    Evolution of commitment in the spatial public goods game through institutional incentives
    - https://www.sciencedirect.com/science/article/pii/S0096300324001188
    Evolution of Commitment and Level of Participation in Public Goods Games
    - https://userweb.fct.unl.pt/~lmp/publications/online-papers/com_part_pgg.pdf

    Focal user, the other four and the focal user form a group
    detail: ./pgg-lattice

    Payoff = (n_coop * r / group_size) - contribution_cost
    where:
        n_coop = total cooperators in group (including focal if C)
        group_size = 1 + number of neighbors (usually 5 on interior, less on edges)
        contribution_cost = 1 if cooperator, 0 if defector
        r = multiplication factor for public good

    """
    size_x, size_y = population.shape
    fitnesses = np.zeros((size_x, size_y))

    for i in range(size_x):
        for j in range(size_y):
            neighbors = []
            if i > 0:
                neighbors.append((i - 1, j))
            if i < size_x - 1:
                neighbors.append((i + 1, j))
            if j > 0:
                neighbors.append((i, j - 1))
            if j < size_y - 1:
                neighbors.append((i, j + 1))

            n_coop = sum(1 for x, y in neighbors if population[x, y] == 'C')
            if population[i, j] == 'C':
                n_coop += 1
                contribution_cost = 1
            else:
                contribution_cost = 0

            group_size = len(neighbors) + 1
            fitnesses[i, j] = (n_coop * r / group_size) - contribution_cost

    return fitnesses

def visualize_population(population):
    color_map = {'C': 'blue', 'D': 'red'}
    color_grid = np.vectorize(color_map.get)(population)

    plt.imshow(color_grid, interpolation='nearest')
    plt.axis('off')
    save_figure(plt, filename=f"population_{int(time.time())}.png")


def save_figure(plt, filename="figure.png"):
    # Ensure target directory exists before saving the figure
    dirpath = os.path.dirname(filename)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    plt.savefig(filename)
    plt.close()
