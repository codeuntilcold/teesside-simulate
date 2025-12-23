import numpy as np
from utils import fermi
from interference_agent import pop, neb
from utils import calculate_fitness, calculate_fitness_pgg, save_figure, visualize_population
import random
# import tkinter as tk
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

def update_population(population, fitnesses, K=0.3):
    new_population = population.copy()

    (size_x, size_y) = population.shape

    for i in range(size_x):
        for j in range(size_y):
            focal_strategy = population[i, j]
            focal_fitness = fitnesses[i, j]

            neighbors = []
            if i > 0:
                neighbors.append((i - 1, j))
            if i < size_x - 1:
                neighbors.append((i + 1, j))
            if j > 0:
                neighbors.append((i, j - 1))
            if j < size_y - 1:
                neighbors.append((i, j + 1))

            if not neighbors:
                continue

            neighbor_pos = random.choice(neighbors)
            neighbor_strategy = population[neighbor_pos]
            neighbor_fitness = fitnesses[neighbor_pos]

            prob = fermi(focal_fitness, neighbor_fitness, K)

            if random.random() < prob:
                new_population[i, j] = neighbor_strategy
    
    return new_population

def simulate_population(size_x=100, size_y=100, generations=50, a=1, b=0,
                        initial_cooperator_ratio=0.5, beta=1.8, payoff_matrix=None,
                        strategy='neb-i', pc=0.25, nc=4, theta=5.5, epsilon=4.5,
                        game_type='pd', r=3.0,
                        save_figures=False, show_final_population=False,
                        save_data=False, figure_path='fig', data_path='data'):
    if payoff_matrix is None:
        payoff_matrix = {
            'C': {'C': 1, 'D': 0},
            'D': {'C': beta, 'D': 0}
        }

    population = np.random.choice(['C', 'D'], size=(size_x, size_y), p=[initial_cooperator_ratio, 1 - initial_cooperator_ratio])
    history_frequency = []
    history_cost = []
    history_fitnesses = []
    history_social_welfare = []

    for gen in range(generations):
        cooperator_count = sum((sum(row == 'C') for row in population))
        if cooperator_count == 0 or cooperator_count == size_x * size_y:
            break

        history_frequency.append(cooperator_count)

        if game_type == 'pgg':
            fitnesses = calculate_fitness_pgg(population, r=r)
        else:
            fitnesses = calculate_fitness(population, payoff_matrix)
        cost = 0

        if strategy == 'pop':
            fitnesses, cost = pop(population, fitnesses, pc=int(pc * size_x * size_y), theta=theta * a, a=a)
        elif strategy == 'neb-i':
            fitnesses, cost = neb('i', population, fitnesses, nc=0, epsilon=epsilon * a, a=a)
        elif strategy == 'neb-ii':
            fitnesses, cost = neb('ii', population, fitnesses, nc=0, epsilon=epsilon * a, a=a)
        elif strategy == 'neb':
            fitnesses, cost = neb(None, population, fitnesses, nc, theta=theta * a, a=a)

        history_cost.append(cost)
        total_fitness = np.sum(fitnesses)
        history_fitnesses.append(total_fitness)
        history_social_welfare.append(total_fitness - cost)

        population = update_population(population, fitnesses, K=0.3)

    if save_figures:
        path = f'{figure_path}/{game_type}-a={a}-{strategy}'
        if strategy == 'neb':
            path += f'-{nc}-{theta}'
        if strategy == 'pop':
            path += f'-{pc}-{theta}'
        if strategy in ['neb-i', 'neb-ii']:
            path += f'-{epsilon}'
        if game_type == 'pgg':
            path += f'-r={r}'

        # Show history
        plt.ylim(0, 100)
        plt.xlim(0, generations)
        plt.plot(np.array(history_frequency) / (size_x * size_y) * 100, label='Cooperator Frequency')
        plt.xlabel('Generation')
        plt.ylabel('Number of Cooperators')
        plt.title('Cooperator Count Over Generations')
        save_figure(plt, filename=f'{path}_cooperator-history.png')

        # Show cost
        plt.figure()
        plt.ylim(0, max(history_cost) * 1.1)
        plt.xlim(0, generations)
        plt.plot(history_cost, label='Interference Cost')
        plt.xlabel('Generation')
        plt.ylabel('Cost')
        plt.title('Interference Cost Over Generations')
        save_figure(plt, filename=f'{path}_cost-history.png')

        # Show social welfare
        plt.figure()
        plt.ylim(min(history_social_welfare) * 0.9, max(history_social_welfare) * 1.1)
        plt.xlim(0, generations)
        plt.plot(history_social_welfare, label='Social Welfare')
        plt.xlabel('Generation')
        plt.ylabel('Social Welfare')
        plt.title('Social Welfare Over Generations')
        save_figure(plt, filename=f'{path}_social-welfare-history.png')

    if show_final_population:
        visualize_population(population)

    if save_data:
        save_data_path = f'{data_path}/{game_type}-a={a}-{strategy}'
        if strategy == 'neb':
            save_data_path += f'-{nc}-{theta}'
        elif strategy == 'pop':
            save_data_path += f'-{pc}-{theta}'
        elif strategy in ['neb-i', 'neb-ii']:
            save_data_path += f'-{epsilon}'
        if game_type == 'pgg':
            save_data_path += f'-r={r}'
        data = pd.DataFrame({
            'Cooperator Count': history_frequency,
            'Interference Cost': history_cost,
            'Social Welfare': history_social_welfare
        })
        # Ensure directory exists before writing CSV
        import os as _os
        _os.makedirs(_os.path.dirname(f'{save_data_path}_data.csv'), exist_ok=True)
        data.to_csv(f'{save_data_path}_data.csv', index_label='Generation')

    return population, history_frequency, history_fitnesses, history_cost, history_social_welfare
