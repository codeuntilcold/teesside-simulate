from interference_agent import neb, pop
from utils import fermi, calculate_fitness, visualize_population, save_figure
from update import update_population, simulate_population

import os
import numpy as np
import pandas as pd
# import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import itertools
import argparse
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, *args, **kwargs):
        return iterable

# # Fix the random seed for reproducibility
# np.random.seed(42)

def __main__():
    # # Run single simulation
    # final_population, history_frequency, history_cost, history_social_welfare = simulate_population(
    #     strategy='neb', nc=4, theta=4.2, epsilon=0, save_figures=True, 
    #     show_final_population=False, save_data=True, figure_path='fig/neb', data_path='data/neb'
    # )

    # # Generate POP plots & data
    # pc = [0.25, 0.5, 0.75, 1.0]
    # for threshold in pc:
    #     final_population, history_frequency, history_cost, history_social_welfare = simulate_population(
    #         strategy='pop', pc=threshold, theta=4.5, epsilon=0, save_figures=True, 
    #         show_final_population=False, save_data=True, figure_path='fig/pop', data_path='data/pop'
    #     )

    # # Generate NEB plots & data
    # nc = [1, 2, 3, 4]
    # for threshold in nc:
    #     final_population, history_frequency, history_cost, history_social_welfare = simulate_population(
    #         strategy='neb', nc=threshold, theta=5.5, epsilon=0, save_figures=True, 
    #         show_final_population=False, save_data=True, figure_path='fig/neb', data_path='data/neb'
    #     )

    parser = argparse.ArgumentParser(description='Simulate population dynamics.')
    parser.add_argument('--strategy', type=str, default='pop', help='Interference strategy to use (pop, neb, neb-i, neb-ii)')
    parser.add_argument('--nc', type=int, default=4, help='Number of cooperators for NEB strategy')
    parser.add_argument('--pc', type=float, default=0.75, help='Cooperation threshold for POP strategy')
    parser.add_argument('--game-type', type=str, default='pd', choices=['pd', 'pgg'], help='Game type: pd (Prisoner\'s Dilemma) or pgg (Public Goods Game)')
    parser.add_argument('--b', type=float, default=1.8, help='Temptation payoff (beta) for Prisoner\'s Dilemma')
    parser.add_argument('--r', type=float, default=3.0, help='Multiplication factor for Public Goods Game')
    parser.add_argument('--seed-start', type=int, default=0, help='Starting seed for simulations')
    parser.add_argument('--seed-end', type=int, default=10, help='Ending seed for simulations')

    args = parser.parse_args()
    strategy = args.strategy
    pc = args.pc
    nc = args.nc
    game_type = args.game_type
    b = args.b
    r = args.r
    seed_start = args.seed_start
    seed_end = args.seed_end

    # Generate theta run for POP
    # Run with different seed
    seeds = range(seed_start, seed_end)
    for seed in tqdm(seeds, desc='Seeds'):
        np.random.seed(seed)
        theta_list = np.arange(4.0, 5.05, 0.1)
        # a_list = np.arange(1e-9, 1 + 1e-8, 0.1)
        
        # total = len(theta_list) * len(a_list)
        res_freq = []
        res_social_welfare = []
        res_cost = []
        res_fitness = []
        # for theta, a in tqdm(itertools.product(theta_list, a_list), total=total, desc=f'Seed {seed}'):           
        a = 1
        for theta in tqdm(theta_list, desc=f'Seed {seed}'):
            final_population, history_frequency, history_fitness, history_cost, history_social_welfare = simulate_population(
                a=a, generations=250, strategy=strategy, nc=nc, pc=pc, theta=theta,
                beta=b, game_type=game_type, r=r,
                save_figures=False, show_final_population=False, save_data=False)

            res_freq.append((seed, theta, a, [int(x) for x in history_frequency]))
            res_social_welfare.append((seed, theta, a, [float(x) for x in history_social_welfare]))
            res_cost.append((seed, theta, a, [float(x) for x in history_cost]))
            res_fitness.append((seed, theta, a, [float(x) for x in history_fitness]))

        # Use strategy and game-type specific directory
        if strategy == 'pop':
            if game_type == 'pgg':
                data_dir = f'data/{strategy}_theta_pgg_{r=}'
            else:
                data_dir = f'data/{strategy}_theta_pd_{b=}'
            param_str = f'{pc=}'
        elif strategy == 'neb':
            if game_type == 'pgg':
                data_dir = f'data/{strategy}_theta_pgg_{r=}'
            else:
                data_dir = f'data/{strategy}_theta_pd_{b=}'
            param_str = f'{nc=}'
        else:
            if game_type == 'pgg':
                data_dir = f'data/{strategy}_theta_pgg_{r=}'
            else:
                data_dir = f'data/{strategy}_theta_pd_{b=}'
            param_str = f'{pc=}'

        os.makedirs(data_dir, exist_ok=True)

        df = pd.DataFrame(res_freq, columns=['Seed', 'Theta', 'A', 'Cooperator_Frequency'])
        df.to_csv(f'{data_dir}/seed_{seed}_{param_str}_cooperator_frequency.csv', index=False)

        df = pd.DataFrame(res_social_welfare, columns=['Seed', 'Theta', 'A', 'Social_Welfare'])
        df.to_csv(f'{data_dir}/seed_{seed}_{param_str}_social_welfare.csv', index=False)

        df = pd.DataFrame(res_cost, columns=['Seed', 'Theta', 'A', 'Cost'])
        df.to_csv(f'{data_dir}/seed_{seed}_{param_str}_cost.csv', index=False)

        df = pd.DataFrame(res_fitness, columns=['Seed', 'Theta', 'A', 'Payoff'])
        df.to_csv(f'{data_dir}/seed_{seed}_{param_str}_population_payoff.csv', index=False)
    
    
    # # Compare different neb strategies
    # neb_4 = []
    # neb_3 = []
    # neb_i = []
    # neb_ii = []
    # beta = 1.8

    # for epsilon in np.arange(0, 2.01, 0.1):
    #     final_pop, freq, cost, social_welfare = simulate_population(strategy='neb', nc=4, theta=4 * beta - 3 + epsilon, save_figures=False)
    #     neb_4.append(sum(cost))

    #     final_pop, freq, cost, social_welfare = simulate_population(strategy='neb', nc=3, theta=4 * beta - 2 + epsilon, save_figures=False)
    #     neb_3.append(sum(cost))

    #     final_pop, freq, cost, social_welfare = simulate_population(strategy='neb-i', theta=0, epsilon=epsilon, save_figures=False)
    #     neb_i.append(sum(cost))

    #     final_pop, freq, cost, social_welfare = simulate_population(strategy='neb-ii', theta=0, epsilon=epsilon, save_figures=False)
    #     neb_ii.append(sum(cost))

    # # Plot comparison
    # plt.plot(np.arange(0, 2.01, 0.1), neb_4, label='neb-4')
    # plt.plot(np.arange(0, 2.01, 0.1), neb_3, label='neb-3')
    # plt.plot(np.arange(0, 2.01, 0.1), neb_i, label='neb-i')
    # plt.plot(np.arange(0, 2.01, 0.1), neb_ii, label='neb-ii')
    # plt.xlabel('Epsilon')
    # plt.ylabel('Total Interference Cost')
    # plt.title('Interference Cost Comparison Across Strategies')
    # plt.legend()
    # save_figure(plt, filename='neb_strategy_comparison.png')

    # # Save comparison data
    # df = pd.DataFrame({
    #     'Epsilon': np.arange(0, 2.01, 0.1),
    #     'neb-4': neb_4,
    #     'neb-3': neb_3,
    #     'neb-i': neb_i,
    #     'neb-ii': neb_ii
    # })
    # df.to_csv('neb_strategy_comparison_data.csv', index=False)

    # Heatmap theta over POP and NEB


__main__()
